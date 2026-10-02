use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use gilrs::{Axis, Button, EventType, Gilrs};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use serialport::SerialPort;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const MAX_RANGE_M: f32 = 12.0;
const HISTORY: usize = 3;
const HIDDEN_1: usize = 128;
const HIDDEN_2: usize = 64;
const SERVO_CENTER: f32 = 0.5;
const SERVO_RANGE: f32 = 0.48;

#[derive(Parser)]
#[command(
    name = "behavioral-cloning-lidar",
    about = "Collecte et apprentissage par clonage comportemental avec LiDAR D500"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Collecte scan LiDAR, servo et duty en conduite manuelle.
    Collect {
        #[arg(long, default_value = "/dev/ttyTHS1")]
        lidar_port: String,
        #[arg(long, default_value = "/dev/ttyACM0")]
        vesc_port: String,
        #[arg(long, default_value = "dataset_lidar/driving_log.csv")]
        csv: PathBuf,
    },
    /// Entraîne le MLP à partir d'un CSV du collecteur LiDAR.
    Train {
        #[arg(long, default_value = "dataset_lidar/driving_log.csv")]
        csv: PathBuf,
        #[arg(long, default_value = "behavioral_cloning_rust/model_lidar.json")]
        model_out: PathBuf,
        #[arg(long, default_value_t = 150)]
        epochs: usize,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f32,
        #[arg(long, default_value_t = 12.0)]
        max_range: f32,
    },
    /// Prédit le servo à partir d'un scan JSON (180 distances en mètres).
    Predict {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        scan: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Collect {
            lidar_port,
            vesc_port,
            csv,
        } => collect(&lidar_port, &vesc_port, &csv),
        Command::Train {
            csv,
            model_out,
            epochs,
            learning_rate,
            max_range,
        } => train(&csv, &model_out, epochs, learning_rate, max_range),
        Command::Predict { model, scan } => predict(&model, &scan),
    }
}

// VESC UART commands, matching the controller implementation used by autopilot.
struct Vesc {
    port: Box<dyn SerialPort>,
}
impl Vesc {
    fn connect(path: &str) -> Result<Self> {
        let port = serialport::new(path, 115_200)
            .timeout(Duration::from_secs(1))
            .open()
            .with_context(|| format!("ouverture VESC {path}"))?;
        Ok(Self { port })
    }
    fn send_i32(&mut self, command: u8, value: i32) -> Result<()> {
        let mut payload = vec![command];
        payload.extend_from_slice(&value.to_be_bytes());
        let mut packet = vec![2, payload.len() as u8];
        packet.extend_from_slice(&payload);
        let crc = crc16(&payload);
        packet.extend_from_slice(&crc.to_be_bytes());
        packet.push(3);
        self.port.write_all(&packet)?;
        self.port.flush()?;
        Ok(())
    }
    fn servo(&mut self, value: f32) -> Result<()> {
        let scaled = ((value.clamp(0., 1.) * 1000.) as i16).to_be_bytes();
        let payload = [12, scaled[0], scaled[1]];
        let mut packet = vec![2, payload.len() as u8];
        packet.extend_from_slice(&payload);
        packet.extend_from_slice(&crc16(&payload).to_be_bytes());
        packet.push(3);
        self.port.write_all(&packet)?;
        self.port.flush()?;
        Ok(())
    }
    fn duty(&mut self, value: f32) -> Result<()> {
        self.send_i32(5, (value.clamp(-0.1, 0.1) * 100_000.) as i32)
    }
    fn stop(&mut self) {
        let _ = self.duty(0.);
        let _ = self.servo(SERVO_CENTER);
    }
}
impl Drop for Vesc {
    fn drop(&mut self) {
        self.stop();
    }
}
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for b in bytes {
        crc ^= (*b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
        }
    }
    crc
}

fn collect(lidar_port: &str, vesc_port: &str, path: &Path) -> Result<()> {
    if !Path::new(lidar_port).exists() {
        eprintln!("[INFO] ouverture LiDAR {lidar_port} (le périphérique doit être présent)");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let exists = path.exists() && path.metadata()?.len() > 0;
    if exists {
        let mut existing = csv::Reader::from_path(path)?;
        let header = existing.headers()?.clone();
        if header.iter().collect::<Vec<_>>() != ["timestamp", "servo", "duty", "lidar"] {
            bail!(
                "le CSV {} n'a pas l'en-tête timestamp,servo,duty,lidar; choisis un nouveau fichier",
                path.display()
            );
        }
    }
    let file = File::options().create(true).append(true).open(path)?;
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(file);
    if !exists {
        writer.write_record(["timestamp", "servo", "duty", "lidar"])?;
        writer.flush()?;
    }

    let mut gamepad = Gilrs::new().context("initialisation de la manette (gilrs)")?;
    if gamepad.gamepads().next().is_none() {
        bail!("aucune manette détectée");
    }
    let mut vesc = Vesc::connect(vesc_port)?;
    let running = Arc::new(AtomicBool::new(true));
    let stop_flag = running.clone();
    ctrlc::set_handler(move || stop_flag.store(false, Ordering::SeqCst))?;
    let mut recording = false;
    let mut axes = (0.0f32, 0.0f32, 0.0f32, 0.0f32); // forward, reverse, steer, emergency
    println!("Manette : RT/LT accélérer/freiner, joystick gauche tourner, A REC/pause, LB arrêt. CSV : {}", path.display());

    // read_scans supplies a complete 180-bin D500 scan on each callback.
    let scan_result = lidarcontrol::read_scans(lidar_port, 230_400, |_, scan| {
        if !running.load(Ordering::SeqCst) {
            vesc.stop();
            bail!("arrêt demandé");
        }
        while let Some(event) = gamepad.next_event() {
            match event.event {
                EventType::ButtonPressed(Button::South, _) => {
                    if !recording {
                        recording = true;
                        println!("[REC] démarré");
                    } else {
                        recording = false;
                        println!("[REC] pause");
                    }
                }
                EventType::ButtonPressed(Button::LeftTrigger2, _) => {
                    recording = false;
                    running.store(false, Ordering::SeqCst);
                    println!("[STOP] arrêt d'urgence demandé");
                }
                EventType::AxisChanged(Axis::RightZ, value, _) => axes.0 = trigger(value),
                EventType::AxisChanged(Axis::LeftZ, value, _) => axes.1 = trigger(value),
                EventType::AxisChanged(Axis::LeftStickX, value, _) => axes.2 = value,
                _ => {}
            }
        }
        for (_, pad) in gamepad.gamepads() {
            if pad.is_pressed(Button::LeftTrigger2) {
                axes.3 = 1.0;
            } else {
                axes.3 = 0.0;
            }
        }
        if axes.3 > 0.5 {
            recording = false;
            running.store(false, Ordering::SeqCst);
            vesc.stop();
            bail!("arrêt LB");
        }
        let steer = deadzone(axes.2);
        let servo = (SERVO_CENTER + steer * SERVO_RANGE).clamp(0.0, 1.0);
        let duty = (deadzone(axes.0) - deadzone(axes.1)).clamp(-1.0, 1.0) * 0.10;
        vesc.servo(servo)?;
        vesc.duty(duty)?;
        if recording {
            let ranges: Vec<f32> = scan
                .ranges_m
                .iter()
                .map(|v| v.clamp(0.0, MAX_RANGE_M))
                .collect();
            writer.write_record([
                format!("{}", scan.timestamp_unix_ms),
                format!("{servo:.4}"),
                format!("{duty:.4}"),
                serde_json::to_string(&ranges)?,
            ])?;
            writer.flush()?;
        }
        print!(
            "\r{} servo={servo:.2} duty={duty:.3}   ",
            if recording { "REC" } else { "MANUEL" }
        );
        std::io::stdout().flush()?;
        Ok(())
    });
    vesc.stop();
    match scan_result {
        Ok(()) => Ok(()),
        Err(_e) if !running.load(Ordering::SeqCst) => {
            println!("\nCollecte terminée.");
            Ok(())
        }
        Err(e) => Err(e.context("acquisition LiDAR interrompue")),
    }
}
fn trigger(v: f32) -> f32 {
    ((v + 1.0) * 0.5).clamp(0.0, 1.0)
}
fn deadzone(v: f32) -> f32 {
    if v.abs() < 0.08 {
        0.0
    } else {
        v.signum() * ((v.abs() - 0.08) / 0.92)
    }
}

#[derive(Clone)]
struct Sample {
    scan: Vec<f32>,
    servo: f32,
    duty: f32,
}
fn load_data(path: &Path, max_range: f32) -> Result<Vec<Sample>> {
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("lecture CSV {}", path.display()))?;
    let headers = reader.headers()?.clone();
    let servo_i = headers
        .iter()
        .position(|x| x == "servo")
        .context("colonne servo absente")?;
    let duty_i = headers
        .iter()
        .position(|x| x == "duty")
        .context("colonne duty absente")?;
    let scan_i = ["lidar", "lidar_data", "scan", "ranges"]
        .iter()
        .find_map(|name| headers.iter().position(|x| x == *name))
        .context("colonne lidar (JSON) absente")?;
    let mut out = Vec::new();
    for (row, result) in reader.records().enumerate() {
        let record = result?;
        let servo: f32 = match record.get(servo_i).unwrap_or("").parse() {
            Ok(v) if (0.0..=1.0).contains(&v) => v,
            _ => continue,
        };
        let duty: f32 = match record.get(duty_i).unwrap_or("").parse() {
            Ok(v) if v.abs() > 0.01 => v,
            _ => continue,
        };
        let raw: Vec<f32> = serde_json::from_str(record.get(scan_i).unwrap_or(""))
            .with_context(|| format!("scan JSON invalide ligne {}", row + 2))?;
        if raw.is_empty() {
            continue;
        }
        let scan = raw
            .into_iter()
            .map(|v| {
                if v.is_finite() {
                    v.clamp(0.0, max_range) / max_range
                } else {
                    1.0
                }
            })
            .collect();
        out.push(Sample { scan, servo, duty });
    }
    if out.len() < 2 {
        bail!("il faut au moins deux lignes valides avec duty actif");
    }
    let rays = out[0].scan.len();
    if out.iter().any(|s| s.scan.len() != rays) {
        bail!("tous les scans doivent avoir le même nombre de rayons");
    }
    Ok(out)
}

#[derive(Serialize, Deserialize)]
struct Model {
    ray_count: usize,
    history_scans: usize,
    max_range_m: f32,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
}
impl Model {
    fn new(input: usize, max_range_m: f32) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        use rand::Rng;
        let init = |n: usize, fan: usize, rng: &mut ChaCha8Rng| {
            (0..n)
                .map(|_| rng.gen_range(-1.0..1.0) * (2.0 / fan as f32).sqrt())
                .collect()
        };
        Self {
            ray_count: input / HISTORY,
            history_scans: HISTORY,
            max_range_m,
            w1: init(HIDDEN_1 * input, input, &mut rng),
            b1: vec![0.; HIDDEN_1],
            w2: init(HIDDEN_2 * HIDDEN_1, HIDDEN_1, &mut rng),
            b2: vec![0.; HIDDEN_2],
            w3: init(HIDDEN_2, HIDDEN_2, &mut rng),
            b3: vec![0.],
        }
    }
    fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let mut h1 = vec![0.; HIDDEN_1];
        for i in 0..HIDDEN_1 {
            let r = &self.w1[i * x.len()..(i + 1) * x.len()];
            h1[i] = (r.iter().zip(x).map(|(a, b)| a * b).sum::<f32>() + self.b1[i]).max(0.);
        }
        let mut h2 = vec![0.; HIDDEN_2];
        for i in 0..HIDDEN_2 {
            let r = &self.w2[i * HIDDEN_1..(i + 1) * HIDDEN_1];
            h2[i] = (r.iter().zip(&h1).map(|(a, b)| a * b).sum::<f32>() + self.b2[i]).max(0.);
        }
        let y = self.w3.iter().zip(&h2).map(|(a, b)| a * b).sum::<f32>() + self.b3[0];
        (h1, h2, y)
    }
    fn predict(&self, history: &[Vec<f32>]) -> f32 {
        let x: Vec<f32> = history.iter().flatten().copied().collect();
        self.forward(&x).2
    }
}

fn history_for(samples: &[Sample], idx: usize) -> Vec<f32> {
    let start = idx.saturating_sub(HISTORY - 1);
    let mut out = Vec::with_capacity(samples[0].scan.len() * HISTORY);
    for _ in 0..HISTORY - (idx - start + 1) {
        out.extend_from_slice(&samples[0].scan);
    }
    for s in &samples[start..=idx] {
        out.extend_from_slice(&s.scan);
    }
    out
}
fn train(csv_path: &Path, out: &Path, epochs: usize, lr: f32, max_range: f32) -> Result<()> {
    if max_range <= 0. || lr <= 0. || epochs == 0 {
        bail!("epochs, learning-rate et max-range doivent être positifs");
    }
    let samples = load_data(csv_path, max_range)?;
    let split = ((samples.len() as f32) * 0.8).floor() as usize;
    if split == 0 || split >= samples.len() {
        bail!("dataset trop petit pour train/validation");
    }
    let input = samples[0].scan.len() * HISTORY;
    let mut model = Model::new(input, max_range);
    println!(
        "[DATA] {} train | {} validation | {} rayons, historique {} scans",
        split,
        samples.len() - split,
        input / HISTORY,
        HISTORY
    );
    let mut best = f32::INFINITY;
    let mut stale = 0;
    let patience = 15;
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let mut step = 0u64;
    // Adam state mirrors the six parameter arrays.
    let mut adam = ModelAdam::new(&model);
    for epoch in 1..=epochs {
        let mut indices: Vec<usize> = (0..split).collect();
        indices.shuffle(&mut rng);
        let mut train_loss = 0.;
        for i in indices {
            let x = history_for(&samples, i);
            step += 1;
            train_loss += model.learn_adam(&x, samples[i].servo, lr, step, &mut adam);
        }
        train_loss /= split as f32;
        let mut val = 0.;
        for i in split..samples.len() {
            let x = history_for(&samples, i);
            val += (model.predict(&[x]).clamp(0., 1.) - samples[i].servo).powi(2);
        }
        val /= (samples.len() - split) as f32;
        println!("Epoch {epoch:03}/{epochs} | train MSE {train_loss:.6} | val MSE {val:.6}");
        if val < best {
            best = val;
            stale = 0;
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            let tmp = out.with_extension("json.tmp");
            fs::write(&tmp, serde_json::to_vec_pretty(&model)?)?;
            fs::rename(tmp, out)?;
        } else {
            stale += 1;
            if stale >= patience {
                println!("Early stopping.");
                break;
            }
        }
    }
    println!(
        "[OK] meilleur modèle validation MSE={best:.6} : {}",
        out.display()
    );
    Ok(())
}

struct ModelAdam {
    states: Vec<(Vec<f32>, Vec<f32>)>,
}
impl ModelAdam {
    fn new(m: &Model) -> Self {
        Self {
            states: [
                m.w1.len(),
                m.b1.len(),
                m.w2.len(),
                m.b2.len(),
                m.w3.len(),
                m.b3.len(),
            ]
            .iter()
            .map(|n| (vec![0.; *n], vec![0.; *n]))
            .collect(),
        }
    }
    fn update(&mut self, n: usize, t: u64, lr: f32, p: &mut [f32], g: &[f32]) {
        let (m, v) = &mut self.states[n];
        for i in 0..p.len() {
            m[i] = 0.9 * m[i] + 0.1 * g[i];
            v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
            let mh = m[i] / (1. - 0.9f32.powf(t as f32));
            let vh = v[i] / (1. - 0.999f32.powf(t as f32));
            p[i] -= lr * mh / (vh.sqrt() + 1e-8);
        }
    }
}
impl Model {
    fn learn_adam(&mut self, x: &[f32], target: f32, lr: f32, t: u64, opt: &mut ModelAdam) -> f32 {
        let (h1, h2, y) = self.forward(x);
        let d3 = 2. * (y - target);
        let d2: Vec<f32> = (0..HIDDEN_2)
            .map(|i| if h2[i] > 0. { d3 * self.w3[i] } else { 0. })
            .collect();
        let d1: Vec<f32> = (0..HIDDEN_1)
            .map(|i| {
                if h1[i] > 0. {
                    (0..HIDDEN_2)
                        .map(|j| d2[j] * self.w2[j * HIDDEN_1 + i])
                        .sum()
                } else {
                    0.
                }
            })
            .collect();
        let mut g1 = vec![0.; self.w1.len()];
        for i in 0..HIDDEN_1 {
            for j in 0..x.len() {
                g1[i * x.len() + j] = d1[i] * x[j];
            }
        }
        let mut g2 = vec![0.; self.w2.len()];
        for i in 0..HIDDEN_2 {
            for j in 0..HIDDEN_1 {
                g2[i * HIDDEN_1 + j] = d2[i] * h1[j];
            }
        }
        let g3: Vec<f32> = h2.iter().map(|v| d3 * v).collect();
        opt.update(0, t, lr, &mut self.w1, &g1);
        opt.update(1, t, lr, &mut self.b1, &d1);
        opt.update(2, t, lr, &mut self.w2, &g2);
        opt.update(3, t, lr, &mut self.b2, &d2);
        opt.update(4, t, lr, &mut self.w3, &g3);
        opt.update(5, t, lr, &mut self.b3, &[d3]);
        (y - target).powi(2)
    }
}

fn predict(model_path: &Path, scan_path: &Path) -> Result<()> {
    let model: Model = serde_json::from_slice(
        &fs::read(model_path)
            .with_context(|| format!("lecture modèle {}", model_path.display()))?,
    )?;
    let scan: Vec<f32> = serde_json::from_slice(&fs::read(scan_path)?)?;
    if scan.len() != model.ray_count {
        bail!(
            "le scan contient {} rayons, modèle attendu {}",
            scan.len(),
            model.ray_count
        );
    }
    let normalized: Vec<f32> = scan
        .iter()
        .map(|v| {
            if v.is_finite() {
                v.clamp(0., model.max_range_m) / model.max_range_m
            } else {
                1.
            }
        })
        .collect();
    let history = vec![normalized.clone(); model.history_scans];
    let servo = model.predict(&history).clamp(0., 1.);
    println!("servo={servo:.4}");
    Ok(())
}
