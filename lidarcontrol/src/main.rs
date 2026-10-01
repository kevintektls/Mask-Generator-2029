use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lidarcontrol::camera::{record_camera_frames, CameraFrameRecord};
use lidarcontrol::localization::{localize_scans, LocalizationPose, OccupancyMap};
use lidarcontrol::mapping::{build_map, write_pose_log, PoseRecord};
use lidarcontrol::navigation::{map_view, plan_path, pure_pursuit, Waypoint};
use lidarcontrol::{read_scans, ScanRecord};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Read;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

#[cfg(target_os = "macos")]
use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs};

const PYTHON_EXECUTABLE: &str = "python3.8";

#[derive(Debug, Parser)]
#[command(
    name = "lidarcontrol",
    about = "D500 acquisition, replay, and 2D mapping"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start camera preview and the LiDAR/VESC viewer.
    Start {
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        /// Receive gamepad commands over a localhost TCP socket (for an SSH tunnel).
        #[arg(long)]
        remote_control: bool,
        /// Address used by the remote gamepad command listener.
        #[arg(long, default_value = "127.0.0.1:5010")]
        control_bind: String,
    },
    /// Read a local Mac gamepad and send its state through an SSH tunnel.
    #[cfg(target_os = "macos")]
    RemoteClient {
        #[arg(long, default_value_t = 5010)]
        port: u16,
    },
    /// Record timestamped 180-bin scans from the D500 to JSON Lines.
    Record {
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        #[arg(long)]
        port: Option<String>,
        #[arg(long)]
        baud: Option<u32>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Replay and validate a recorded JSON Lines scan file.
    Replay {
        #[arg(long)]
        input: PathBuf,
        /// Delay between scans, scaled by this factor (0 means as fast as possible).
        #[arg(long, default_value_t = 1.0)]
        speed: f64,
    },
    /// Build a 2D occupancy map from a recorded scan JSONL file.
    Map {
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "lidarcontrol/maps/floor")]
        output: PathBuf,
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        #[arg(long)]
        resolution: Option<f32>,
    },
    /// Record timestamped OAK-D Lite mono frames through the DepthAI bridge.
    CameraRecord {
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        #[arg(long)]
        addr: Option<String>,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Pair each mapped LiDAR pose with its nearest timestamped camera frame.
    Sync {
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        #[arg(long)]
        poses: PathBuf,
        #[arg(long)]
        frames: PathBuf,
        #[arg(long, default_value = "lidarcontrol/data/sensor_pairs.jsonl")]
        output: PathBuf,
        #[arg(long)]
        tolerance_ms: Option<u64>,
    },
    /// Localize recorded LiDAR scans against a saved occupancy map.
    Localize {
        #[arg(long, default_value = "lidarcontrol/maps/floor.yaml")]
        map: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, default_value = "lidarcontrol/data/localization.jsonl")]
        output: PathBuf,
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
        #[arg(long, default_value_t = 0.0)]
        initial_x: f32,
        #[arg(long, default_value_t = 0.0)]
        initial_y: f32,
        #[arg(long, default_value_t = 0.0)]
        initial_yaw_deg: f32,
        #[arg(long)]
        min_confidence: Option<f32>,
    },
    /// Plan a collision-margin A* route through known free map cells.
    Plan {
        #[arg(long, default_value = "lidarcontrol/maps/floor.yaml")]
        map: PathBuf,
        #[arg(long, num_args = 2)]
        start: Vec<f32>,
        #[arg(long, num_args = 2)]
        goal: Vec<f32>,
        #[arg(long, default_value_t = 0.25)]
        margin: f32,
    },
    /// Start the local map-click UI. This is planning-only and never drives motors.
    Ui {
        #[arg(long, default_value = "lidarcontrol/maps/floor.yaml")]
        map: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8765")]
        bind: String,
        #[arg(long, default_value_t = 0.25)]
        margin: f32,
    },
    /// Replay localization and evaluate tracking/safety commands offline.
    Simulate {
        #[arg(long, default_value = "lidarcontrol/maps/floor.yaml")]
        map: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long, num_args = 2)]
        start: Vec<f32>,
        #[arg(long, num_args = 2)]
        goal: Vec<f32>,
        #[arg(long, default_value_t = 0.0)]
        initial_yaw_deg: f32,
        #[arg(long, default_value = "lidarcontrol/config.toml")]
        config: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    lidar_port: String,
    lidar_baud: u32,
    scan_log: PathBuf,
    manual_scan_csv: PathBuf,
    map_resolution_m: f32,
    camera_bridge_addr: String,
    camera_output: PathBuf,
    camera_fps: u32,
    camera_startup_timeout_s: u64,
    camera_sync_threshold_ms: f32,
    vesc_port: String,
    camera_sync_tolerance_ms: u64,
    localization_min_confidence: f32,
    safety_margin_m: f32,
    obstacle_stop_distance_m: f32,
    simulation_speed_mps: f32,
    simulation_wheelbase_m: Option<f32>,
    simulation_max_steering_rad: Option<f32>,
    simulation_servo_center: Option<f32>,
    simulation_servo_left: Option<f32>,
    simulation_servo_right: Option<f32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lidar_port: "/dev/ttyTHS1".into(),
            lidar_baud: 230_400,
            scan_log: "lidarcontrol/data/scan.jsonl".into(),
            manual_scan_csv: "lidarcontrol/data/floor_scan.csv".into(),
            map_resolution_m: 0.05,
            camera_bridge_addr: "127.0.0.1:9010".into(),
            camera_output: "lidarcontrol/data/camera".into(),
            camera_fps: 24,
            camera_startup_timeout_s: 60,
            camera_sync_threshold_ms: 5.0,
            vesc_port: "/dev/ttyACM0".into(),
            camera_sync_tolerance_ms: 80,
            localization_min_confidence: 0.30,
            safety_margin_m: 0.25,
            obstacle_stop_distance_m: 0.30,
            simulation_speed_mps: 0.20,
            simulation_wheelbase_m: None,
            simulation_max_steering_rad: None,
            simulation_servo_center: None,
            simulation_servo_left: None,
            simulation_servo_right: None,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lidarcontrol=info".into()),
        )
        .init();

    match Cli::parse().command {
        Command::Start {
            config,
            remote_control,
            control_bind,
        } => {
            let config_text = fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            start_preview_and_controller(&config, remote_control, &control_bind)
        }
        #[cfg(target_os = "macos")]
        Command::RemoteClient { port } => run_remote_client(port),
        Command::Record {
            config,
            port,
            baud,
            output,
        } => {
            let config_text = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            let port = port.unwrap_or(config.lidar_port);
            let baud = baud.unwrap_or(config.lidar_baud);
            let output = output.unwrap_or(config.scan_log);
            record(&port, baud, &output)
        }
        Command::Replay { input, speed } => replay(&input, speed),
        Command::Map {
            input,
            output,
            config,
            resolution,
        } => {
            let config_text = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            map_scans(
                &input,
                &output,
                resolution.unwrap_or(config.map_resolution_m),
            )
        }
        Command::CameraRecord {
            config,
            addr,
            output,
        } => {
            let config_text = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            let addr = addr.unwrap_or(config.camera_bridge_addr);
            let output = output.unwrap_or(config.camera_output);
            record_camera_frames(&addr, &output)
        }
        Command::Sync {
            config,
            poses,
            frames,
            output,
            tolerance_ms,
        } => {
            let config_text = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            sync_sensor_logs(
                &poses,
                &frames,
                &output,
                tolerance_ms.unwrap_or(config.camera_sync_tolerance_ms),
            )
        }
        Command::Localize {
            map,
            input,
            output,
            config,
            initial_x,
            initial_y,
            initial_yaw_deg,
            min_confidence,
        } => {
            let config_text = std::fs::read_to_string(&config)
                .with_context(|| format!("reading config {}", config.display()))?;
            let config: Config = toml::from_str(&config_text)
                .with_context(|| format!("parsing config {}", config.display()))?;
            localize_log(
                &map,
                &input,
                &output,
                LocalizationPose {
                    x_m: initial_x,
                    y_m: initial_y,
                    yaw_rad: initial_yaw_deg.to_radians(),
                },
                min_confidence.unwrap_or(config.localization_min_confidence),
            )
        }
        Command::Plan {
            map,
            start,
            goal,
            margin,
        } => {
            let occupancy = OccupancyMap::load_yaml(&map)?;
            let path = plan_path(
                &occupancy,
                point_arg(&start, "start")?,
                point_arg(&goal, "goal")?,
                margin,
            )?;
            println!("{}", serde_json::to_string_pretty(&path)?);
            Ok(())
        }
        Command::Ui { map, bind, margin } => serve_ui(&map, &bind, margin),
        Command::Simulate {
            map,
            input,
            start,
            goal,
            initial_yaw_deg,
            config,
        } => {
            let cfg: Config = toml::from_str(
                &fs::read_to_string(&config)
                    .with_context(|| format!("reading config {}", config.display()))?,
            )?;
            simulate_route(
                &map,
                &input,
                point_arg(&start, "start")?,
                point_arg(&goal, "goal")?,
                initial_yaw_deg,
                &cfg,
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn remote_trigger_value(gamepad: &gilrs::Gamepad, button: Button, axis: Axis) -> f32 {
    let value = if let Some(data) = gamepad.button_data(button) {
        data.value()
    } else {
        let axis_value = gamepad.value(axis);
        if axis_value < 0.0 {
            (axis_value + 1.0) * 0.5
        } else {
            axis_value
        }
    };
    value.clamp(0.0, 1.0)
}

#[cfg(target_os = "macos")]
fn connected_gamepad(gilrs: &Gilrs) -> Option<GamepadId> {
    gilrs
        .gamepads()
        .find_map(|(id, gamepad)| gamepad.is_connected().then_some(id))
}

#[cfg(target_os = "macos")]
fn run_remote_client(port: u16) -> Result<()> {
    let mut gilrs = Gilrs::new()
        .map_err(|error| anyhow::anyhow!("initialisation de la manette sur le Mac: {error:?}"))?;
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);
    ctrlc::set_handler(move || shutdown_flag.store(true, Ordering::SeqCst))
        .context("installation du gestionnaire Ctrl-C")?;

    println!("Recherche d'une manette (F710 : sélecteur sur D pour macOS)…");
    let gamepad_id = loop {
        while let Some(Event { .. }) = gilrs.next_event() {}
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        if let Some(id) = connected_gamepad(&gilrs) {
            break id;
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    };
    println!("Manette détectée : {}", gilrs.gamepad(gamepad_id).name());

    let address = format!("127.0.0.1:{port}");
    let mut stream = loop {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        match TcpStream::connect(&address) {
            Ok(stream) => break stream,
            Err(error) => {
                eprintln!("Tunnel SSH pas encore prêt ({error}); nouvel essai…");
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    };
    stream.set_nodelay(true)?;
    println!("Contrôle actif via le tunnel SSH. Ctrl-C arrête la voiture.");

    let result = (|| -> Result<()> {
        loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            while let Some(Event { id, event, .. }) = gilrs.next_event() {
                if id == gamepad_id && matches!(event, EventType::Disconnected) {
                    println!("Manette déconnectée : arrêt de la voiture.");
                    write!(stream, "{{\"type\":\"stop\"}}\n")?;
                    stream.flush()?;
                    return Ok(());
                }
            }
            let gamepad = gilrs.gamepad(gamepad_id);
            if !gamepad.is_connected() {
                write!(stream, "{{\"type\":\"stop\"}}\n")?;
                stream.flush()?;
                break;
            }
            let rt = (remote_trigger_value(&gamepad, Button::RightTrigger2, Axis::RightZ) * 255.0)
                .round() as u8;
            let lt = (remote_trigger_value(&gamepad, Button::LeftTrigger2, Axis::LeftZ) * 255.0)
                .round() as u8;
            let lx = (gamepad.value(Axis::LeftStickX).clamp(-1.0, 1.0) * 32767.0).round() as i16;
            let a = gamepad.is_pressed(Button::South);
            let lb = gamepad.is_pressed(Button::LeftTrigger);
            writeln!(
                stream,
                "{{\"rt\":{rt},\"lt\":{lt},\"lx\":{lx},\"a\":{a},\"lb\":{lb}}}"
            )?;
            stream.flush()?;
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if shutdown.load(Ordering::SeqCst) {
            write!(stream, "{{\"type\":\"stop\"}}\n")?;
            stream.flush()?;
        }
        Ok(())
    })();
    result
}

fn point_arg(values: &[f32], name: &str) -> Result<Waypoint> {
    if values.len() != 2 || values.iter().any(|value| !value.is_finite()) {
        anyhow::bail!("--{name} requires two finite metre values: X Y");
    }
    Ok(Waypoint {
        x_m: values[0],
        y_m: values[1],
    })
}

fn start_preview_and_controller(
    config: &Config,
    remote_control: bool,
    control_bind: &str,
) -> Result<()> {
    if remote_control
        && !control_bind.starts_with("127.0.0.1:")
        && !control_bind.starts_with("localhost:")
    {
        anyhow::bail!("remote gamepad control must bind to localhost");
    }
    if !(1..=24).contains(&config.camera_fps) {
        anyhow::bail!("camera_fps must be in [1, 24]");
    }
    if !(10..=300).contains(&config.camera_startup_timeout_s) {
        anyhow::bail!("camera_startup_timeout_s must be in [10, 300]");
    }
    if !config.camera_sync_threshold_ms.is_finite()
        || !(0.5..=20.0).contains(&config.camera_sync_threshold_ms)
    {
        anyhow::bail!("camera_sync_threshold_ms must be in [0.5, 20]");
    }
    let stopping = Arc::new(AtomicBool::new(false));
    {
        let stopping = Arc::clone(&stopping);
        ctrlc::set_handler(move || stopping.store(true, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("lidarcontrol must be inside the repository root")?
        .to_path_buf();
    let bridge_script = repo_root.join("lidarcontrol/tools/oak_bridge.py");
    let controller_script = repo_root.join("scripts/Behavioral_Cloning_Lidar.py");
    if !bridge_script.is_file() || !controller_script.is_file() {
        anyhow::bail!("camera bridge or existing LiDAR/gamepad controller script is missing from the repository");
    }
    check_manual_controller_dependencies(&controller_script, &repo_root, remote_control)?;

    let mut camera = spawn_camera_bridge(&bridge_script, &repo_root, config)?;
    info!("starting camera; waiting for its first frame before launching LiDAR/gamepad control");

    info!(
        timeout_s = config.camera_startup_timeout_s,
        "waiting for the first OAK-D frame"
    );
    let ready_deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(config.camera_startup_timeout_s);
    while std::time::Instant::now() < ready_deadline {
        if stopping.load(Ordering::SeqCst) {
            stop_child(&mut camera)?;
            info!("startup cancelled with Ctrl-C");
            return Ok(());
        }
        if let Some(status) = camera.try_wait()? {
            if stopping.load(Ordering::SeqCst) {
                return Ok(());
            }
            anyhow::bail!("camera bridge exited before becoming ready ({status})");
        }
        if camera_frame_ready() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if stopping.load(Ordering::SeqCst) {
        stop_child(&mut camera)?;
        info!("startup cancelled with Ctrl-C");
        return Ok(());
    }
    if !camera_frame_ready() {
        if let Some(status) = camera.try_wait()? {
            anyhow::bail!("camera bridge exited without producing a preview frame ({status}); see its Python error above");
        }
        stop_child(&mut camera)?;
        anyhow::bail!("OAK-D produced no preview frame within {} seconds; LiDAR/VESC controller was not started", config.camera_startup_timeout_s);
    }

    let mut controller = match ProcessCommand::new(PYTHON_EXECUTABLE)
        .arg(&controller_script)
        .arg("--preview")
        .arg("--csv")
        .arg(&config.manual_scan_csv)
        .arg("--lidar-port")
        .arg(&config.lidar_port)
        .arg("--vesc-port")
        .arg(&config.vesc_port)
        .args(if remote_control {
            vec!["--remote-control", "--control-bind", control_bind]
        } else {
            Vec::new()
        })
        .current_dir(&repo_root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("starting existing LiDAR/gamepad/VESC controller with python3.8")
    {
        Ok(child) => child,
        Err(error) => {
            stop_child(&mut camera)?;
            return Err(error);
        }
    };

    println!("Preview combinée : http://<ip-jetson>:5001/");
    if remote_control {
        println!("Contrôle manette distant : {control_bind} (tunnel SSH requis)");
        println!("Manette : RT avancer · LT reculer · joystick gauche direction · A enregistrement/pause · LB arrêt");
    } else {
        println!("Manette : RT avancer · LT reculer · joystick gauche direction · A enregistrement/pause · LB arrêt");
    }
    println!("Ctrl-C arrête les deux processus.");
    loop {
        if stopping.load(Ordering::SeqCst) {
            stop_child(&mut controller)?;
            stop_child(&mut camera)?;
            info!("camera and manual control stopped");
            return Ok(());
        }
        if let Some(status) = controller.try_wait()? {
            stop_child(&mut camera)?;
            if status.success() {
                return Ok(());
            }
            anyhow::bail!(
                "LiDAR/gamepad/VESC controller exited with {status}; camera bridge stopped"
            );
        }
        if let Some(status) = camera.try_wait()? {
            warn!(%status, "camera bridge stopped; retrying in 3 seconds while manual control stays up");
            std::thread::sleep(std::time::Duration::from_secs(3));
            camera = spawn_camera_bridge(&bridge_script, &repo_root, config)?;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn spawn_camera_bridge(
    bridge_script: &std::path::Path,
    repo_root: &std::path::Path,
    config: &Config,
) -> Result<Child> {
    ProcessCommand::new(PYTHON_EXECUTABLE)
        .arg(bridge_script)
        .arg("--fps")
        .arg(config.camera_fps.to_string())
        .arg("--sync-threshold-ms")
        .arg(config.camera_sync_threshold_ms.to_string())
        .current_dir(repo_root)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .context("starting OAK-D camera bridge with python3.8")
}

fn camera_frame_ready() -> bool {
    use std::io::Write;
    use std::net::SocketAddr;
    let address: SocketAddr = "127.0.0.1:9011".parse().expect("static socket address");
    let Ok(mut stream) =
        TcpStream::connect_timeout(&address, std::time::Duration::from_millis(100))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(250)));
    if stream
        .write_all(b"GET /status HTTP/1.0\r\nHost: localhost\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = Vec::with_capacity(256);
    if stream.read_to_end(&mut response).is_err() {
        return false;
    }
    (response.starts_with(b"HTTP/1.0 200") || response.starts_with(b"HTTP/1.1 200"))
        && response
            .windows(b"\"ready\":true".len())
            .any(|window| window == b"\"ready\":true")
}

fn check_manual_controller_dependencies(
    script: &PathBuf,
    repo_root: &PathBuf,
    remote_control: bool,
) -> Result<()> {
    let mut command = ProcessCommand::new(PYTHON_EXECUTABLE);
    command
        .arg(script)
        .arg("--check-dependencies")
        .current_dir(repo_root);
    if remote_control {
        command.arg("--remote-control");
    }
    let output = command
        .output()
        .context("checking Python dependencies for the manual controller")?;
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stdout));
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        anyhow::bail!("manual control dependencies are missing. Install them for Python 3.8 with: python3.8 -m pip install --user -r requirements.txt");
    }
    Ok(())
}

fn stop_child(child: &mut Child) -> Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    #[cfg(unix)]
    unsafe {
        libc::kill(child.id() as libc::pid_t, libc::SIGINT);
    }
    #[cfg(not(unix))]
    child.kill()?;

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while std::time::Instant::now() < deadline {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    child.kill()?;
    let _ = child.wait()?;
    Ok(())
}

#[derive(Deserialize)]
struct PlanRequest {
    start: Waypoint,
    goal: Waypoint,
}

fn serve_ui(map_path: &PathBuf, bind: &str, margin_m: f32) -> Result<()> {
    let listener = TcpListener::bind(bind).with_context(|| format!("binding UI at {bind}"))?;
    info!(
        address = bind,
        "planning UI ready (planning only; motor output disabled)"
    );
    println!("Ouvrir http://{bind}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(error) = handle_ui_request(stream, map_path, margin_m) {
                    warn!(%error, "UI request failed");
                }
            }
            Err(error) => warn!(%error, "UI connection failed"),
        }
    }
    Ok(())
}

fn handle_ui_request(mut stream: TcpStream, map_path: &PathBuf, margin_m: f32) -> Result<()> {
    stream.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
    let mut request = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end;
    loop {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            anyhow::bail!("client closed before request headers");
        }
        request.extend_from_slice(&chunk[..count]);
        if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
            header_end = index + 4;
            break;
        }
        if request.len() > 16_384 {
            anyhow::bail!("HTTP headers too large");
        }
    }
    let header = std::str::from_utf8(&request[..header_end])?.to_owned();
    let first = header.lines().next().unwrap_or("").to_owned();
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let route = parts.next().unwrap_or("/").to_owned();
    let content_length = header
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    if content_length > 16_384 {
        return write_http(
            &mut stream,
            "413 Payload Too Large",
            "application/json",
            br#"{"error":"body too large"}"#,
        );
    }
    while request.len() - header_end < content_length {
        let count = stream.read(&mut chunk)?;
        if count == 0 {
            anyhow::bail!("client closed during request body");
        }
        request.extend_from_slice(&chunk[..count]);
    }
    match (method.as_str(), route.as_str()) {
        ("GET", "/") => write_http(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            include_bytes!("ui.html"),
        ),
        ("GET", "/api/map") => {
            let map = OccupancyMap::load_yaml(map_path)?;
            let body = serde_json::to_vec(&map_view(&map))?;
            write_http(&mut stream, "200 OK", "application/json", &body)
        }
        ("POST", "/api/plan") => {
            let result = (|| -> Result<_> {
                let body: PlanRequest =
                    serde_json::from_slice(&request[header_end..header_end + content_length])?;
                let map = OccupancyMap::load_yaml(map_path)?;
                plan_path(&map, body.start, body.goal, margin_m)
            })();
            match result {
                Ok(path) => write_http(
                    &mut stream,
                    "200 OK",
                    "application/json",
                    &serde_json::to_vec(&serde_json::json!({"path": path}))?,
                ),
                Err(error) => write_http(
                    &mut stream,
                    "400 Bad Request",
                    "application/json",
                    &serde_json::to_vec(&serde_json::json!({"error": error.to_string()}))?,
                ),
            }
        }
        _ => write_http(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found",
        ),
    }
}

fn write_http(stream: &mut TcpStream, status: &str, content_type: &str, body: &[u8]) -> Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: null\r\n\r\n", body.len())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn simulate_route(
    map_path: &PathBuf,
    input: &PathBuf,
    start: Waypoint,
    goal: Waypoint,
    yaw_deg: f32,
    config: &Config,
) -> Result<()> {
    let required = [
        config.simulation_wheelbase_m,
        config.simulation_max_steering_rad,
        config.simulation_servo_center,
        config.simulation_servo_left,
        config.simulation_servo_right,
    ];
    if required.iter().any(Option::is_none) {
        anyhow::bail!("simulation path tracking needs explicit simulation_wheelbase_m, simulation_max_steering_rad and servo center/left/right calibration values in config.toml; no guessed vehicle geometry is used");
    }
    let wheelbase = config.simulation_wheelbase_m.unwrap();
    let max_steer = config.simulation_max_steering_rad.unwrap();
    let servo_center = config.simulation_servo_center.unwrap();
    let servo_left = config.simulation_servo_left.unwrap();
    let servo_right = config.simulation_servo_right.unwrap();
    let map = OccupancyMap::load_yaml(map_path)?;
    let path = plan_path(&map, start, goal, config.safety_margin_m)?;
    let scans = load_scans(input)?;
    let seed = LocalizationPose {
        x_m: start.x_m,
        y_m: start.y_m,
        yaw_rad: yaw_deg.to_radians(),
    };
    let records = localize_scans(&map, &scans, seed, config.localization_min_confidence)?;
    let mut stopped = false;
    let mut previous_scan_ns = None;
    for (scan, pose) in scans.iter().zip(records.iter()) {
        let nearest_front = scan.ranges_m[75..105]
            .iter()
            .copied()
            .filter(|r| *r > 0.0 && *r < lidarcontrol::MAX_RANGE_M)
            .reduce(f32::min);
        let stale = previous_scan_ns
            .is_some_and(|previous: u128| scan.monotonic_ns.saturating_sub(previous) > 500_000_000);
        let obstacle = nearest_front.is_some_and(|range| range <= config.obstacle_stop_distance_m);
        let stop = pose.stop || stale || obstacle;
        if stop {
            println!(
                "SIM STOP scan={} reason={}{}{}",
                pose.scan_index,
                if pose.stop { "localization " } else { "" },
                if stale { "stale_scan " } else { "" },
                if obstacle { "front_obstacle" } else { "" }
            );
            stopped = true;
            break;
        }
        previous_scan_ns = Some(scan.monotonic_ns);
        let command = pure_pursuit(
            pose.pose,
            &path,
            0.40,
            wheelbase,
            max_steer,
            servo_center,
            servo_left,
            servo_right,
            config.simulation_speed_mps,
        )?;
        println!("SIM scan={} pose=({:.2},{:.2},{:.1}°) confidence={:.2} servo={:.3} speed={:.2}m/s front_m={:.2}",
            pose.scan_index, pose.pose.x_m, pose.pose.y_m, pose.pose.yaw_rad.to_degrees(), pose.confidence,
            command.servo, command.speed_mps, nearest_front.unwrap_or(lidarcontrol::MAX_RANGE_M));
    }
    if !stopped {
        info!(
            scans = records.len(),
            route_points = path.len(),
            "offline navigation replay complete; no actuator was connected"
        );
    }
    Ok(())
}

fn record(port: &str, baud: u32, output: &PathBuf) -> Result<()> {
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let writer = Arc::new(std::sync::Mutex::new(BufWriter::new(file)));
    let running = Arc::new(AtomicBool::new(true));
    {
        let running = Arc::clone(&running);
        ctrlc::set_handler(move || running.store(false, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }
    info!(port, baud, output = %output.display(), "starting D500 recording; Ctrl-C to stop");

    let mut count = 0u64;
    let result = read_scans(port, baud, |sequence, scan| {
        if !running.load(Ordering::SeqCst) {
            anyhow::bail!("recording stopped by Ctrl-C");
        }
        let mut writer = writer.lock().expect("scan writer mutex poisoned");
        serde_json::to_writer(&mut *writer, &scan)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        count += 1;
        if count % 10 == 0 {
            info!(sequence, scan_count = count, "recorded scan");
        }
        Ok(())
    });
    match result {
        Ok(()) => Ok(()),
        Err(_error) if !running.load(Ordering::SeqCst) => {
            info!(scan_count = count, "recording stopped cleanly");
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn replay(input: &PathBuf, speed: f64) -> Result<()> {
    if !speed.is_finite() || speed < 0.0 {
        anyhow::bail!("--speed must be a finite number >= 0");
    }
    let file = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let reader = BufReader::new(file);
    let mut previous_ns = None;
    let mut count = 0u64;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let scan: ScanRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing scan on line {}", line_index + 1))?;
        if scan.schema != 1 || scan.ranges_m.len() != 180 {
            anyhow::bail!(
                "unsupported scan schema or range count on line {}",
                line_index + 1
            );
        }
        if scan
            .ranges_m
            .iter()
            .any(|range| !range.is_finite() || *range < 0.0 || *range > 12.0)
        {
            anyhow::bail!("invalid distance on line {}", line_index + 1);
        }
        if let Some(previous) = previous_ns {
            if scan.monotonic_ns < previous {
                anyhow::bail!("non-monotonic timestamp on line {}", line_index + 1);
            }
            if speed > 0.0 {
                let wait_ns = ((scan.monotonic_ns - previous) as f64 / speed) as u64;
                std::thread::sleep(std::time::Duration::from_nanos(wait_ns));
            }
        }
        let valid = scan.ranges_m.iter().filter(|r| **r < 12.0).count();
        let nearest = scan
            .ranges_m
            .iter()
            .copied()
            .filter(|r| *r < 12.0)
            .reduce(f32::min);
        println!(
            "replay scan={count} timestamp={} valid_bins={valid}/180 nearest_m={}",
            scan.timestamp_unix_ms,
            nearest
                .map(|n| format!("{n:.3}"))
                .unwrap_or_else(|| "none".to_owned())
        );
        previous_ns = Some(scan.monotonic_ns);
        count += 1;
    }
    if count == 0 {
        warn!(input = %input.display(), "replay file contained no scans");
    } else {
        info!(scan_count = count, "replay complete");
    }
    Ok(())
}

fn map_scans(input: &PathBuf, output: &PathBuf, resolution: f32) -> Result<()> {
    let scans = load_scans(input)?;
    info!(
        scan_count = scans.len(),
        resolution_m = resolution,
        "building occupancy map with consecutive-scan ICP"
    );
    let (grid, poses) = build_map(&scans, resolution)?;

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut pgm = output.clone();
    pgm.set_extension("pgm");
    let mut yaml = output.clone();
    yaml.set_extension("yaml");
    let mut pose_log = output.clone();
    pose_log.set_extension("poses.jsonl");
    grid.save(&pgm, &yaml)?;
    write_pose_log(&pose_log, &poses)?;

    let rejected = poses.iter().filter(|p| !p.accepted_match).count();
    println!("Map written: {}", pgm.display());
    println!("Metadata written: {}", yaml.display());
    println!("Pose log written: {}", pose_log.display());
    println!("scans={} rejected_matches={}", poses.len(), rejected);
    for pose in poses.iter().filter(|p| !p.accepted_match) {
        warn!(
            scan = pose.scan_index,
            inliers = pose.inliers,
            rmse_m = pose.rmse_m,
            "scan match rejected; pose held at previous estimate"
        );
    }
    Ok(())
}

fn localize_log(
    map_path: &PathBuf,
    input: &PathBuf,
    output: &PathBuf,
    initial_pose: LocalizationPose,
    minimum_confidence: f32,
) -> Result<()> {
    let map = OccupancyMap::load_yaml(map_path)?;
    let scans = load_scans(input)?;
    let records = localize_scans(&map, &scans, initial_pose, minimum_confidence)?;
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("creating {}", output.display()))?,
    );
    for record in &records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
        if record.stop {
            warn!(
                scan = record.scan_index,
                confidence = record.confidence,
                "STOP: localization confidence below threshold"
            );
            println!(
                "STOP scan={} confidence={:.3}",
                record.scan_index, record.confidence
            );
        } else {
            println!(
                "scan={} pose=({:.2}, {:.2}, {:.1}°) confidence={:.3}",
                record.scan_index,
                record.pose.x_m,
                record.pose.y_m,
                record.pose.yaw_rad.to_degrees(),
                record.confidence
            );
        }
    }
    writer.flush()?;
    info!(output = %output.display(), records = records.len(), "localization log written");
    Ok(())
}

fn load_scans(input: &PathBuf) -> Result<Vec<ScanRecord>> {
    if input
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
    {
        return load_python_collector_csv(input);
    }
    let file = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let reader = BufReader::new(file);
    let mut scans = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {}", line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let scan: ScanRecord = serde_json::from_str(&line)
            .with_context(|| format!("parsing scan on line {}", line_index + 1))?;
        if scan.schema != 1 || scan.ranges_m.len() != 180 {
            anyhow::bail!(
                "unsupported scan schema or range count on line {}",
                line_index + 1
            );
        }
        if scan
            .ranges_m
            .iter()
            .any(|range| !range.is_finite() || *range < 0.0 || *range > 12.0)
        {
            anyhow::bail!("invalid distance on line {}", line_index + 1);
        }
        scans.push(scan);
    }
    if scans.is_empty() {
        anyhow::bail!("scan log {} contains no scans", input.display());
    }
    Ok(scans)
}

/// Import the existing collector format: timestamp,servo,duty,lidar.
/// The old timestamp is local ISO text without a UTC offset, so keep its order
/// and place the original text in `source_timestamp` instead of claiming UTC.
fn load_python_collector_csv(input: &PathBuf) -> Result<Vec<ScanRecord>> {
    let mut reader = csv::Reader::from_path(input)
        .with_context(|| format!("opening legacy scan CSV {}", input.display()))?;
    let headers = reader.headers()?.clone();
    let expected = ["timestamp", "servo", "duty", "lidar"];
    if headers.iter().collect::<Vec<_>>() != expected {
        anyhow::bail!(
            "unsupported collector CSV header in {}; expected timestamp,servo,duty,lidar",
            input.display()
        );
    }
    let mut scans = Vec::new();
    for (index, row) in reader.records().enumerate() {
        let row = row.with_context(|| format!("reading collector CSV row {}", index + 2))?;
        let ranges: Vec<f32> = serde_json::from_str(&row[3])
            .with_context(|| format!("parsing lidar array in CSV row {}", index + 2))?;
        validate_ranges(&ranges, index + 2)?;
        scans.push(ScanRecord {
            schema: 1,
            sensor: "LDROBOT_D500_STL_19P".to_owned(),
            timestamp_unix_ms: 0,
            timestamp_unix_ns: 0,
            monotonic_ns: index as u128,
            source_timestamp: Some(row[0].to_owned()),
            duration_ms: 0.0,
            ranges_m: ranges,
        });
    }
    if scans.is_empty() {
        anyhow::bail!("scan CSV {} contains no scans", input.display());
    }
    Ok(scans)
}

fn validate_ranges(ranges: &[f32], line: usize) -> Result<()> {
    if ranges.len() != 180 {
        anyhow::bail!(
            "scan on line {line} has {} bins; expected 180",
            ranges.len()
        );
    }
    if ranges
        .iter()
        .any(|range| !range.is_finite() || *range < 0.0 || *range > 12.0)
    {
        anyhow::bail!("invalid distance on line {line}");
    }
    Ok(())
}

#[derive(Serialize)]
struct SensorPair {
    lidar_scan_index: usize,
    lidar_timestamp_unix_ns: u128,
    camera_sequence: u64,
    camera_timestamp_unix_ns: u64,
    camera_minus_lidar_ms: f64,
    pose: lidarcontrol::mapping::Pose2,
    localization_confidence: f32,
    camera_file: String,
}

fn sync_sensor_logs(
    poses_path: &PathBuf,
    frames_path: &PathBuf,
    output: &PathBuf,
    tolerance_ms: u64,
) -> Result<()> {
    let poses = read_jsonl::<PoseRecord>(poses_path)?;
    let frames = read_jsonl::<CameraFrameRecord>(frames_path)?;
    if poses.is_empty() || frames.is_empty() {
        anyhow::bail!("LiDAR pose log and camera frame log must both contain records");
    }
    if poses.iter().any(|pose| pose.timestamp_unix_ns == 0) {
        anyhow::bail!("LiDAR pose log has no Unix-nanosecond timestamps; legacy CSV timestamps do not include a timezone");
    }

    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("creating {}", output.display()))?,
    );
    let tolerance_ns = tolerance_ms as i128 * 1_000_000;
    let mut matched = 0usize;
    for pose in &poses {
        let nearest = frames.iter().min_by_key(|frame| {
            (frame.timestamp_unix_ns as i128 - pose.timestamp_unix_ns as i128).abs()
        });
        let Some(frame) = nearest else { continue };
        let delta_ns = frame.timestamp_unix_ns as i128 - pose.timestamp_unix_ns as i128;
        if delta_ns.abs() > tolerance_ns {
            continue;
        }
        let pair = SensorPair {
            lidar_scan_index: pose.scan_index,
            lidar_timestamp_unix_ns: pose.timestamp_unix_ns,
            camera_sequence: frame.sequence,
            camera_timestamp_unix_ns: frame.timestamp_unix_ns,
            camera_minus_lidar_ms: delta_ns as f64 / 1_000_000.0,
            pose: pose.pose,
            localization_confidence: pose.confidence,
            camera_file: frame.file.clone(),
        };
        serde_json::to_writer(&mut writer, &pair)?;
        writer.write_all(b"\n")?;
        matched += 1;
    }
    writer.flush()?;
    println!(
        "timestamp pairs: {matched}/{} scans within {} ms",
        poses.len(),
        tolerance_ms
    );
    println!("pair log: {}", output.display());
    Ok(())
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<Vec<T>> {
    let reader =
        BufReader::new(File::open(path).with_context(|| format!("opening {}", path.display()))?);
    let mut values = Vec::new();
    for (line_index, line) in reader.lines().enumerate() {
        let line =
            line.with_context(|| format!("reading {} line {}", path.display(), line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(
            serde_json::from_str(&line)
                .with_context(|| format!("parsing {} line {}", path.display(), line_index + 1))?,
        );
    }
    Ok(values)
}
