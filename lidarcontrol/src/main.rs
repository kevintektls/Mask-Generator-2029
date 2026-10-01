use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lidarcontrol::{read_scans, ScanRecord};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

#[derive(Debug, Parser)]
#[command(name = "lidarcontrol", about = "D500 acquisition and replay (stage 1)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
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
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    lidar_port: String,
    lidar_baud: u32,
    scan_log: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lidar_port: "/dev/ttyTHS1".into(),
            lidar_baud: 230_400,
            scan_log: "lidarcontrol/data/scan.jsonl".into(),
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
    }
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
        Err(error) if !running.load(Ordering::SeqCst) => {
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
