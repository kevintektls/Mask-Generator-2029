use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lidarcontrol::mapping::{build_map, write_pose_log};
use lidarcontrol::{read_scans, ScanRecord};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{info, warn};

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
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    lidar_port: String,
    lidar_baud: u32,
    scan_log: PathBuf,
    map_resolution_m: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lidar_port: "/dev/ttyTHS1".into(),
            lidar_baud: 230_400,
            scan_log: "lidarcontrol/data/scan.jsonl".into(),
            map_resolution_m: 0.05,
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
