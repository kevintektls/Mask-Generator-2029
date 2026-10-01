//! Offline 2D occupancy mapping from D500 scan logs.
//!
//! Consecutive scans are aligned with point-to-point ICP. This is a practical
//! first mapping step when wheel odometry is unavailable; it is not a full
//! loop-closing SLAM system.

use crate::{ScanRecord, MAX_RANGE_M, SCAN_BINS};
use anyhow::{bail, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const MAX_ASSOCIATION_M: f32 = 0.75;
const MAX_MATCH_RMSE_M: f32 = 0.30;
const MAX_STEP_TRANSLATION_M: f32 = 1.0;
const MAX_STEP_ROTATION_RAD: f32 = 0.8;

#[derive(Debug, Clone, Copy, Default, Serialize, serde::Deserialize)]
pub struct Pose2 {
    pub x_m: f32,
    pub y_m: f32,
    /// Counter-clockwise radians in the map frame.
    pub yaw_rad: f32,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PoseRecord {
    pub scan_index: usize,
    pub timestamp_unix_ms: u128,
    pub timestamp_unix_ns: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_timestamp: Option<String>,
    pub pose: Pose2,
    pub confidence: f32,
    pub accepted_match: bool,
    pub inliers: usize,
    pub rmse_m: f32,
}

#[derive(Debug, Clone, Copy)]
struct Point2 {
    x: f32,
    y: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct Transform2 {
    x: f32,
    y: f32,
    yaw: f32,
}

impl Transform2 {
    fn apply(self, point: Point2) -> Point2 {
        let (s, c) = self.yaw.sin_cos();
        Point2 {
            x: c * point.x - s * point.y + self.x,
            y: s * point.x + c * point.y + self.y,
        }
    }

    /// Compose `self` after `rhs`: result(p) = self(rhs(p)).
    fn compose(self, rhs: Self) -> Self {
        let translated = self.apply(Point2 { x: rhs.x, y: rhs.y });
        Self {
            x: translated.x,
            y: translated.y,
            yaw: normalize_angle(self.yaw + rhs.yaw),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MatchResult {
    current_to_previous: Transform2,
    inliers: usize,
    rmse_m: f32,
}

#[derive(Default)]
pub struct OccupancyGrid {
    resolution_m: f32,
    cells: HashMap<(i32, i32), i8>,
}

impl OccupancyGrid {
    pub fn new(resolution_m: f32) -> Result<Self> {
        if !resolution_m.is_finite() || !(0.01..=0.25).contains(&resolution_m) {
            bail!("map resolution must be between 0.01 and 0.25 metres");
        }
        Ok(Self {
            resolution_m,
            cells: HashMap::new(),
        })
    }

    fn world_to_cell(&self, x: f32, y: f32) -> (i32, i32) {
        (
            (x / self.resolution_m).floor() as i32,
            (y / self.resolution_m).floor() as i32,
        )
    }

    fn update(&mut self, cell: (i32, i32), delta: i8) {
        let odds = self.cells.entry(cell).or_insert(0);
        *odds = (*odds + delta).clamp(-20, 20);
    }

    pub fn add_scan(&mut self, scan: &ScanRecord, pose: Pose2) {
        let sensor = Point2 {
            x: pose.x_m,
            y: pose.y_m,
        };
        for (bin, &range) in scan.ranges_m.iter().enumerate() {
            if !range.is_finite() || range <= 0.0 || range > MAX_RANGE_M {
                continue;
            }
            let angle_right = (bin as f32 - (SCAN_BINS as f32 / 2.0)) * PI / 180.0;
            let angle_map = pose.yaw_rad - angle_right;
            let endpoint = Point2 {
                x: sensor.x + range * angle_map.cos(),
                y: sensor.y + range * angle_map.sin(),
            };
            let start_cell = self.world_to_cell(sensor.x, sensor.y);
            let end_cell = self.world_to_cell(endpoint.x, endpoint.y);
            let is_hit = range < MAX_RANGE_M - f32::EPSILON;
            self.trace_ray(start_cell, end_cell, is_hit);
        }
    }

    fn trace_ray(&mut self, start: (i32, i32), end: (i32, i32), hit: bool) {
        let (mut x, mut y) = start;
        let dx = (end.0 - x).abs();
        let sx = if x < end.0 { 1 } else { -1 };
        let dy = -(end.1 - y).abs();
        let sy = if y < end.1 { 1 } else { -1 };
        let mut error = dx + dy;

        loop {
            if (x, y) == end {
                if hit {
                    self.update((x, y), 3);
                }
                break;
            }
            self.update((x, y), -1);
            let twice_error = 2 * error;
            if twice_error >= dy {
                error += dy;
                x += sx;
            }
            if twice_error <= dx {
                error += dx;
                y += sy;
            }
        }
    }

    pub fn save(&self, pgm_path: &Path, yaml_path: &Path) -> Result<()> {
        if self.cells.is_empty() {
            bail!("cannot save an empty occupancy grid");
        }
        let min_x = self.cells.keys().map(|(x, _)| *x).min().unwrap();
        let max_x = self.cells.keys().map(|(x, _)| *x).max().unwrap();
        let min_y = self.cells.keys().map(|(_, y)| *y).min().unwrap();
        let max_y = self.cells.keys().map(|(_, y)| *y).max().unwrap();
        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;
        if width == 0 || height == 0 || width.saturating_mul(height) > 100_000_000 {
            bail!("map bounds are invalid or exceed 100 million cells");
        }

        let file = File::create(pgm_path)?;
        let mut writer = BufWriter::new(file);
        write!(writer, "P5\n{width} {height}\n255\n")?;
        for row in 0..height {
            let world_y = max_y - row as i32;
            for col in 0..width {
                let world_x = min_x + col as i32;
                let pixel = match self.cells.get(&(world_x, world_y)).copied().unwrap_or(0) {
                    odds if odds >= 2 => 0u8,    // occupied: black
                    odds if odds <= -1 => 254u8, // free: white
                    _ => 205u8,                  // unknown: grey
                };
                writer.write_all(&[pixel])?;
            }
        }
        writer.flush()?;

        let origin_x = min_x as f32 * self.resolution_m;
        let origin_y = min_y as f32 * self.resolution_m;
        let mut yaml = BufWriter::new(File::create(yaml_path)?);
        writeln!(
            yaml,
            "image: {}",
            pgm_path.file_name().unwrap_or_default().to_string_lossy()
        )?;
        writeln!(yaml, "resolution: {}", self.resolution_m)?;
        writeln!(yaml, "origin: [{origin_x}, {origin_y}, 0.0]")?;
        writeln!(yaml, "negate: 0")?;
        writeln!(yaml, "occupied_thresh: 0.65")?;
        writeln!(yaml, "free_thresh: 0.196")?;
        writeln!(yaml, "mode: trinary")?;
        yaml.flush()?;
        Ok(())
    }
}

fn scan_points(scan: &ScanRecord) -> Vec<Point2> {
    scan.ranges_m
        .iter()
        .enumerate()
        .filter_map(|(bin, range)| {
            if !range.is_finite() || *range <= 0.05 || *range >= MAX_RANGE_M {
                return None;
            }
            let angle_right = (bin as f32 - (SCAN_BINS as f32 / 2.0)) * PI / 180.0;
            let angle_left = -angle_right;
            Some(Point2 {
                x: range * angle_left.cos(),
                y: range * angle_left.sin(),
            })
        })
        .collect()
}

pub fn build_map(
    scans: &[ScanRecord],
    resolution_m: f32,
) -> Result<(OccupancyGrid, Vec<PoseRecord>)> {
    if scans.is_empty() {
        bail!("scan log contains no scans");
    }
    let mut grid = OccupancyGrid::new(resolution_m)?;
    let mut previous_points = Vec::<Point2>::new();
    let mut pose = Pose2::default();
    let mut pose_records = Vec::with_capacity(scans.len());

    for (scan_index, scan) in scans.iter().enumerate() {
        if scan.ranges_m.len() != SCAN_BINS {
            bail!(
                "scan {scan_index} has {} bins; expected {SCAN_BINS}",
                scan.ranges_m.len()
            );
        }
        let current_points = scan_points(scan);
        let (accepted_match, inliers, rmse_m, confidence) = if previous_points.is_empty() {
            (true, current_points.len(), 0.0, 1.0)
        } else {
            match match_scans(&current_points, &previous_points) {
                Some(result)
                    if result.inliers >= 12
                        && result.rmse_m <= MAX_MATCH_RMSE_M
                        && result
                            .current_to_previous
                            .x
                            .hypot(result.current_to_previous.y)
                            <= MAX_STEP_TRANSLATION_M
                        && result.current_to_previous.yaw.abs() <= MAX_STEP_ROTATION_RAD =>
                {
                    pose = compose_pose(pose, result.current_to_previous);
                    let confidence = (result.inliers as f32 / current_points.len().max(1) as f32)
                        * (1.0 - result.rmse_m / MAX_MATCH_RMSE_M).clamp(0.0, 1.0);
                    (
                        true,
                        result.inliers,
                        result.rmse_m,
                        confidence.clamp(0.0, 1.0),
                    )
                }
                Some(result) => (false, result.inliers, result.rmse_m, 0.0),
                None => (false, 0, f32::INFINITY, 0.0),
            }
        };

        grid.add_scan(scan, pose);
        pose_records.push(PoseRecord {
            scan_index,
            timestamp_unix_ms: scan.timestamp_unix_ms,
            timestamp_unix_ns: scan.timestamp_unix_ns,
            source_timestamp: scan.source_timestamp.clone(),
            pose,
            confidence,
            accepted_match,
            inliers,
            rmse_m,
        });
        if !current_points.is_empty() {
            previous_points = current_points;
        }
    }
    Ok((grid, pose_records))
}

fn match_scans(current: &[Point2], previous: &[Point2]) -> Option<MatchResult> {
    if current.len() < 12 || previous.len() < 12 {
        return None;
    }
    let mut transformed = current.to_vec();
    let mut estimate = Transform2::default();
    let mut inlier_count = 0;
    let mut rmse = f32::INFINITY;

    for _ in 0..20 {
        let mut pairs = Vec::with_capacity(current.len());
        for source in &transformed {
            let (target, distance_squared) = previous
                .iter()
                .map(|target| {
                    let dx = source.x - target.x;
                    let dy = source.y - target.y;
                    (*target, dx * dx + dy * dy)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))?;
            if distance_squared <= MAX_ASSOCIATION_M * MAX_ASSOCIATION_M {
                pairs.push((*source, target, distance_squared));
            }
        }
        if pairs.len() < 12 {
            return None;
        }
        pairs.sort_by(|a, b| a.2.total_cmp(&b.2));
        pairs.truncate((pairs.len() * 4 / 5).max(12));
        let from: Vec<_> = pairs.iter().map(|(source, _, _)| *source).collect();
        let to: Vec<_> = pairs.iter().map(|(_, target, _)| *target).collect();
        let correction = fit_rigid_transform(&from, &to)?;
        for point in &mut transformed {
            *point = correction.apply(*point);
        }
        estimate = correction.compose(estimate);
        inlier_count = pairs.len();
        rmse = (pairs.iter().map(|(_, _, d2)| *d2).sum::<f32>() / pairs.len() as f32).sqrt();
        if correction.x.hypot(correction.y) < 0.0005 && correction.yaw.abs() < 0.0005 {
            break;
        }
    }
    Some(MatchResult {
        current_to_previous: estimate,
        inliers: inlier_count,
        rmse_m: rmse,
    })
}

fn fit_rigid_transform(from: &[Point2], to: &[Point2]) -> Option<Transform2> {
    if from.len() != to.len() || from.len() < 2 {
        return None;
    }
    let n = from.len() as f32;
    let from_mean = Point2 {
        x: from.iter().map(|p| p.x).sum::<f32>() / n,
        y: from.iter().map(|p| p.y).sum::<f32>() / n,
    };
    let to_mean = Point2 {
        x: to.iter().map(|p| p.x).sum::<f32>() / n,
        y: to.iter().map(|p| p.y).sum::<f32>() / n,
    };
    let mut dot = 0.0;
    let mut cross = 0.0;
    for (source, target) in from.iter().zip(to) {
        let sx = source.x - from_mean.x;
        let sy = source.y - from_mean.y;
        let tx = target.x - to_mean.x;
        let ty = target.y - to_mean.y;
        dot += sx * tx + sy * ty;
        cross += sx * ty - sy * tx;
    }
    let yaw = cross.atan2(dot);
    let (s, c) = yaw.sin_cos();
    Some(Transform2 {
        x: to_mean.x - (c * from_mean.x - s * from_mean.y),
        y: to_mean.y - (s * from_mean.x + c * from_mean.y),
        yaw,
    })
}

fn compose_pose(pose: Pose2, delta: Transform2) -> Pose2 {
    let result = Transform2 {
        x: pose.x_m,
        y: pose.y_m,
        yaw: pose.yaw_rad,
    }
    .compose(delta);
    Pose2 {
        x_m: result.x,
        y_m: result.y,
        yaw_rad: result.yaw,
    }
}

fn normalize_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

pub fn write_pose_log(path: &Path, poses: &[PoseRecord]) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    for pose in poses {
        serde_json::to_writer(&mut writer, pose)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(points: &[(usize, f32)], timestamp: u128) -> ScanRecord {
        let mut ranges = vec![MAX_RANGE_M; SCAN_BINS];
        for &(bin, range) in points {
            ranges[bin] = range;
        }
        ScanRecord {
            schema: 1,
            sensor: "test".into(),
            timestamp_unix_ms: timestamp,
            timestamp_unix_ns: timestamp * 1_000_000,
            source_timestamp: None,
            monotonic_ns: timestamp * 1_000_000,
            duration_ms: 100.0,
            ranges_m: ranges,
        }
    }

    #[test]
    fn rigid_fit_recovers_known_transform() {
        let from = vec![
            Point2 { x: 1.0, y: 0.0 },
            Point2 { x: 0.0, y: 2.0 },
            Point2 { x: -1.0, y: 1.0 },
        ];
        let expected = Transform2 {
            x: 0.2,
            y: -0.1,
            yaw: 0.15,
        };
        let to: Vec<_> = from.iter().map(|p| expected.apply(*p)).collect();
        let actual = fit_rigid_transform(&from, &to).unwrap();
        assert!((actual.x - expected.x).abs() < 1e-5);
        assert!((actual.y - expected.y).abs() < 1e-5);
        assert!((actual.yaw - expected.yaw).abs() < 1e-5);
    }

    #[test]
    fn grid_saves_map_image_and_metadata() {
        let mut grid = OccupancyGrid::new(0.05).unwrap();
        grid.add_scan(&scan(&[(90, 1.0)], 1), Pose2::default());
        let directory =
            std::env::temp_dir().join(format!("lidarcontrol-map-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let pgm = directory.join("map.pgm");
        let yaml = directory.join("map.yaml");
        grid.save(&pgm, &yaml).unwrap();
        assert!(std::fs::metadata(&pgm).unwrap().len() > 10);
        let metadata = std::fs::read_to_string(&yaml).unwrap();
        assert!(metadata.contains("resolution: 0.05"));
        assert!(metadata.contains("origin:"));
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn builds_pose_for_each_scan_and_rejects_empty_input() {
        let first = scan(
            &[
                (20, 2.0),
                (40, 2.2),
                (60, 2.4),
                (80, 2.6),
                (100, 2.8),
                (120, 3.0),
                (140, 3.2),
                (160, 3.4),
                (10, 2.5),
                (30, 2.7),
                (50, 2.9),
                (70, 3.1),
            ],
            1,
        );
        let second = first.clone();
        let (_, poses) = build_map(&[first, second], 0.05).unwrap();
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].scan_index, 0);
        assert!(build_map(&[], 0.05).is_err());
    }
}
