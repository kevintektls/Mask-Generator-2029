//! LiDAR scan-to-map localization on a saved PGM/YAML occupancy map.

use crate::{ScanRecord, MAX_RANGE_M, SCAN_BINS};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::f32::consts::PI;
use std::fs;
use std::path::Path;

const SCORE_CLIP_M: f32 = 0.50;
const HIT_DISTANCE_M: f32 = 0.20;

#[derive(Debug, Clone)]
pub struct OccupancyMap {
    pub width: usize,
    pub height: usize,
    pub resolution_m: f32,
    pub origin_x_m: f32,
    pub origin_y_m: f32,
    pub(crate) known_free: Vec<bool>,
    pub(crate) distances_m: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, serde::Deserialize)]
pub struct LocalizationPose {
    pub x_m: f32,
    pub y_m: f32,
    pub yaw_rad: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalizationRecord {
    pub scan_index: usize,
    pub timestamp_unix_ns: u128,
    pub pose: LocalizationPose,
    pub confidence: f32,
    pub mean_error_m: f32,
    pub stop: bool,
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    pose: LocalizationPose,
    score: f32,
    confidence: f32,
}

impl OccupancyMap {
    pub fn load_yaml(path: &Path) -> Result<Self> {
        let metadata = fs::read_to_string(path)
            .with_context(|| format!("reading map metadata {}", path.display()))?;
        let mut image_file = None;
        let mut resolution = None;
        let mut origin = None;
        for line in metadata.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value.trim();
            match key.trim() {
                "image" => image_file = Some(value.to_owned()),
                "resolution" => {
                    resolution = Some(value.parse::<f32>().context("invalid map resolution")?)
                }
                "origin" => {
                    let values: Vec<f32> = value
                        .trim_matches(['[', ']'])
                        .split(',')
                        .map(|part| part.trim().parse::<f32>())
                        .collect::<std::result::Result<_, _>>()
                        .context("invalid map origin")?;
                    if values.len() != 3 {
                        bail!("map origin must contain x, y, yaw");
                    }
                    origin = Some([values[0], values[1], values[2]]);
                }
                _ => {}
            }
        }
        let image_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(image_file.context("map YAML has no image field")?);
        let (width, height, pixels) = read_pgm(&image_path)?;
        let resolution_m = resolution.context("map YAML has no resolution field")?;
        let origin = origin.context("map YAML has no origin field")?;
        if !resolution_m.is_finite() || resolution_m <= 0.0 {
            bail!("map resolution must be finite and positive");
        }
        let known_free = pixels.iter().map(|pixel| *pixel >= 200).collect();
        let distances_m = distance_field(width, height, resolution_m, &pixels);
        Ok(Self {
            width,
            height,
            resolution_m,
            origin_x_m: origin[0],
            origin_y_m: origin[1],
            known_free,
            distances_m,
        })
    }

    fn distance_at_world(&self, x_m: f32, y_m: f32) -> f32 {
        let col = ((x_m - self.origin_x_m) / self.resolution_m).floor() as isize;
        let row_from_bottom = ((y_m - self.origin_y_m) / self.resolution_m).floor() as isize;
        if col < 0
            || row_from_bottom < 0
            || col >= self.width as isize
            || row_from_bottom >= self.height as isize
        {
            return SCORE_CLIP_M;
        }
        let row = self.height - 1 - row_from_bottom as usize;
        self.distances_m[row * self.width + col as usize].min(SCORE_CLIP_M)
    }

    pub fn world_to_cell(&self, x_m: f32, y_m: f32) -> Option<(usize, usize)> {
        let col = ((x_m - self.origin_x_m) / self.resolution_m).floor() as isize;
        let row_from_bottom = ((y_m - self.origin_y_m) / self.resolution_m).floor() as isize;
        if col < 0
            || row_from_bottom < 0
            || col >= self.width as isize
            || row_from_bottom >= self.height as isize
        {
            return None;
        }
        Some((col as usize, self.height - 1 - row_from_bottom as usize))
    }

    pub fn cell_to_world(&self, col: usize, row: usize) -> (f32, f32) {
        (
            self.origin_x_m + (col as f32 + 0.5) * self.resolution_m,
            self.origin_y_m + (self.height as f32 - row as f32 - 0.5) * self.resolution_m,
        )
    }

    pub fn traversable(&self, col: usize, row: usize, safety_margin_m: f32) -> bool {
        if col >= self.width || row >= self.height {
            return false;
        }
        let index = row * self.width + col;
        self.known_free[index] && self.distances_m[index] >= safety_margin_m
    }

    fn score(&self, scan: &ScanRecord, pose: LocalizationPose) -> Option<Candidate> {
        let mut total = 0.0;
        let mut inliers = 0usize;
        let mut count = 0usize;
        let (s, c) = pose.yaw_rad.sin_cos();
        for (bin, &range) in scan.ranges_m.iter().enumerate() {
            if !range.is_finite() || range <= 0.05 || range >= MAX_RANGE_M {
                continue;
            }
            let angle_left = -((bin as f32 - SCAN_BINS as f32 / 2.0) * PI / 180.0);
            let local_x = range * angle_left.cos();
            let local_y = range * angle_left.sin();
            let world_x = pose.x_m + c * local_x - s * local_y;
            let world_y = pose.y_m + s * local_x + c * local_y;
            let distance = self.distance_at_world(world_x, world_y);
            total += distance;
            inliers += usize::from(distance < HIT_DISTANCE_M);
            count += 1;
        }
        if count < 12 {
            return None;
        }
        Some(Candidate {
            pose,
            score: total / count as f32,
            confidence: inliers as f32 / count as f32,
        })
    }

    /// Search locally around the previous pose, coarse first then refined.
    pub fn localize(
        &self,
        scan: &ScanRecord,
        seed: LocalizationPose,
    ) -> Option<(LocalizationPose, f32, f32)> {
        let mut best = self.score(scan, seed)?;
        for (xy_radius, xy_step, yaw_radius, yaw_step) in [
            (0.30, 0.10, 15.0_f32.to_radians(), 5.0_f32.to_radians()),
            (0.10, 0.02, 5.0_f32.to_radians(), 1.0_f32.to_radians()),
        ] {
            let center = best.pose;
            let mut dx = -xy_radius;
            while dx <= xy_radius + 1e-5 {
                let mut dy = -xy_radius;
                while dy <= xy_radius + 1e-5 {
                    let mut dyaw = -yaw_radius;
                    while dyaw <= yaw_radius + 1e-5 {
                        let candidate_pose = LocalizationPose {
                            x_m: center.x_m + dx,
                            y_m: center.y_m + dy,
                            yaw_rad: normalize_angle(center.yaw_rad + dyaw),
                        };
                        if let Some(candidate) = self.score(scan, candidate_pose) {
                            if candidate.score < best.score {
                                best = candidate;
                            }
                        }
                        dyaw += yaw_step;
                    }
                    dy += xy_step;
                }
                dx += xy_step;
            }
        }
        Some((best.pose, best.confidence, best.score))
    }
}

pub fn localize_scans(
    map: &OccupancyMap,
    scans: &[ScanRecord],
    initial_pose: LocalizationPose,
    minimum_confidence: f32,
) -> Result<Vec<LocalizationRecord>> {
    if scans.is_empty() {
        bail!("scan input contains no scans");
    }
    if !minimum_confidence.is_finite() || !(0.0..=1.0).contains(&minimum_confidence) {
        bail!("minimum confidence must be between 0 and 1");
    }
    let mut seed = initial_pose;
    let mut records = Vec::with_capacity(scans.len());
    let mut stopped = false;
    for (scan_index, scan) in scans.iter().enumerate() {
        if scan.ranges_m.len() != SCAN_BINS {
            bail!(
                "scan {scan_index} has {} bins; expected {SCAN_BINS}",
                scan.ranges_m.len()
            );
        }
        let result = map.localize(scan, seed);
        let (pose, confidence, mean_error_m) = match result {
            Some((pose, confidence, score)) if confidence >= minimum_confidence => {
                seed = pose;
                (pose, confidence, score)
            }
            Some((_, confidence, score)) => (seed, confidence, score),
            None => (seed, 0.0, SCORE_CLIP_M),
        };
        let stop = confidence < minimum_confidence;
        stopped |= stop;
        records.push(LocalizationRecord {
            scan_index,
            timestamp_unix_ns: scan.timestamp_unix_ns,
            pose,
            confidence,
            mean_error_m,
            stop,
        });
        if stopped {
            break;
        }
    }
    Ok(records)
}

fn read_pgm(path: &Path) -> Result<(usize, usize, Vec<u8>)> {
    let bytes = fs::read(path).with_context(|| format!("reading map image {}", path.display()))?;
    let mut cursor = 0usize;
    let magic = next_pgm_token(&bytes, &mut cursor)?;
    if magic != b"P5" {
        bail!("map image must be binary PGM (P5)");
    }
    let width = parse_pgm_usize(next_pgm_token(&bytes, &mut cursor)?, "width")?;
    let height = parse_pgm_usize(next_pgm_token(&bytes, &mut cursor)?, "height")?;
    let max_value = parse_pgm_usize(next_pgm_token(&bytes, &mut cursor)?, "max value")?;
    if max_value != 255 {
        bail!("PGM max value must be 255");
    }
    if bytes.get(cursor) == Some(&b'\r') && bytes.get(cursor + 1) == Some(&b'\n') {
        cursor += 2;
    } else if bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    } else {
        bail!("PGM header has no pixel separator");
    }
    let pixel_count = width
        .checked_mul(height)
        .context("PGM dimensions overflow")?;
    if width == 0 || height == 0 || bytes.len().saturating_sub(cursor) != pixel_count {
        bail!("PGM pixel payload size does not match dimensions");
    }
    Ok((width, height, bytes[cursor..].to_vec()))
}

fn next_pgm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a [u8]> {
    loop {
        while *cursor < bytes.len() && bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        }
        if *cursor < bytes.len() && bytes[*cursor] == b'#' {
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
            continue;
        }
        break;
    }
    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    if start == *cursor {
        bail!("truncated PGM header");
    }
    Ok(&bytes[start..*cursor])
}

fn parse_pgm_usize(token: &[u8], field: &str) -> Result<usize> {
    std::str::from_utf8(token)?
        .parse()
        .with_context(|| format!("invalid PGM {field}"))
}

fn distance_field(width: usize, height: usize, resolution_m: f32, pixels: &[u8]) -> Vec<f32> {
    const INF: u32 = u32::MAX / 4;
    let count = width * height;
    let mut distance = vec![INF; count];
    let mut queue = BinaryHeap::new();
    for (index, &pixel) in pixels.iter().enumerate() {
        if pixel < 100 {
            distance[index] = 0;
            queue.push(Reverse((0u32, index)));
        }
    }
    let neighbors = [
        (-1isize, -1isize, 1414u32),
        (0, -1, 1000),
        (1, -1, 1414),
        (-1, 0, 1000),
        (1, 0, 1000),
        (-1, 1, 1414),
        (0, 1, 1000),
        (1, 1, 1414),
    ];
    while let Some(Reverse((cost, index))) = queue.pop() {
        if cost != distance[index] {
            continue;
        }
        let x = index % width;
        let y = index / width;
        for &(dx, dy, weight) in &neighbors {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                continue;
            }
            let next = ny as usize * width + nx as usize;
            let next_cost = cost.saturating_add(weight);
            if next_cost < distance[next] {
                distance[next] = next_cost;
                queue.push(Reverse((next_cost, next)));
            }
        }
    }
    distance
        .into_iter()
        .map(|value| {
            if value == INF {
                SCORE_CLIP_M
            } else {
                (value as f32 / 1000.0 * resolution_m).min(SCORE_CLIP_M)
            }
        })
        .collect()
}

fn normalize_angle(angle: f32) -> f32 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_pgm_and_map_metadata() {
        let dir =
            std::env::temp_dir().join(format!("lidarcontrol-localization-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("map.pgm"),
            b"P5\n3 2\n255\n\xff\xff\x00\xff\xcd\xff",
        )
        .unwrap();
        fs::write(
            dir.join("map.yaml"),
            "image: map.pgm\nresolution: 0.05\norigin: [-1.0, -2.0, 0.0]\n",
        )
        .unwrap();
        let map = OccupancyMap::load_yaml(&dir.join("map.yaml")).unwrap();
        assert_eq!((map.width, map.height), (3, 2));
        assert_eq!(map.resolution_m, 0.05);
        assert_eq!(map.origin_x_m, -1.0);
        assert_eq!(map.origin_y_m, -2.0);
        assert_eq!(map.distance_at_world(-0.875, -1.925), 0.0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn stop_is_set_when_no_scan_points_match_map_well() {
        let map = OccupancyMap {
            width: 20,
            height: 20,
            resolution_m: 0.05,
            origin_x_m: 0.0,
            origin_y_m: 0.0,
            known_free: vec![false; 400],
            distances_m: vec![SCORE_CLIP_M; 400],
        };
        let scan = ScanRecord {
            schema: 1,
            sensor: "test".into(),
            timestamp_unix_ms: 1,
            timestamp_unix_ns: 1_000_000,
            source_timestamp: None,
            monotonic_ns: 1,
            duration_ms: 1.0,
            ranges_m: vec![2.0; SCAN_BINS],
        };
        let records = localize_scans(&map, &[scan], LocalizationPose::default(), 0.3).unwrap();
        assert_eq!(records.len(), 1);
        assert!(records[0].stop);
        assert_eq!(records[0].confidence, 0.0);
    }
}
