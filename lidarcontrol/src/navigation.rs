//! Global grid planning and path tracking math.

use crate::localization::{LocalizationPose, OccupancyMap};
use anyhow::{bail, Result};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize, PartialEq)]
pub struct Waypoint {
    pub x_m: f32,
    pub y_m: f32,
}

#[derive(Debug, Clone, Copy, Serialize, serde::Deserialize, PartialEq)]
pub struct DriveCommand {
    /// Normalized servo value in the VESC's [0, 1] domain.
    pub servo: f32,
    /// Requested forward speed in m/s (for simulation or a calibrated adapter).
    pub speed_mps: f32,
}

pub fn plan_path(
    map: &OccupancyMap,
    start: Waypoint,
    goal: Waypoint,
    safety_margin_m: f32,
) -> Result<Vec<Waypoint>> {
    if !safety_margin_m.is_finite() || safety_margin_m < 0.0 {
        bail!("safety margin must be finite and non-negative");
    }
    let start_cell = map
        .world_to_cell(start.x_m, start.y_m)
        .ok_or_else(|| anyhow::anyhow!("start is outside the saved map"))?;
    let goal_cell = map
        .world_to_cell(goal.x_m, goal.y_m)
        .ok_or_else(|| anyhow::anyhow!("goal is outside the saved map"))?;
    if !map.traversable(start_cell.0, start_cell.1, safety_margin_m) {
        bail!("start cell is occupied, unknown, or inside the safety margin");
    }
    if !map.traversable(goal_cell.0, goal_cell.1, safety_margin_m) {
        bail!("goal cell is occupied, unknown, or inside the safety margin");
    }

    let cell_count = map.width * map.height;
    let start_index = start_cell.1 * map.width + start_cell.0;
    let goal_index = goal_cell.1 * map.width + goal_cell.0;
    let mut costs = vec![u64::MAX; cell_count];
    let mut previous = vec![usize::MAX; cell_count];
    let mut open = BinaryHeap::new();
    costs[start_index] = 0;
    open.push(Reverse((heuristic(start_cell, goal_cell), start_index)));

    let directions: [(isize, isize, u32); 8] = [
        (-1, -1, 1414),
        (0, -1, 1000),
        (1, -1, 1414),
        (-1, 0, 1000),
        (1, 0, 1000),
        (-1, 1, 1414),
        (0, 1, 1000),
        (1, 1, 1414),
    ];

    while let Some(Reverse((_priority, current))) = open.pop() {
        if current == goal_index {
            break;
        }
        let col = current % map.width;
        let row = current / map.width;
        for &(dx, dy, step_cost) in &directions {
            let next_col = col as isize + dx;
            let next_row = row as isize + dy;
            if next_col < 0
                || next_row < 0
                || next_col >= map.width as isize
                || next_row >= map.height as isize
            {
                continue;
            }
            let next_cell = (next_col as usize, next_row as usize);
            if !map.traversable(next_cell.0, next_cell.1, safety_margin_m) {
                continue;
            }
            // Do not cut across blocked/unknown corners diagonally.
            if dx != 0 && dy != 0 {
                let side_a = ((col as isize + dx) as usize, row);
                let side_b = (col, (row as isize + dy) as usize);
                if !map.traversable(side_a.0, side_a.1, safety_margin_m)
                    || !map.traversable(side_b.0, side_b.1, safety_margin_m)
                {
                    continue;
                }
            }
            let next = next_cell.1 * map.width + next_cell.0;
            let tentative = costs[current].saturating_add(step_cost as u64);
            if tentative < costs[next] {
                costs[next] = tentative;
                previous[next] = current;
                let estimate = tentative + heuristic(next_cell, goal_cell);
                open.push(Reverse((estimate, next)));
            }
        }
    }

    if costs[goal_index] == u64::MAX {
        bail!("no traversable route from start to goal");
    }
    let mut indices = vec![goal_index];
    let mut cursor = goal_index;
    while cursor != start_index {
        cursor = previous[cursor];
        if cursor == usize::MAX {
            bail!("path reconstruction failed");
        }
        indices.push(cursor);
    }
    indices.reverse();
    Ok(indices
        .into_iter()
        .map(|index| {
            let (x_m, y_m) = map.cell_to_world(index % map.width, index / map.width);
            Waypoint { x_m, y_m }
        })
        .collect())
}

fn heuristic(a: (usize, usize), b: (usize, usize)) -> u64 {
    let dx = a.0.abs_diff(b.0) as u64;
    let dy = a.1.abs_diff(b.1) as u64;
    let diagonal = dx.min(dy);
    diagonal * 1414 + (dx.max(dy) - diagonal) * 1000
}

/// Pure-pursuit steering calculation. Servo calibration remains a required
/// runtime parameter; callers must not actuate with placeholder values.
pub fn pure_pursuit(
    pose: LocalizationPose,
    path: &[Waypoint],
    lookahead_m: f32,
    wheelbase_m: f32,
    max_steering_rad: f32,
    servo_center: f32,
    servo_left: f32,
    servo_right: f32,
    speed_mps: f32,
) -> Result<DriveCommand> {
    if path.is_empty() {
        bail!("cannot follow an empty path");
    }
    if !lookahead_m.is_finite()
        || lookahead_m <= 0.0
        || !wheelbase_m.is_finite()
        || wheelbase_m <= 0.0
        || !max_steering_rad.is_finite()
        || max_steering_rad <= 0.0
        || !speed_mps.is_finite()
        || speed_mps < 0.0
        || [servo_center, servo_left, servo_right]
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        bail!("invalid pure-pursuit calibration or speed parameter");
    }
    let target = path
        .iter()
        .copied()
        .find(|point| (point.x_m - pose.x_m).hypot(point.y_m - pose.y_m) >= lookahead_m)
        .unwrap_or(*path.last().unwrap());
    let dx = target.x_m - pose.x_m;
    let dy = target.y_m - pose.y_m;
    let alpha = normalize_angle(dy.atan2(dx) - pose.yaw_rad);
    let distance_squared = (dx * dx + dy * dy).max(lookahead_m * lookahead_m);
    let curvature = 2.0 * alpha.sin() / distance_squared.sqrt();
    let steering_angle = (wheelbase_m * curvature)
        .atan()
        .clamp(-max_steering_rad, max_steering_rad);
    let servo = if steering_angle >= 0.0 {
        servo_center + (servo_right - servo_center) * steering_angle / max_steering_rad
    } else {
        servo_center + (servo_center - servo_left) * steering_angle / max_steering_rad
    };
    Ok(DriveCommand {
        servo: servo.clamp(0.0, 1.0),
        speed_mps,
    })
}

fn normalize_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(2.0 * std::f32::consts::PI) - std::f32::consts::PI
}

/// Serialize a compact map view for the local browser UI. Unknown and occupied
/// cells are both blocked; this conservative view is what the planner uses.
pub fn map_view(map: &OccupancyMap) -> serde_json::Value {
    let cells: Vec<u8> = (0..map.height)
        .flat_map(|row| (0..map.width).map(move |col| u8::from(map.traversable(col, row, 0.0))))
        .collect();
    serde_json::json!({
        "width": map.width,
        "height": map.height,
        "resolution_m": map.resolution_m,
        "origin_x_m": map.origin_x_m,
        "origin_y_m": map.origin_y_m,
        "cells": cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pursuit_requires_explicit_vehicle_calibration() {
        let result = pure_pursuit(
            LocalizationPose::default(),
            &[Waypoint { x_m: 1.0, y_m: 1.0 }],
            0.4,
            0.25,
            0.4,
            0.5,
            0.0,
            1.0,
            0.2,
        )
        .unwrap();
        assert!(result.servo > 0.5);
        assert_eq!(result.speed_mps, 0.2);
        assert!(pure_pursuit(
            LocalizationPose::default(),
            &[],
            0.4,
            0.25,
            0.4,
            0.5,
            0.0,
            1.0,
            0.2
        )
        .is_err());
    }

    #[test]
    fn refuses_goal_in_unknown_space() {
        let map = OccupancyMap {
            width: 4,
            height: 4,
            resolution_m: 0.1,
            origin_x_m: 0.0,
            origin_y_m: 0.0,
            known_free: vec![false; 16],
            distances_m: vec![1.0; 16],
        };
        assert!(plan_path(
            &map,
            Waypoint {
                x_m: 0.05,
                y_m: 0.05
            },
            Waypoint {
                x_m: 0.25,
                y_m: 0.25
            },
            0.1
        )
        .is_err());
    }

    #[test]
    fn astar_returns_route_inside_known_free_cells() {
        let map = OccupancyMap {
            width: 5,
            height: 5,
            resolution_m: 1.0,
            origin_x_m: 0.0,
            origin_y_m: 0.0,
            known_free: vec![true; 25],
            distances_m: vec![2.0; 25],
        };
        let path = plan_path(
            &map,
            Waypoint { x_m: 0.5, y_m: 0.5 },
            Waypoint { x_m: 4.5, y_m: 4.5 },
            0.5,
        )
        .unwrap();
        assert_eq!(path.first(), Some(&Waypoint { x_m: 0.5, y_m: 0.5 }));
        assert_eq!(path.last(), Some(&Waypoint { x_m: 4.5, y_m: 4.5 }));
        assert_eq!(path.len(), 5);
    }
}
