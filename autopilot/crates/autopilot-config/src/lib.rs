//! Shared configuration constants and CLI for the autopilot stack.

use clap::Parser;
use std::path::PathBuf;

// ── Display / stream ─────────────────────────────────────────────────────────
pub const DISPLAY_W: u32 = 640;
pub const DISPLAY_H: u32 = 480;
pub const CAM_FPS: u32 = 60;
pub const STREAM_PORT: u16 = 8080;

// ── Camera bridge (Python depthai 2.x) ───────────────────────────────────────
pub const MONO_W: u32 = 640;
pub const MONO_H: u32 = 480;
pub const CAMERA_BRIDGE_ADDR: &str = "127.0.0.1:9000";
pub const CAMERA_BRIDGE_MAGIC: &[u8; 4] = b"OAK1";

// ── Vision ───────────────────────────────────────────────────────────────────
pub const CROP_TOP_RATIO: f64 = 0.20;
pub const ULTRA_BINARY_THRESH: u8 = 220;
pub const MASK_W: u32 = 160;
pub const MASK_H: u32 = 120;

// ── VESC ─────────────────────────────────────────────────────────────────────
pub const VESC_PORT: &str = "/dev/ttyACM0";
pub const VESC_BAUDRATE: u32 = 115_200;
pub const VESC_TIMEOUT_MS: u64 = 1000;
pub const VESC_CONNECT_RETRIES: u32 = 8;
pub const VESC_CONNECT_SETTLE_MS: u64 = 1000;

// ── Driving ──────────────────────────────────────────────────────────────────
pub const SERVO_CENTER: f32 = 0.5;
pub const SERVO_RANGE: f32 = 0.48;
pub const DUTY_MIN: f32 = 0.050;
pub const DUTY_MAX: f32 = 0.070;
pub const STEER_THRESHOLD: f32 = 0.08;

pub const EMERGENCY_BRAKE_A: f32 = 15.0;
pub const SHUTDOWN_BRAKE_A: f32 = 10.0;

pub const DEFAULT_MODEL_PATH: &str = "../model/pilot_model.pth";

/// Adaptive throttle: slow down when steering away from center.
pub fn adaptive_duty(servo_pos: f32) -> f32 {
    let steering_intensity = (servo_pos - SERVO_CENTER).abs();
    if steering_intensity < STEER_THRESHOLD {
        DUTY_MAX
    } else {
        let factor = ((steering_intensity - STEER_THRESHOLD) / (SERVO_RANGE - STEER_THRESHOLD))
            .clamp(0.0, 1.0);
        DUTY_MAX - factor * (DUTY_MAX - DUTY_MIN)
    }
}

#[derive(Debug, Parser)]
#[command(name = "autopilot", about = "Robot car hybrid autopilot (Rust)")]
pub struct Cli {
    /// Path to the behavioral-cloning `.pth` weights file.
    #[arg(long, default_value = DEFAULT_MODEL_PATH)]
    pub model: PathBuf,

    /// VESC serial port.
    #[arg(long, default_value = VESC_PORT)]
    pub vesc_port: String,

    /// MJPEG stream HTTP port.
    #[arg(long, default_value_t = STREAM_PORT)]
    pub stream_port: u16,

    /// Camera FPS target (used by `tools/camera_bridge.py`).
    #[arg(long, default_value_t = CAM_FPS)]
    pub cam_fps: u32,

    /// TCP address of `tools/camera_bridge.py`.
    #[arg(long, default_value = CAMERA_BRIDGE_ADDR)]
    pub camera_addr: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_duty_straight_is_max() {
        assert!((adaptive_duty(SERVO_CENTER) - DUTY_MAX).abs() < f32::EPSILON);
    }

    #[test]
    fn adaptive_duty_full_steer_is_min() {
        let duty = adaptive_duty(SERVO_CENTER + SERVO_RANGE);
        assert!((duty - DUTY_MIN).abs() < 0.001);
    }
}
