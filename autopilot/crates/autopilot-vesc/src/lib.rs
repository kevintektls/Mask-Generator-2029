//! VESC motor controller over serial (duty, brake, servo).

use anyhow::{Context, Result};
use autopilot_config::{
    SERVO_CENTER, VESC_BAUDRATE, VESC_CONNECT_RETRIES, VESC_CONNECT_SETTLE_MS, VESC_PORT,
    VESC_TIMEOUT_MS,
};
use serialport::SerialPort;
use std::io::Write;
use std::thread;
use std::time::Duration;
use thiserror::Error;

const COMM_SET_DUTY: u8 = 5;
const COMM_SET_CURRENT_BRAKE: u8 = 7;
const COMM_SET_SERVO_POS: u8 = 12;

const DUTY_SCALE: f32 = 100_000.0;
const CURRENT_SCALE: f32 = 1_000.0;
const SERVO_SCALE: f32 = 1_000.0;

#[derive(Debug, Error)]
pub enum VescError {
    #[error("serial I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("encode buffer too small")]
    BufferTooSmall,
}

/// Encode a VESC UART packet (short packet, payload < 256 bytes).
fn encode_packet(payload: &[u8], out: &mut [u8]) -> Result<usize, VescError> {
    if payload.is_empty() || out.len() < payload.len() + 6 {
        return Err(VescError::BufferTooSmall);
    }
    out[0] = 2;
    out[1] = payload.len() as u8;
    out[2..2 + payload.len()].copy_from_slice(payload);
    let crc = crc16(&out[2..2 + payload.len()]);
    let end = 2 + payload.len();
    out[end] = (crc >> 8) as u8;
    out[end + 1] = (crc & 0xFF) as u8;
    out[end + 2] = 3;
    Ok(end + 3)
}

fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        crc ^= (byte as u16) << 8;
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

fn encode_set_duty(duty: f32) -> Result<Vec<u8>> {
    let scaled = (duty * DUTY_SCALE) as i32;
    let mut payload = vec![COMM_SET_DUTY];
    payload.extend_from_slice(&scaled.to_be_bytes());
    let mut buf = vec![0u8; 32];
    let n = encode_packet(&payload, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

fn encode_set_brake(amps: f32) -> Result<Vec<u8>> {
    let scaled = (amps * CURRENT_SCALE) as i32;
    let mut payload = vec![COMM_SET_CURRENT_BRAKE];
    payload.extend_from_slice(&scaled.to_be_bytes());
    let mut buf = vec![0u8; 32];
    let n = encode_packet(&payload, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

fn encode_set_servo(pos: f32) -> Result<Vec<u8>> {
    let clamped = pos.clamp(0.0, 1.0);
    let scaled = (clamped * SERVO_SCALE) as i16;
    let mut payload = vec![COMM_SET_SERVO_POS];
    payload.extend_from_slice(&scaled.to_be_bytes());
    let mut buf = vec![0u8; 32];
    let n = encode_packet(&payload, &mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

pub struct VescClient {
    port: Box<dyn SerialPort>,
    shutdown_brake_a: f32,
}

impl VescClient {
    pub fn connect(port_path: &str) -> Result<Self> {
        let mut last_err = None;
        for attempt in 1..=VESC_CONNECT_RETRIES {
            match Self::open_port(port_path) {
                Ok(client) => {
                    tracing::info!("VESC connected on attempt {attempt}");
                    return Ok(client);
                }
                Err(e) => {
                    tracing::warn!("VESC connect attempt {attempt} failed: {e}");
                    last_err = Some(e);
                    thread::sleep(Duration::from_millis(VESC_CONNECT_SETTLE_MS));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("VESC connect failed")))
    }

    pub fn connect_default() -> Result<Self> {
        Self::connect(VESC_PORT)
    }

    fn open_port(port_path: &str) -> Result<Self> {
        let port = serialport::new(port_path, VESC_BAUDRATE)
            .timeout(Duration::from_millis(VESC_TIMEOUT_MS))
            .open()
            .with_context(|| format!("opening VESC port {port_path}"))?;
        Ok(Self {
            port,
            shutdown_brake_a: autopilot_config::SHUTDOWN_BRAKE_A,
        })
    }

    fn send_raw(&mut self, packet: &[u8]) -> Result<()> {
        self.port.write_all(packet)?;
        self.port.flush()?;
        Ok(())
    }

    pub fn set_duty(&mut self, duty: f32) -> Result<()> {
        self.send_raw(&encode_set_duty(duty)?)
    }

    pub fn set_brake(&mut self, amps: f32) -> Result<()> {
        self.send_raw(&encode_set_brake(amps)?)
    }

    pub fn set_servo(&mut self, pos: f32) -> Result<()> {
        self.send_raw(&encode_set_servo(pos)?)
    }

    pub fn servo_center(&mut self) -> Result<()> {
        self.set_servo(SERVO_CENTER)
    }

    pub fn safe_stop(&mut self) -> Result<()> {
        let _ = self.set_duty(0.0);
        let _ = self.set_brake(self.shutdown_brake_a);
        self.servo_center()
    }
}

impl Drop for VescClient {
    fn drop(&mut self) {
        let _ = self.safe_stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_has_framing() {
        let pkt = encode_set_servo(0.5).unwrap();
        assert_eq!(pkt[0], 2);
        assert_eq!(*pkt.last().unwrap(), 3);
    }
}
