//! LDROBOT D500 / STL-19P scan acquisition and JSONL recording.
//!
//! The serial framing and angle conventions mirror the working reader in
//! `scripts/Behavioral_Cloning_Lidar.py`.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub mod mapping;

pub const PACKET_HEADER: u8 = 0x54;
pub const PACKET_VERLEN: u8 = 0x2c;
pub const PACKET_POINT_COUNT: usize = 12;
pub const PACKET_SIZE: usize = 47;
pub const SCAN_BINS: usize = 180;
pub const MAX_RANGE_M: f32 = 12.0;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScanRecord {
    pub schema: u32,
    pub sensor: String,
    /// UTC time when the final packet of this revolution was received.
    pub timestamp_unix_ms: u128,
    /// Original timestamp text from legacy CSV logs, whose timezone is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_timestamp: Option<String>,
    /// Monotonic time since this process started, useful for replay ordering.
    pub monotonic_ns: u128,
    pub duration_ms: f32,
    /// 180 one-degree bins from -90° (left) to +90° (right), in metres.
    pub ranges_m: Vec<f32>,
}

impl ScanRecord {
    fn new(
        timestamp_unix_ms: u128,
        monotonic_ns: u128,
        duration_ms: f32,
        ranges_m: Vec<f32>,
    ) -> Self {
        Self {
            schema: 1,
            sensor: "LDROBOT_D500_STL_19P".to_owned(),
            timestamp_unix_ms,
            source_timestamp: None,
            monotonic_ns,
            duration_ms,
            ranges_m,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub angle_deg: f32,
    pub distance_m: f32,
}

/// CRC-8 used by the D500 packet format (polynomial 0x4D, initial value 0).
pub fn crc8(data: &[u8]) -> u8 {
    let mut crc = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            crc = if crc & 0x80 != 0 {
                (crc << 1) ^ 0x4d
            } else {
                crc << 1
            };
        }
    }
    crc
}

pub fn decode_packet(packet: &[u8]) -> Result<[Point; PACKET_POINT_COUNT]> {
    if packet.len() != PACKET_SIZE {
        bail!(
            "D500 packet has {} bytes, expected {PACKET_SIZE}",
            packet.len()
        );
    }
    if packet[0] != PACKET_HEADER || packet[1] != PACKET_VERLEN {
        bail!("invalid D500 packet header/version");
    }
    if crc8(&packet[..PACKET_SIZE - 1]) != packet[PACKET_SIZE - 1] {
        bail!("D500 packet CRC mismatch");
    }

    let start = u16::from_le_bytes([packet[4], packet[5]]) as f32 / 100.0;
    let end = u16::from_le_bytes([packet[42], packet[43]]) as f32 / 100.0;
    let delta = (end - start).rem_euclid(360.0);
    let points = std::array::from_fn(|i| {
        let offset = 6 + i * 3;
        let distance_mm = u16::from_le_bytes([packet[offset], packet[offset + 1]]) as f32;
        Point {
            angle_deg: (start + delta * i as f32 / (PACKET_POINT_COUNT - 1) as f32)
                .rem_euclid(360.0),
            distance_m: distance_mm / 1000.0,
        }
    });
    Ok(points)
}

/// Extracts framed packets from the UART byte stream, resynchronizing on errors.
#[derive(Default)]
pub struct PacketDecoder {
    buffer: Vec<u8>,
}

impl PacketDecoder {
    pub fn push(&mut self, bytes: &[u8]) -> Vec<[Point; PACKET_POINT_COUNT]> {
        self.buffer.extend_from_slice(bytes);
        let mut packets = Vec::new();
        loop {
            let Some(header) = self.buffer.iter().position(|b| *b == PACKET_HEADER) else {
                self.buffer.clear();
                break;
            };
            if header > 0 {
                self.buffer.drain(..header);
            }
            if self.buffer.len() < 2 {
                break;
            }
            if self.buffer[1] != PACKET_VERLEN {
                self.buffer.drain(..1);
                continue;
            }
            if self.buffer.len() < PACKET_SIZE {
                break;
            }
            match decode_packet(&self.buffer[..PACKET_SIZE]) {
                Ok(points) => {
                    packets.push(points);
                    self.buffer.drain(..PACKET_SIZE);
                }
                Err(_) => {
                    // Drop one byte and search again, matching the Python reader's recovery.
                    self.buffer.drain(..1);
                }
            }
        }
        packets
    }
}

pub struct ScanAssembler {
    ranges: Vec<f32>,
    previous_angle: Option<f32>,
    valid_point_count: usize,
    scan_started: Option<Instant>,
    process_started: Instant,
    sequence: u64,
}

impl Default for ScanAssembler {
    fn default() -> Self {
        Self::new()
    }
}

impl ScanAssembler {
    pub fn new() -> Self {
        Self {
            ranges: vec![MAX_RANGE_M; SCAN_BINS],
            previous_angle: None,
            valid_point_count: 0,
            scan_started: None,
            process_started: Instant::now(),
            sequence: 0,
        }
    }

    /// Add one decoded packet. Returns a complete scan at a valid 0/360° wrap.
    pub fn add_packet(
        &mut self,
        points: &[Point; PACKET_POINT_COUNT],
        received_at: Instant,
    ) -> Option<(u64, ScanRecord)> {
        for point in points {
            if let Some(previous) = self.previous_angle {
                if point.angle_deg < previous - 180.0 {
                    if self.valid_point_count >= 100 {
                        let started = self.scan_started.unwrap_or(received_at);
                        let duration_ms =
                            received_at.duration_since(started).as_secs_f32() * 1000.0;
                        let monotonic_ns = self.process_started.elapsed().as_nanos();
                        let timestamp_unix_ms = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or(Duration::ZERO)
                            .as_millis();
                        let record = ScanRecord::new(
                            timestamp_unix_ms,
                            monotonic_ns,
                            duration_ms,
                            std::mem::replace(&mut self.ranges, vec![MAX_RANGE_M; SCAN_BINS]),
                        );
                        self.valid_point_count = 0;
                        self.scan_started = Some(received_at);
                        self.sequence += 1;
                        self.previous_angle = Some(point.angle_deg);
                        self.add_point(*point);
                        return Some((self.sequence, record));
                    }
                    self.ranges.fill(MAX_RANGE_M);
                    self.valid_point_count = 0;
                    self.scan_started = Some(received_at);
                }
            } else {
                self.scan_started = Some(received_at);
            }
            self.add_point(*point);
            self.previous_angle = Some(point.angle_deg);
        }
        None
    }

    fn add_point(&mut self, point: Point) {
        let signed_angle = if point.angle_deg <= 180.0 {
            point.angle_deg
        } else {
            point.angle_deg - 360.0
        };
        if (-90.0..=90.0).contains(&signed_angle)
            && point.distance_m > 0.0
            && point.distance_m <= MAX_RANGE_M
        {
            let bin = ((signed_angle + 90.0) as usize).min(SCAN_BINS - 1);
            self.ranges[bin] = self.ranges[bin].min(point.distance_m);
            self.valid_point_count += 1;
        }
    }
}

/// Reads complete scans from the configured UART. The UART format and defaults
/// follow the existing project collector: 8N1, 230400 baud, 12 samples/packet.
pub fn read_scans<F>(port_name: &str, baud_rate: u32, mut on_scan: F) -> Result<()>
where
    F: FnMut(u64, ScanRecord) -> Result<()>,
{
    let mut port = serialport::new(port_name, baud_rate)
        .data_bits(serialport::DataBits::Eight)
        .parity(serialport::Parity::None)
        .stop_bits(serialport::StopBits::One)
        .timeout(Duration::from_millis(200))
        .open()
        .with_context(|| format!("opening D500 UART {port_name} at {baud_rate} baud"))?;
    tracing::info!(port = port_name, baud_rate, "D500 UART opened");

    let mut decoder = PacketDecoder::default();
    let mut assembler = ScanAssembler::new();
    let mut bytes = [0u8; 4096];
    loop {
        match port.read(&mut bytes) {
            Ok(0) => continue,
            Ok(n) => {
                let received_at = Instant::now();
                for packet in decoder.push(&bytes[..n]) {
                    if let Some((sequence, scan)) = assembler.add_packet(&packet, received_at) {
                        on_scan(sequence, scan)?;
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::TimedOut => continue,
            Err(error) => return Err(error).context("reading D500 UART"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(start_deg: f32, end_deg: f32, distance_mm: u16) -> [u8; PACKET_SIZE] {
        let mut packet = [0u8; PACKET_SIZE];
        packet[0] = PACKET_HEADER;
        packet[1] = PACKET_VERLEN;
        packet[4..6].copy_from_slice(&((start_deg * 100.0) as u16).to_le_bytes());
        for i in 0..PACKET_POINT_COUNT {
            let offset = 6 + i * 3;
            packet[offset..offset + 2].copy_from_slice(&distance_mm.to_le_bytes());
        }
        packet[42..44].copy_from_slice(&((end_deg * 100.0) as u16).to_le_bytes());
        packet[46] = crc8(&packet[..46]);
        packet
    }

    #[test]
    fn decodes_distance_and_interpolated_angles() {
        let points = decode_packet(&make_packet(10.0, 21.0, 1250)).unwrap();
        assert_eq!(
            points[0],
            Point {
                angle_deg: 10.0,
                distance_m: 1.25
            }
        );
        assert!((points[11].angle_deg - 21.0).abs() < 1e-5);
        assert_eq!(points[6].distance_m, 1.25);
    }

    #[test]
    fn rejects_corrupt_crc_and_resynchronizes() {
        let mut corrupt = make_packet(0.0, 11.0, 1000);
        corrupt[15] ^= 1;
        assert!(decode_packet(&corrupt).is_err());

        let valid = make_packet(20.0, 31.0, 2000);
        let mut decoder = PacketDecoder::default();
        let mut stream = vec![0, 1, 2];
        stream.extend_from_slice(&corrupt);
        stream.extend_from_slice(&valid);
        let packets = decoder.push(&stream);
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0][0].distance_m, 2.0);
    }

    #[test]
    fn serializes_and_reloads_jsonl_record() {
        let record = ScanRecord::new(123, 456, 50.0, vec![MAX_RANGE_M; SCAN_BINS]);
        let line = serde_json::to_string(&record).unwrap();
        let restored: ScanRecord = serde_json::from_str(&line).unwrap();
        assert_eq!(restored, record);
        assert_eq!(restored.ranges_m.len(), SCAN_BINS);
    }
}
