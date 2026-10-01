//! OAK-D Lite JPEG frame receiver for the DepthAI v2.29 Python bridge.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const MAGIC: &[u8; 4] = b"OAK1";
const HEADER_SIZE: usize = 20;
const MAX_JPEG_BYTES: usize = 2 * 1024 * 1024;

#[derive(Debug, Serialize, serde::Deserialize)]
pub struct CameraFrameRecord {
    pub sequence: u64,
    /// Host wall-clock timestamp made by the Python bridge when it dequeues the frame.
    pub timestamp_unix_ns: u64,
    pub width: u16,
    pub height: u16,
    pub file: String,
}

pub fn record_camera_frames(addr: &str, output_dir: &Path) -> Result<()> {
    let address: SocketAddr = addr
        .parse()
        .with_context(|| format!("invalid camera bridge address '{addr}'"))?;
    let mut stream =
        TcpStream::connect_timeout(&address, Duration::from_secs(10)).with_context(|| {
            format!("connecting to OAK bridge at {addr}; start tools/oak_bridge.py first")
        })?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;

    let frames_dir = output_dir.join("frames");
    fs::create_dir_all(&frames_dir)
        .with_context(|| format!("creating {}", frames_dir.display()))?;
    let metadata_path = output_dir.join("frames.jsonl");
    let mut metadata = BufWriter::new(File::create(&metadata_path)?);
    tracing::info!(addr, output = %output_dir.display(), "recording OAK-D Lite frames; Ctrl-C to stop");

    let running = Arc::new(AtomicBool::new(true));
    {
        let running = Arc::clone(&running);
        ctrlc::set_handler(move || running.store(false, Ordering::SeqCst))
            .context("installing Ctrl-C handler")?;
    }

    let mut sequence = 0u64;
    while running.load(Ordering::SeqCst) {
        let frame = match read_frame(&mut stream) {
            Ok(frame) => frame,
            Err(error)
                if matches!(
                    error.downcast_ref::<std::io::Error>().map(|e| e.kind()),
                    Some(std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
                ) && !running.load(Ordering::SeqCst) =>
            {
                break;
            }
            Err(error) => return Err(error).context("receiving OAK camera frame"),
        };
        let filename = format!("frame_{sequence:08}.jpg");
        let path = frames_dir.join(&filename);
        fs::write(&path, &frame.jpeg).with_context(|| format!("writing {}", path.display()))?;
        let record = CameraFrameRecord {
            sequence,
            timestamp_unix_ns: frame.timestamp_unix_ns,
            width: frame.width,
            height: frame.height,
            file: PathBuf::from("frames")
                .join(filename)
                .to_string_lossy()
                .into_owned(),
        };
        serde_json::to_writer(&mut metadata, &record)?;
        metadata.write_all(b"\n")?;
        if sequence % 30 == 0 {
            metadata.flush()?;
            tracing::info!(
                sequence,
                timestamp_unix_ns = record.timestamp_unix_ns,
                "camera frame recorded"
            );
        }
        sequence += 1;
    }
    metadata.flush()?;
    tracing::info!(sequence, "camera recording stopped cleanly");
    Ok(())
}

struct CameraFrame {
    timestamp_unix_ns: u64,
    width: u16,
    height: u16,
    jpeg: Vec<u8>,
}

fn read_frame(stream: &mut impl Read) -> Result<CameraFrame> {
    let mut header = [0u8; HEADER_SIZE];
    stream.read_exact(&mut header)?;
    if &header[..4] != MAGIC {
        bail!("invalid OAK bridge frame magic");
    }
    let timestamp_unix_ns = u64::from_le_bytes(header[4..12].try_into().unwrap());
    let width = u16::from_le_bytes(header[12..14].try_into().unwrap());
    let height = u16::from_le_bytes(header[14..16].try_into().unwrap());
    let length = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    if width == 0 || height == 0 || width > 1920 || height > 1080 {
        bail!("invalid OAK image size {width}x{height}");
    }
    if length < 4 || length > MAX_JPEG_BYTES {
        bail!("invalid OAK JPEG payload length {length}");
    }
    let mut jpeg = vec![0u8; length];
    stream.read_exact(&mut jpeg)?;
    if jpeg[..2] != [0xff, 0xd8] || jpeg[length - 2..] != [0xff, 0xd9] {
        bail!("OAK bridge payload is not a complete JPEG image");
    }
    Ok(CameraFrame {
        timestamp_unix_ns,
        width,
        height,
        jpeg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn camera_protocol_rejects_bad_magic_and_accepts_jpeg_frame() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&42u64.to_le_bytes());
        bytes.extend_from_slice(&640u16.to_le_bytes());
        bytes.extend_from_slice(&480u16.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xd8, 0xff, 0xd9]);
        let frame = read_frame(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(frame.timestamp_unix_ns, 42);
        assert_eq!((frame.width, frame.height), (640, 480));

        let mut bad = vec![0; HEADER_SIZE];
        bad[4..12].copy_from_slice(&42u64.to_le_bytes());
        assert!(read_frame(&mut Cursor::new(bad)).is_err());
    }
}
