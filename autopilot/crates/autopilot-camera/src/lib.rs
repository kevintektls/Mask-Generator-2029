//! Model-ready lane masks over TCP from the Python vision bridge.

use anyhow::{bail, Context, Result};
use autopilot_config::{CAMERA_BRIDGE_MAGIC, MASK_H, MASK_W};
use image::GrayImage;
use std::io::Read;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

const HEADER_LEN: usize = 4 + 4 + 4; // magic + w + h
const FRAME_BYTES: usize = (MASK_W * MASK_H) as usize;

/// TCP client for 160x120 masks from `tools/camera_bridge.py` (vision_preprocess).
pub struct MonoCamera {
    stream: TcpStream,
    rx_buf: Vec<u8>,
    scratch: Vec<u8>,
}

impl MonoCamera {
    /// Connect to the camera bridge (start `camera_bridge.py` first).
    pub fn connect(addr: &str) -> Result<Self> {
        let socket: SocketAddr = addr
            .parse()
            .with_context(|| format!("invalid camera address '{addr}'"))?;
        let stream = TcpStream::connect_timeout(&socket, Duration::from_secs(10))
            .with_context(|| format!("connecting to camera bridge at {addr}"))?;
        stream.set_read_timeout(Some(Duration::from_millis(2)))?;
        stream.set_nonblocking(true)?;
        stream.set_nodelay(true)?;
        tracing::info!("camera bridge connected at {addr}");
        Ok(Self {
            stream,
            rx_buf: Vec::with_capacity(FRAME_BYTES + HEADER_LEN + 4096),
            scratch: vec![0u8; 64 * 1024],
        })
    }

    /// Latest 160x120 mask from the bridge; drops older buffered frames.
    pub fn try_get_mask(&mut self) -> Result<Option<GrayImage>> {
        loop {
            match self.stream.read(&mut self.scratch) {
                Ok(0) => bail!("camera bridge closed the connection"),
                Ok(n) => self.rx_buf.extend_from_slice(&self.scratch[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => break,
                Err(e) => return Err(e.into()),
            }
        }

        let mut latest = None;
        while let Some(frame) = self.decode_one_frame()? {
            latest = Some(frame);
        }
        Ok(latest)
    }

    fn decode_one_frame(&mut self) -> Result<Option<GrayImage>> {
        loop {
            if self.rx_buf.len() < HEADER_LEN {
                return Ok(None);
            }

            if &self.rx_buf[..4] != CAMERA_BRIDGE_MAGIC {
                if let Some(idx) = self.rx_buf.windows(4).position(|w| w == CAMERA_BRIDGE_MAGIC) {
                    self.rx_buf.drain(..idx);
                    continue;
                }
                if self.rx_buf.len() > 256 * 1024 {
                    tracing::warn!("camera bridge: clearing desync buffer");
                    self.rx_buf.clear();
                }
                return Ok(None);
            }

            let w = u32::from_le_bytes(self.rx_buf[4..8].try_into().unwrap());
            let h = u32::from_le_bytes(self.rx_buf[8..12].try_into().unwrap());
            if w != MASK_W || h != MASK_H {
                tracing::warn!("camera bridge: unexpected frame {w}x{h}, resyncing");
                self.rx_buf.drain(..4);
                continue;
            }

            let total = HEADER_LEN + FRAME_BYTES;
            if self.rx_buf.len() < total {
                return Ok(None);
            }

            let pixels = self.rx_buf[HEADER_LEN..total].to_vec();
            self.rx_buf.drain(..total);
            let img = GrayImage::from_raw(w, h, pixels)
                .context("invalid mask frame buffer")?;
            return Ok(Some(img));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_mask_dimensions() {
        assert_eq!(MASK_W, 160);
        assert_eq!(MASK_H, 120);
        assert_eq!(FRAME_BYTES, 19_200);
    }
}
