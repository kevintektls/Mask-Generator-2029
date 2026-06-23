//! Grayscale camera frames over TCP from the Python DepthAI bridge.

use anyhow::{bail, Context, Result};
use autopilot_config::{CAMERA_BRIDGE_MAGIC, MONO_H, MONO_W};
use image::{GrayImage, Luma};
use std::io::Read;
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

const HEADER_LEN: usize = 4 + 4 + 4; // magic + w + h

/// TCP client for mono frames published by `tools/camera_bridge.py`.
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
        tracing::info!("camera bridge connected at {addr}");
        Ok(Self {
            stream,
            rx_buf: Vec::with_capacity(MONO_W as usize * MONO_H as usize + HEADER_LEN),
            scratch: vec![0u8; 4096],
        })
    }

    /// Non-blocking frame grab; returns `None` if no full frame is available yet.
    pub fn try_get_gray(&mut self) -> Result<Option<GrayImage>> {
        loop {
            if let Some(frame) = self.decode_frame()? {
                return Ok(Some(frame));
            }

            match self.stream.read(&mut self.scratch) {
                Ok(0) => bail!("camera bridge closed the connection"),
                Ok(n) => self.rx_buf.extend_from_slice(&self.scratch[..n]),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => return Ok(None),
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn decode_frame(&mut self) -> Result<Option<GrayImage>> {
        loop {
            if self.rx_buf.len() < HEADER_LEN {
                return Ok(None);
            }

            if &self.rx_buf[..4] != CAMERA_BRIDGE_MAGIC {
                if let Some(idx) = self.rx_buf.windows(4).position(|w| w == CAMERA_BRIDGE_MAGIC) {
                    self.rx_buf.drain(..idx);
                    continue;
                }
                self.rx_buf.clear();
                bail!("lost sync with camera bridge stream");
            }

            let w = u32::from_le_bytes(self.rx_buf[4..8].try_into().unwrap());
            let h = u32::from_le_bytes(self.rx_buf[8..12].try_into().unwrap());
            let payload = (w as usize)
                .checked_mul(h as usize)
                .context("invalid frame dimensions")?;
            let total = HEADER_LEN + payload;
            if self.rx_buf.len() < total {
                return Ok(None);
            }

            let pixels = &self.rx_buf[HEADER_LEN..total];
            let mut img = GrayImage::new(w, h);
            for (i, px) in img.pixels_mut().enumerate() {
                *px = Luma([pixels[i]]);
            }
            self.rx_buf.drain(..total);
            return Ok(Some(img));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_resolution_constants() {
        assert_eq!(MONO_W, 640);
        assert_eq!(MONO_H, 480);
    }
}
