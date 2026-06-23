//! DepthAI mono camera pipeline (CAM_B, 480p, grayscale).

use anyhow::{Context, Result};
use depthai::{
    camera::{CameraNode, CameraOutputConfig, ImageFrame, OutputQueue},
    common::{CameraBoardSocket, ImageFrameType, ResizeMode},
    device::Device,
    pipeline::Pipeline,
};
use opencv::core::Mat;

const MONO_W: u32 = 640;
const MONO_H: u32 = 480;
/// OAK-D Lite left mono camera stream.
pub struct MonoCamera {
    _device: Device,
    _pipeline: Pipeline,
    queue: OutputQueue,
}

impl MonoCamera {
    /// Open CAM_B mono camera at 640×480 @ `fps`.
    pub fn open(fps: u32) -> Result<Self> {
        let device = Device::new().context("connecting DepthAI device")?;
        let pipeline = Pipeline::with_device(&device)?;

        let cam = pipeline
            .create_with::<CameraNode, _>(CameraBoardSocket::CamB)
            .context("creating mono camera on CAM_B")?;

        let out = cam.request_output(CameraOutputConfig {
            size: (MONO_W, MONO_H),
            frame_type: Some(ImageFrameType::GRAY8),
            resize_mode: ResizeMode::Crop,
            fps: Some(fps as f32),
            enable_undistortion: None,
        })?;

        let queue = out.create_queue(2, false)?;

        pipeline.start().context("starting DepthAI pipeline")?;

        tracing::info!("DepthAI mono camera ready ({MONO_W}x{MONO_H} @ {fps} fps)");

        Ok(Self {
            _device: device,
            _pipeline: pipeline,
            queue,
        })
    }

    /// Non-blocking frame grab; returns `None` if the queue is empty.
    pub fn try_get_gray(&self) -> Result<Option<Mat>> {
        match self.queue.try_next()? {
            Some(frame) => Ok(Some(frame_to_mat(&frame)?)),
            None => Ok(None),
        }
    }
}

fn frame_to_mat(frame: &ImageFrame) -> Result<Mat> {
    let w = frame.width();
    let h = frame.height();
    let bytes = frame.bytes();
    let expected = (w as usize) * (h as usize);
    anyhow::ensure!(
        bytes.len() >= expected,
        "frame bytes {} < expected {}",
        bytes.len(),
        expected
    );

    let mat = Mat::from_slice(&bytes[..expected])
        .context("creating Mat from frame bytes")?
        .reshape(1, h as i32)
        .context("reshaping frame")?
        .try_clone()
        .context("cloning frame Mat")?;

    Ok(mat)
}

#[cfg(test)]
mod tests {
    #[test]
    fn mono_resolution_matches_python() {
        assert_eq!(super::MONO_W, 640);
        assert_eq!(super::MONO_H, 480);
    }
}
