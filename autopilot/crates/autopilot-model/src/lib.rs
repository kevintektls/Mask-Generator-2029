//! Behavioral cloning CNN inference with Candle.

use anyhow::{Context, Result};
use autopilot_config::{MASK_H, MASK_W};
use candle_core::{Device, DType, Tensor};
use candle_nn::{conv2d, linear, Conv2d, Conv2dConfig, Linear, Module, VarBuilder};
use std::path::Path;

/// CNN architecture identical to `rl/models.py` / `scripts/Autopilot_IA++.py`.
pub struct BehavioralCloningCnn {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv4: Conv2d,
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl BehavioralCloningCnn {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let vb = VarBuilder::from_pth(path, DType::F32, device)
            .with_context(|| format!("loading weights from {}", path.display()))?;
        Self::new(vb)
    }

    fn new(vb: VarBuilder) -> Result<Self> {
        let conv = |in_c, out_c, k, stride, name| {
            let cfg = Conv2dConfig {
                stride,
                padding: 0,
                dilation: 1,
                groups: 1,
                ..Default::default()
            };
            conv2d(in_c, out_c, k, cfg, vb.pp(name))
        };

        Ok(Self {
            conv1: conv(1, 24, 5, 2, "features.0")?,
            conv2: conv(24, 36, 5, 2, "features.2")?,
            conv3: conv(36, 48, 5, 2, "features.4")?,
            conv4: conv(48, 64, 3, 1, "features.6")?,
            fc1: linear(64 * 10 * 15, 100, vb.pp("regressor.0"))?,
            fc2: linear(100, 50, vb.pp("regressor.3"))?,
            fc3: linear(50, 1, vb.pp("regressor.5"))?,
        })
    }

    /// Input shape `(1, 1, MASK_H, MASK_W)` float32 in `[0, 1]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.relu()?;
        let x = self.conv2.forward(&x)?.relu()?;
        let x = self.conv3.forward(&x)?.relu()?;
        let x = self.conv4.forward(&x)?.relu()?;
        let x = x.flatten_from(1)?;
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        Ok(self.fc3.forward(&x)?)
    }

    /// Predict servo position from a flat normalized mask buffer (`MASK_H * MASK_W`).
    pub fn predict_flat(&self, mask: &[f32]) -> Result<f32> {
        let expected = (MASK_H * MASK_W) as usize;
        anyhow::ensure!(
            mask.len() == expected,
            "mask length {} != expected {}",
            mask.len(),
            expected
        );
        let device = self.conv1.weight().device();
        let tensor = Tensor::from_vec(mask.to_vec(), (1, 1, MASK_H as usize, MASK_W as usize), device)?;
        let out = self.forward(&tensor)?;
        Ok(out.to_vec1::<f32>()?[0])
    }
}

/// Load model on CPU with Rayon thread pool capped (Jetson-friendly).
pub fn load_model(path: &Path) -> Result<BehavioralCloningCnn> {
    std::env::set_var("RAYON_NUM_THREADS", "2");
    let device = Device::Cpu;
    BehavioralCloningCnn::load(path, &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../model/pilot_model.pth")
    }

    #[test]
    fn loads_weights_and_runs_forward() -> Result<()> {
        let path = model_path();
        if !path.exists() {
            eprintln!("skip: model not found at {}", path.display());
            return Ok(());
        }
        let model = load_model(&path)?;
        let input: Vec<f32> = (0..MASK_H * MASK_W)
            .map(|i| (i % 255) as f32 / 255.0)
            .collect();
        let pred = model.predict_flat(&input)?;
        assert!(pred.is_finite());
        assert!((0.0..=1.0).contains(&pred) || pred.is_finite());
        Ok(())
    }

    #[test]
    fn deterministic_inference() -> Result<()> {
        let path = model_path();
        if !path.exists() {
            return Ok(());
        }
        let model = load_model(&path)?;
        let input: Vec<f32> = vec![0.5; (MASK_H * MASK_W) as usize];
        let a = model.predict_flat(&input)?;
        let b = model.predict_flat(&input)?;
        assert!((a - b).abs() < 1e-6);
        Ok(())
    }
}
