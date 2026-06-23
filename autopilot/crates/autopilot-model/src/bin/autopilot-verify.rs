//! CLI to run a single inference (used by `tools/verify_model.py`).

use anyhow::{Context, Result};
use autopilot_model::load_model;
use std::io::Read;
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: autopilot-verify <model.pth>")?;

    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf)?;

    let floats: Vec<f32> = buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let model = load_model(&model_path)?;
    let pred = model.predict_flat(&floats)?;
    println!("{pred}");
    Ok(())
}
