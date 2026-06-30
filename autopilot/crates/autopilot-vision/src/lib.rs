//! Lane-line vision pipeline (pure Rust), matching `rl/vision.py`.

use anyhow::Result;
use autopilot_config::{CROP_TOP_RATIO, DISPLAY_H, DISPLAY_W, MASK_H, MASK_W, ULTRA_BINARY_THRESH};
use image::{imageops::FilterType, GrayImage, Luma, Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use imageproc::filter::gaussian_blur_f32;

/// Extract binary lane mask from a grayscale camera frame.
pub fn detect_lines(frame_gray: &GrayImage) -> GrayImage {
    let (w, h) = frame_gray.dimensions();
    let start_y = (h as f64 * CROP_TOP_RATIO) as u32;
    let roi_h = h.saturating_sub(start_y);

    let roi = image::imageops::crop_imm(frame_gray, 0, start_y, w, roi_h).to_image();
    let blurred = gaussian_blur_gray(&roi);
    let binary_roi = threshold_binary(&blurred, ULTRA_BINARY_THRESH);

    let mut clean_mask = GrayImage::from_pixel(w, h, Luma([0]));
    for y in 0..roi_h {
        for x in 0..w {
            clean_mask.put_pixel(x, start_y + y, *binary_roi.get_pixel(x, y));
        }
    }

    morphology_open_3x3(&clean_mask)
}

/// Resize mask to model input and normalize to `[0, 1]`.
pub fn mask_to_input(mask: &GrayImage) -> Result<Vec<f32>> {
    let resized = if mask.dimensions() == (MASK_W, MASK_H) {
        mask.clone()
    } else {
        image::imageops::resize(mask, MASK_W, MASK_H, FilterType::Triangle)
    };
    Ok(resized
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect())
}

/// Upscale a model mask for MJPEG display.
pub fn upscale_mask_for_display(mask: &GrayImage) -> GrayImage {
    if mask.dimensions() == (DISPLAY_W, DISPLAY_H) {
        return mask.clone();
    }
    image::imageops::resize(mask, DISPLAY_W, DISPLAY_H, FilterType::Triangle)
}

/// Build an RGB display frame with status overlay for MJPEG streaming.
pub fn build_display_frame(mask: &GrayImage, servo: f32, duty: f32) -> Result<RgbImage> {
    let resized = image::imageops::resize(mask, DISPLAY_W, DISPLAY_H, FilterType::Triangle);
    let mut rgb = RgbImage::new(DISPLAY_W, DISPLAY_H);
    for (x, y, p) in resized.enumerate_pixels() {
        let v = p[0];
        rgb.put_pixel(x, y, Rgb([v, v, v]));
    }

    let status = format!("Servo: {servo:.2} | Duty: {duty:.3}");
    if let Some(font) = load_overlay_font() {
        draw_text_mut(
            &mut rgb,
            Rgb([0, 255, 0]),
            10,
            8,
            ab_glyph::PxScale::from(16.0),
            &font,
            &status,
        );
    }

    Ok(rgb)
}

fn gaussian_blur_gray(img: &GrayImage) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut f32_img = image::ImageBuffer::<Luma<f32>, Vec<f32>>::new(w, h);
    for (x, y, p) in img.enumerate_pixels() {
        f32_img.put_pixel(x, y, Luma([p[0] as f32]));
    }
    let blurred = gaussian_blur_f32(&f32_img, 1.1);
    let mut out = GrayImage::new(w, h);
    for (x, y, p) in blurred.enumerate_pixels() {
        out.put_pixel(x, y, Luma([p[0].round().clamp(0.0, 255.0) as u8]));
    }
    out
}

fn threshold_binary(img: &GrayImage, thresh: u8) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = img.get_pixel(x, y)[0];
            out.put_pixel(x, y, Luma([if v >= thresh { 255 } else { 0 }]));
        }
    }
    out
}

fn morphology_open_3x3(img: &GrayImage) -> GrayImage {
    morphology_dilate_3x3(&morphology_erode_3x3(img))
}

fn morphology_dilate_3x3(img: &GrayImage) -> GrayImage {
    morph_op(img, |vals| *vals.iter().max().unwrap())
}

fn morphology_erode_3x3(img: &GrayImage) -> GrayImage {
    morph_op(img, |vals| *vals.iter().min().unwrap())
}

fn morph_op(img: &GrayImage, op: impl Fn(&[u8]) -> u8) -> GrayImage {
    let (w, h) = img.dimensions();
    let mut out = GrayImage::from_pixel(w, h, Luma([0]));
    for y in 0..h {
        for x in 0..w {
            let mut vals = Vec::with_capacity(9);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && (nx as u32) < w && (ny as u32) < h {
                        vals.push(img.get_pixel(nx as u32, ny as u32)[0]);
                    } else {
                        vals.push(0);
                    }
                }
            }
            out.put_pixel(x, y, Luma([op(&vals)]));
        }
    }
    out
}

fn load_overlay_font() -> Option<ab_glyph::FontArc> {
    const PATHS: &[&str] = &[
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ];
    for path in PATHS {
        if let Ok(bytes) = std::fs::read(path) {
            if let Ok(font) = ab_glyph::FontArc::try_from_vec(bytes) {
                return Some(font);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_lines_output_size_matches_input() {
        let frame = GrayImage::from_pixel(640, 480, Luma([128]));
        let mask = detect_lines(&frame);
        assert_eq!(mask.dimensions(), (640, 480));
    }
}
