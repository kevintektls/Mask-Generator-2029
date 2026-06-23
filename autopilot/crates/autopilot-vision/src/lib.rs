//! Lane-line vision pipeline (OpenCV), matching `rl/vision.py`.

use anyhow::Result;
use autopilot_config::{CROP_TOP_RATIO, MASK_H, MASK_W, ULTRA_BINARY_THRESH};
use opencv::{
    core::{Mat, Rect, Scalar, Size, CV_8UC1},
    imgproc::{
        self, COLOR_GRAY2BGR, GAUSSIAN_BLUR, INTER_AREA, LINE_8, MORPH_OPEN, MORPH_RECT,
        THRESH_BINARY,
    },
    prelude::*,
};

/// Extract binary lane mask from a grayscale camera frame.
pub fn detect_lines(frame_gray: &Mat) -> Result<Mat> {
    let h = frame_gray.rows();
    let w = frame_gray.cols();
    let start_y = (h as f64 * CROP_TOP_RATIO) as i32;
    let roi_h = h - start_y;

    let roi = Mat::roi(frame_gray, Rect::new(0, start_y, w, roi_h))?;
    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &roi,
        &mut blurred,
        Size::new(5, 5),
        0.0,
        0.0,
        opencv::core::BORDER_DEFAULT,
    )?;

    let mut binary_sol = Mat::default();
    imgproc::threshold(
        &blurred,
        &mut binary_sol,
        ULTRA_BINARY_THRESH as f64,
        255.0,
        THRESH_BINARY,
    )?;

    let mut clean_mask = Mat::zeros(h, w, CV_8UC1)?.to_mat()?;
    let mut roi_out = Mat::roi(&clean_mask, Rect::new(0, start_y, w, roi_h))?;
    binary_sol.copy_to(&mut roi_out)?;

    let kernel = imgproc::get_structuring_element(MORPH_RECT, Size::new(3, 3), (-1, -1))?;
    let mut opened = Mat::default();
    imgproc::morphology_ex(
        &clean_mask,
        &mut opened,
        MORPH_OPEN,
        &kernel,
        (-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;

    Ok(opened)
}

/// Resize mask to model input and normalize to `[0, 1]`.
pub fn mask_to_input(mask: &Mat) -> Result<Vec<f32>> {
    let mut resized = Mat::default();
    imgproc::resize(
        mask,
        &mut resized,
        Size::new(MASK_W as i32, MASK_H as i32),
        0.0,
        0.0,
        INTER_AREA,
    )?;

    let len = (MASK_W * MASK_H) as usize;
    let mut out = Vec::with_capacity(len);
    for y in 0..MASK_H as i32 {
        for x in 0..MASK_W as i32 {
            let v = *resized.at_2d::<u8>(y, x)? as f32 / 255.0;
            out.push(v);
        }
    }
    Ok(out)
}

/// Build a BGR display frame with status overlay for MJPEG streaming.
pub fn build_display_frame(mask: &Mat, servo: f32, duty: f32) -> Result<Mat> {
    use autopilot_config::{DISPLAY_H, DISPLAY_W};

    let mut display = Mat::default();
    imgproc::resize(
        mask,
        &mut display,
        Size::new(DISPLAY_W as i32, DISPLAY_H as i32),
        0.0,
        0.0,
        INTER_AREA,
    )?;

    let mut bgr = Mat::default();
    imgproc::cvt_color(&display, &mut bgr, COLOR_GRAY2BGR, 0)?;

    let status = format!("Servo: {servo:.2} | Duty: {duty:.3}");
    imgproc::put_text(
        &mut bgr,
        &status,
        opencv::core::Point::new(10, 20),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        1,
        LINE_8,
        false,
    )?;

    Ok(bgr)
}
