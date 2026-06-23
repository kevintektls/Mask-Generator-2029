//! MJPEG HTTP streaming server.

use anyhow::Result;
use autopilot_config::CAM_FPS;
use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use futures_util::stream;
use image::{ImageEncoder, RgbImage};
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::oneshot;

pub type FrameBuffer = Arc<Mutex<Option<Vec<u8>>>>;

#[derive(Clone)]
struct AppState {
    frame: FrameBuffer,
}

/// Encode an RGB frame to JPEG bytes.
pub fn encode_jpeg(rgb: &RgbImage) -> Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
    encoder.write_image(rgb.as_raw(), rgb.width(), rgb.height(), image::ExtendedColorType::Rgb8)?;
    Ok(buf.into_inner())
}

/// Publish a new frame to the shared buffer.
pub fn publish_frame(buffer: &FrameBuffer, jpeg: Vec<u8>) {
    if let Ok(mut guard) = buffer.lock() {
        *guard = Some(jpeg);
    }
}

/// Start the MJPEG HTTP server; returns a shutdown sender.
pub async fn start_server(port: u16, frame: FrameBuffer) -> Result<oneshot::Sender<()>> {
    let state = AppState { frame };

    let app = Router::new()
        .route("/", get(stream_handler))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("MJPEG stream at http://localhost:{port}");

    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async {
            let _ = rx.await;
        });
        if let Err(e) = server.await {
            tracing::error!("stream server error: {e}");
        }
    });

    Ok(tx)
}

async fn stream_handler(State(state): State<AppState>) -> Response {
    let boundary = "frame";
    let header_value = format!("multipart/x-mixed-replace; boundary={boundary}");
    let frame_interval = Duration::from_secs_f64(1.0 / CAM_FPS as f64);

    let body_stream = stream::unfold(state, move |state| async move {
        loop {
            let jpeg = state.frame.lock().ok().and_then(|g| g.clone());
            if let Some(buffer) = jpeg {
                let mut part = Vec::new();
                part.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
                part.extend_from_slice(b"Content-Type: image/jpeg\r\n");
                part.extend_from_slice(format!("Content-Length: {}\r\n\r\n", buffer.len()).as_bytes());
                part.extend_from_slice(&buffer);
                part.extend_from_slice(b"\r\n");
                return Some((Ok::<_, std::convert::Infallible>(part), state));
            }
            tokio::time::sleep(frame_interval).await;
        }
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, header_value)
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}
