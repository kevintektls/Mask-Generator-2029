//! Autopilot IA++ — hybrid autonomous driving for Jetson Nano.

use anyhow::{Context, Result};
use autopilot_camera::MonoCamera;
use autopilot_config::{adaptive_duty, Cli, EMERGENCY_BRAKE_A};
use autopilot_input::GamepadMonitor;
use autopilot_model::load_model;
use autopilot_stream::{encode_jpeg, publish_frame, start_server, FrameBuffer};
use autopilot_vision::{build_display_frame, detect_lines, mask_to_input};
use autopilot_vesc::VescClient;
use clap::Parser;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "autopilot=info".into()),
        )
        .init();

    let cli = Cli::parse();

    info!("initializing autopilot IA++");

    let model = load_model(&cli.model)
        .with_context(|| format!("loading model {}", cli.model.display()))?;

    let mut vesc = VescClient::connect(&cli.vesc_port)?;
    info!("VESC connected");

    let gamepad = GamepadMonitor::try_start()?;

    let frame_buffer: FrameBuffer = Arc::new(Mutex::new(None));
    let shutdown = start_server(cli.stream_port, Arc::clone(&frame_buffer)).await?;
    info!(
        "video stream http://localhost:{} (or Jetson IP)",
        cli.stream_port
    );

    let mut camera = MonoCamera::connect(&cli.camera_addr)?;

    vesc.servo_center()?;
    vesc.set_duty(0.0)?;
    std::thread::sleep(Duration::from_secs(1));

    info!("autopilot operational — LB = emergency brake");

    let run_flag = Arc::new(AtomicBool::new(true));
    {
        let flag = Arc::clone(&run_flag);
        ctrlc::set_handler(move || {
            info!("interrupt received — stopping");
            flag.store(false, Ordering::SeqCst);
        })?;
    }

    let run_result = run_loop(
        &model,
        &mut vesc,
        gamepad.as_ref(),
        &mut camera,
        &frame_buffer,
        &run_flag,
    );

    info!("shutting down systems");
    let _ = vesc.safe_stop();
    let _ = shutdown.send(());

    run_result
}

fn run_loop(
    model: &autopilot_model::BehavioralCloningCnn,
    vesc: &mut VescClient,
    gamepad: Option<&GamepadMonitor>,
    camera: &mut MonoCamera,
    frame_buffer: &FrameBuffer,
    run_flag: &AtomicBool,
) -> Result<()> {
    while run_flag.load(Ordering::SeqCst) {
        if let Some(pad) = gamepad {
            if pad.is_emergency() {
                info!("EMERGENCY: LB pressed — applying motor brake");
                vesc.set_brake(EMERGENCY_BRAKE_A)?;
                vesc.servo_center()?;
                break;
            }
        }

        let Some(gray) = camera.try_get_gray()? else {
            std::thread::sleep(Duration::from_micros(500));
            continue;
        };

        let mask = detect_lines(&gray);
        let input = mask_to_input(&mask)?;
        let prediction = model.predict_flat(&input)?;
        let servo_pos = prediction.clamp(0.0, 1.0);
        let current_duty = adaptive_duty(servo_pos);

        vesc.set_servo(servo_pos)?;
        vesc.set_duty(current_duty)?;

        let display = build_display_frame(&mask, servo_pos, current_duty)?;
        if let Ok(jpeg) = encode_jpeg(&display) {
            publish_frame(frame_buffer, jpeg);
        }
    }

    Ok(())
}
