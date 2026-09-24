use anyhow::{Context, Result};
use autopilot_vesc::VescClient;
use gilrs::{Axis, Button, Event, EventType, GamepadId, Gilrs};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const VESC_PORT: &str = "/dev/ttyACM0";
const DEADZONE: f32 = 0.08;
const MAX_DUTY_CYCLE: f32 = 0.15;
const SERVO_CENTER: f32 = 0.5;
const SERVO_RANGE: f32 = 0.5;
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const SLOW_FACTOR: f32 = 0.5;

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}

fn apply_deadzone(value: f32) -> f32 {
    if value.abs() < DEADZONE {
        0.0
    } else {
        value.signum() * (value.abs() - DEADZONE) / (1.0 - DEADZONE)
    }
}

fn triggers_to_duty(forward_raw: f32, backward_raw: f32) -> f32 {
    let throttle = apply_deadzone(clamp(forward_raw - backward_raw, -1.0, 1.0));
    clamp(throttle * MAX_DUTY_CYCLE, -MAX_DUTY_CYCLE, MAX_DUTY_CYCLE)
}

fn axis_to_servo(axis_value: f32) -> f32 {
    clamp(
        SERVO_CENTER + apply_deadzone(axis_value) * SERVO_RANGE,
        0.0,
        1.0,
    )
}

// Triggers may be exposed as analog buttons or as axes, depending on the OS mapping.
fn trigger_value(gamepad: &gilrs::Gamepad, button: Button, axis: Axis) -> f32 {
    if let Some(data) = gamepad.button_data(button) {
        return clamp(data.value(), 0.0, 1.0);
    }

    let value = gamepad.value(axis);
    if value < 0.0 {
        clamp((value + 1.0) * 0.5, 0.0, 1.0)
    } else {
        clamp(value, 0.0, 1.0)
    }
}

fn connected_gamepad(gilrs: &Gilrs) -> Option<GamepadId> {
    gilrs
        .gamepads()
        .find_map(|(id, pad)| pad.is_connected().then_some(id))
}

fn main() -> Result<()> {
    println!("[INFO] Robot Car Controller Starting...");
    let mut gilrs =
        Gilrs::new().map_err(|e| anyhow::anyhow!("initializing gamepad support: {e}"))?;
    let gamepad_id = loop {
        while let Some(Event { .. }) = gilrs.next_event() {}
        if let Some(id) = connected_gamepad(&gilrs) {
            break id;
        }
        println!("[INFO] Waiting for gamepad to be connected...");
        thread::sleep(Duration::from_secs(1));
    };
    println!(
        "[INFO] Gamepad connected: {}",
        gilrs.gamepad(gamepad_id).name()
    );

    println!("[INFO] Connecting to VESC on port {VESC_PORT}...");
    let mut vesc = VescClient::connect(VESC_PORT).context("connecting to VESC")?;
    thread::sleep(Duration::from_millis(500));
    println!("[INFO] RT = forward | LT = reverse | right stick = steering");
    println!(
        "[WARNING] Duty cycle is limited to {:.1}% for safety.",
        MAX_DUTY_CYCLE * 100.0
    );
    vesc.set_servo(SERVO_CENTER)?;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);
    ctrlc::set_handler(move || shutdown_flag.store(true, Ordering::SeqCst))
        .context("installing Ctrl-C handler")?;

    let result = drive_loop(&mut gilrs, gamepad_id, &mut vesc, &shutdown);
    println!("\n[INFO] Stopping the robot and cleaning up...");
    let stop_result = vesc.safe_stop();
    result?;
    stop_result?;
    Ok(())
}

fn drive_loop(
    gilrs: &mut Gilrs,
    gamepad_id: GamepadId,
    vesc: &mut VescClient,
    shutdown: &AtomicBool,
) -> Result<()> {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            println!("\n[INFO] Keyboard interrupt received, stopping the robot...");
            return Ok(());
        }
        while let Some(Event { id, event, .. }) = gilrs.next_event() {
            if id == gamepad_id && matches!(event, EventType::Disconnected) {
                println!("[WARNING] Gamepad disconnected; stopping.");
                return Ok(());
            }
        }

        let gamepad = gilrs.gamepad(gamepad_id);
        if !gamepad.is_connected() {
            println!("[WARNING] Gamepad disconnected; stopping.");
            return Ok(());
        }

        let forward = trigger_value(&gamepad, Button::RightTrigger2, Axis::RightZ);
        let backward = trigger_value(&gamepad, Button::LeftTrigger2, Axis::LeftZ);
        let mut duty = triggers_to_duty(forward, backward);
        if gamepad.is_pressed(Button::LeftTrigger) {
            duty *= SLOW_FACTOR;
        }

        vesc.set_duty(duty)?;
        vesc.set_servo(axis_to_servo(gamepad.value(Axis::LeftStickX)))?;
        thread::sleep(POLL_INTERVAL);
    }
}
