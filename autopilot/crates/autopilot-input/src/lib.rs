//! Gamepad emergency-stop input via gilrs.

use anyhow::Result;
use gilrs::{Button, EventType, Gilrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const POLL_INTERVAL_MS: u64 = 20;

/// Background-polled gamepad watcher. LB (Left Bumper) triggers emergency stop.
pub struct GamepadMonitor {
    emergency: Arc<AtomicBool>,
    _handle: Option<thread::JoinHandle<()>>,
}

impl GamepadMonitor {
    /// Start monitoring; returns `None` if no gamepad is found.
    pub fn try_start() -> Result<Option<Self>> {
        let mut gilrs = match Gilrs::new() {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("gilrs init failed: {e}");
                return Ok(None);
            }
        };

        if gilrs.gamepads().next().is_none() {
            tracing::info!("no gamepad detected — emergency LB disabled");
            return Ok(None);
        }

        tracing::info!("gamepad connected for emergency stop (LB)");
        let emergency = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&emergency);

        let handle = thread::spawn(move || {
            let mut lb_pressed = false;
            loop {
                while let Some(Event { event, .. }) = gilrs.next_event() {
                    if let EventType::ButtonPressed(btn, _) = event {
                        if is_emergency_button(btn) {
                            lb_pressed = true;
                            flag.store(true, Ordering::SeqCst);
                        }
                    }
                    if let EventType::ButtonRepeated(btn, _) = event {
                        if is_emergency_button(btn) {
                            lb_pressed = true;
                            flag.store(true, Ordering::SeqCst);
                        }
                    }
                    if let EventType::ButtonReleased(btn, _) = event {
                        if is_emergency_button(btn) {
                            lb_pressed = false;
                        }
                    }
                }

                // Also poll current state for held LB
                for (_id, gamepad) in gilrs.gamepads() {
                    if gamepad.is_pressed(Button::LeftTrigger2) {
                        lb_pressed = true;
                        flag.store(true, Ordering::SeqCst);
                    }
                }

                if lb_pressed {
                    flag.store(true, Ordering::SeqCst);
                }

                thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
            }
        });

        Ok(Some(Self {
            emergency,
            _handle: Some(handle),
        }))
    }

    pub fn is_emergency(&self) -> bool {
        self.emergency.load(Ordering::SeqCst)
    }
}

fn is_emergency_button(btn: Button) -> bool {
    btn == Button::LeftTrigger2
}

use gilrs::Event;
