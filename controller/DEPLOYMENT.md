# Run the controller as a service on Jetson Nano

This guide builds the Rust controller on the Jetson Nano and configures it to start at boot. The commands assume the Linux username is `jetson` and the repository is at `/home/jetson/Mask-Generator-2029`. Replace those paths and the username if yours are different.

## 1. Install build dependencies

Install the native packages used by the gamepad and serial-port crates:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libudev-dev
```

The controller uses Rust edition 2024, so use Rust 1.85 or newer. If Rust is already installed, check it with:

```bash
rustc --version
cargo --version
```

## 2. Build the controller

Copy or clone this repository onto the Jetson, then build the optimized binary:

```bash
cd /home/jetson/Mask-Generator-2029
cargo build --release --manifest-path controller/Cargo.toml
```

The binary will be at:

```text
/home/jetson/Mask-Generator-2029/controller/target/release/controller
```

## 3. Set the speed limit

Edit `controller/.env` on the Jetson:

```bash
nano /home/jetson/Mask-Generator-2029/controller/.env
```

Set the maximum duty cycle as a fraction from greater than `0` through `1`:

```env
MAX_DUTY_CYCLE=0.15
```

For example, `0.15` limits the maximum duty cycle to 15%. The controller loads this file from its directory when started by the service.

## 4. Check device names and permissions

Connect the VESC and gamepad, then inspect the serial port and device groups:

```bash
ls -l /dev/ttyACM*
getent group dialout input
```

The controller currently uses `/dev/ttyACM0` for the VESC. If your VESC appears under another name, update `VESC_PORT` in `controller/src/main.rs` and rebuild. The service below runs as `jetson` and adds the `dialout` and `input` supplementary groups for serial and gamepad access.

## 5. Create the systemd service

Create the unit file:

```bash
sudo nano /etc/systemd/system/robot-controller.service
```

Paste the following, adjusting the username and paths if needed:

```ini
[Unit]
Description=Robot Car Controller
After=multi-user.target

[Service]
Type=simple
User=jetson
SupplementaryGroups=dialout input
WorkingDirectory=/home/jetson/Mask-Generator-2029/controller
ExecStart=/home/jetson/Mask-Generator-2029/controller/target/release/controller
Restart=always
RestartSec=2
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
```

`KillSignal=SIGINT` lets the controller handle service stops with its safe-stop path. The controller waits for a gamepad at startup; if the gamepad disconnects while driving, the controller sends a stop and the service restarts it.

## 6. Enable and start it

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now robot-controller.service
sudo systemctl status robot-controller.service
```

`enable` configures the service to start at boot. `--now` starts it immediately as well.

View live logs:

```bash
journalctl -u robot-controller.service -f
```

Stop or restart it manually:

```bash
sudo systemctl stop robot-controller.service
sudo systemctl restart robot-controller.service
```

After changing `.env`, restart the service to load the new limit. After changing Rust code, rebuild and restart it:

```bash
cd /home/jetson/Mask-Generator-2029
cargo build --release --manifest-path controller/Cargo.toml
sudo systemctl restart robot-controller.service
```
