#!/usr/bin/env python3
"""
OAK-D Lite Camera Viewer  —  DepthAI v2.29 API with MJPEG Streaming
Streams left mono, right mono, and optionally depth to a local HTTP server.

Requirements:
    pip install depthai==2.29.0 opencv-python numpy

Usage:
    python camera_website.py [FPS]
    python camera_website.py 20
    python camera_website.py 20 --depth

Controls:
    Ctrl+C   - Quit/Stop the server safely
    Browser  - Open http://localhost:8080 or http://<your-ip>:8080
"""

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

try:
    import depthai as dai
except ImportError:
    print("DepthAI not installed.  Run:  pip install depthai==2.29.0")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("OpenCV / NumPy not installed.  Run:  pip install opencv-python numpy")
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────
PREVIEW_SIZE = (640, 480)  # (width, height) for every displayed window
DEPTH_MAX_MM = 8_000  # Initial max depth shown (mm)
SNAPSHOT_DIR = Path("snapshots")
PORT = 8080
RECONNECT_DELAY_S = 5.0
USB2_MAX_FPS = 20
USB2_PREVIEW_SIZE = (640, 480)
USB2_DEPTH_PREVIEW_SIZE = (480, 360)
JPEG_QUALITY = 92

FPS = 20
ENABLE_DEPTH = False
STREAM_SIZE = PREVIEW_SIZE

COLORMAPS = [cv2.COLORMAP_TURBO, cv2.COLORMAP_BONE, cv2.COLORMAP_HSV]
COLORMAP_NAMES = ["TURBO", "BONE", "HSV"]

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "320x240": (320, 240),
    "480x360": (480, 360),
    "640x480": (640, 480),
}


class PipelineRestart(Exception):
    """Raised when the device pipeline must be rebuilt with new settings."""


class RuntimeSettings:
    def __init__(
        self,
        fps: int = 20,
        preview_size: tuple[int, int] = PREVIEW_SIZE,
        enable_depth: bool = False,
    ):
        self._lock = threading.Lock()
        self.fps = fps
        self.preview_size = preview_size
        self.enable_depth = enable_depth
        self.jpeg_quality = JPEG_QUALITY
        self.colormap_idx = 0
        self.depth_max_mm = DEPTH_MAX_MM
        self.show_fps = True
        self.usb2_mode = False
        self.restart_requested = False
        self.stream_size = preview_size

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            preset = next(
                (
                    name
                    for name, size in RESOLUTION_PRESETS.items()
                    if size == self.preview_size
                ),
                f"{self.preview_size[0]}x{self.preview_size[1]}",
            )
            return {
                "fps": self.fps,
                "resolution": preset,
                "jpeg_quality": self.jpeg_quality,
                "enable_depth": self.enable_depth,
                "colormap": COLORMAP_NAMES[self.colormap_idx],
                "depth_max_m": self.depth_max_mm // 1000,
                "show_fps": self.show_fps,
                "usb2_mode": self.usb2_mode,
                "stream_width": self.stream_size[0],
                "stream_height": self.stream_size[1],
                "restart_pending": self.restart_requested,
            }

    def update(self, data: dict[str, Any]) -> dict[str, Any]:
        restart = False
        with self._lock:
            if "jpeg_quality" in data:
                self.jpeg_quality = max(50, min(100, int(data["jpeg_quality"])))
            if "colormap" in data:
                name = str(data["colormap"]).upper()
                if name in COLORMAP_NAMES:
                    self.colormap_idx = COLORMAP_NAMES.index(name)
            if "depth_max_m" in data:
                meters = max(1, min(20, int(data["depth_max_m"])))
                self.depth_max_mm = meters * 1000
            if "show_fps" in data:
                self.show_fps = bool(data["show_fps"])

            if "fps" in data:
                new_fps = max(5, min(30, int(data["fps"])))
                if new_fps != self.fps:
                    self.fps = new_fps
                    restart = True
            if "resolution" in data:
                preset = str(data["resolution"])
                if preset in RESOLUTION_PRESETS:
                    new_size = RESOLUTION_PRESETS[preset]
                    if new_size != self.preview_size:
                        self.preview_size = new_size
                        restart = True
            if "enable_depth" in data:
                new_depth = bool(data["enable_depth"])
                if new_depth != self.enable_depth:
                    self.enable_depth = new_depth
                    restart = True

            if restart:
                self.restart_requested = True

        return {"restart": restart}

    def read_render_state(self) -> tuple[int, int, bool]:
        with self._lock:
            return self.colormap_idx, self.depth_max_mm, self.show_fps

    def consume_restart(self) -> bool:
        with self._lock:
            if self.restart_requested:
                self.restart_requested = False
                return True
            return False

    def set_stream_size(self, size: tuple[int, int]):
        with self._lock:
            self.stream_size = size


runtime_settings: Optional[RuntimeSettings] = None


# ── MJPEG Stream Server State & Logic ─────────────────────────────────────────
_stream_frames: dict[str, bytes] = {}
_stream_lock = threading.Lock()


def _encode_frame(frame: np.ndarray) -> bytes:
    quality = runtime_settings.jpeg_quality if runtime_settings else JPEG_QUALITY
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _update_stream(name: str, frame: np.ndarray):
    with _stream_lock:
        _stream_frames[name] = _encode_frame(frame)


class MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence noisy request logs in the terminal

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, code: int, payload: dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path == "/api/settings":
            if runtime_settings is None:
                self._send_json(503, {"ok": False, "error": "Settings unavailable"})
                return
            self._send_json(200, {"ok": True, "settings": runtime_settings.to_dict()})
            return

        settings = runtime_settings
        enable_depth = settings.enable_depth if settings else ENABLE_DEPTH
        stream_size = settings.stream_size if settings else STREAM_SIZE
        stream_fps = settings.fps if settings else FPS

        name = path.strip("/").upper() or None
        allowed = ("LEFT", "RIGHT", "DEPTH") if enable_depth else ("LEFT", "RIGHT")

        if name == "DEPTH" and not enable_depth:
            self.send_error(404)
            return

        if name and name in allowed:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            try:
                while True:
                    with _stream_lock:
                        frame = _stream_frames.get(name)
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                    time.sleep(1 / max(stream_fps, 1))
            except (ConnectionResetError, BrokenPipeError):
                pass
            except Exception as e:
                print(f"Stream error: {e}")
        elif name:
            self.send_error(404)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            depth_panel = (
                f'<div class="stream-card" id="depth-card" '
                f'style="display:{"block" if enable_depth else "none"};">'
                f'<h3>Depth Map</h3>'
                f'<img id="img-depth" src="/depth" '
                f'style="width:{stream_size[0]}px;height:{stream_size[1]}px;"></div>'
            )
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>OAK-D Lite Streams</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            background: #111;
            color: #eee;
            font-family: system-ui, sans-serif;
        }}
        .layout {{
            display: flex;
            min-height: 100vh;
            gap: 16px;
            padding: 16px;
        }}
        .sidebar {{
            width: 280px;
            flex-shrink: 0;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 16px;
            height: fit-content;
            position: sticky;
            top: 16px;
        }}
        .sidebar h2 {{ margin: 0 0 12px; font-size: 1.1rem; }}
        .sidebar label {{
            display: block;
            margin-bottom: 14px;
            font-size: 0.85rem;
            color: #bbb;
        }}
        .sidebar input[type="range"],
        .sidebar select {{
            width: 100%;
            margin-top: 6px;
        }}
        .sidebar .checkbox input {{ width: auto; margin-right: 8px; }}
        .sidebar button {{
            width: 100%;
            padding: 10px;
            margin-top: 8px;
            border: none;
            border-radius: 6px;
            background: #2d6cdf;
            color: white;
            cursor: pointer;
            font-weight: 600;
        }}
        .sidebar button.secondary {{ background: #333; }}
        .status {{
            margin-top: 12px;
            font-size: 0.8rem;
            color: #8fd;
            min-height: 1.2em;
        }}
        .streams {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-content: flex-start;
        }}
        .stream-card {{
            text-align: center;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 8px;
        }}
        .stream-card h3 {{ margin: 4px 0 8px; font-size: 0.95rem; }}
        .stream-card img {{
            display: block;
            border: 2px solid #333;
            background: #000;
        }}
    </style>
</head>
<body>
    <div class="layout">
        <aside class="sidebar">
            <h2>Settings</h2>
            <label>FPS (device)
                <input id="fps" type="range" min="5" max="30" step="1">
                <span id="fps_val"></span>
            </label>
            <label>Resolution
                <select id="resolution">
                    <option value="320x240">320x240</option>
                    <option value="480x360">480x360</option>
                    <option value="640x480">640x480</option>
                </select>
            </label>
            <label>JPEG quality
                <input id="jpeg_quality" type="range" min="50" max="100" step="1">
                <span id="jpeg_quality_val"></span>
            </label>
            <label class="checkbox">
                <input id="enable_depth" type="checkbox">
                Enable depth stream (restarts camera)
            </label>
            <div id="depth-controls">
                <label>Depth colormap
                    <select id="colormap">
                        <option value="TURBO">TURBO</option>
                        <option value="BONE">BONE</option>
                        <option value="HSV">HSV</option>
                    </select>
                </label>
                <label>Depth max (m)
                    <input id="depth_max_m" type="range" min="1" max="20" step="1">
                    <span id="depth_max_val"></span>
                </label>
            </div>
            <label class="checkbox">
                <input id="show_fps" type="checkbox">
                Show FPS overlay
            </label>
            <button id="apply_pipeline">Apply camera settings</button>
            <button class="secondary" id="reload_streams">Reload streams</button>
            <div class="status" id="status"></div>
        </aside>
        <main class="streams">
            <div class="stream-card">
                <h3>Left Eye</h3>
                <img id="img-left" src="/left" style="width:{stream_size[0]}px;height:{stream_size[1]}px;">
            </div>
            <div class="stream-card">
                <h3>Right Eye</h3>
                <img id="img-right" src="/right" style="width:{stream_size[0]}px;height:{stream_size[1]}px;">
            </div>
            {depth_panel}
        </main>
    </div>
    <script>
        const statusEl = document.getElementById("status");
        const instantIds = ["jpeg_quality", "colormap", "depth_max_m", "show_fps"];
        const pipelineIds = ["fps", "resolution", "enable_depth"];

        function setStatus(msg, ok = true) {{
            statusEl.textContent = msg;
            statusEl.style.color = ok ? "#8fd" : "#f88";
        }}

        function readForm() {{
            const data = {{
                fps: Number(document.getElementById("fps").value),
                resolution: document.getElementById("resolution").value,
                jpeg_quality: Number(document.getElementById("jpeg_quality").value),
                show_fps: document.getElementById("show_fps").checked,
            }};
            const depthToggle = document.getElementById("enable_depth");
            if (depthToggle) data.enable_depth = depthToggle.checked;
            const colormap = document.getElementById("colormap");
            if (colormap) data.colormap = colormap.value;
            const depthMax = document.getElementById("depth_max_m");
            if (depthMax) data.depth_max_m = Number(depthMax.value);
            return data;
        }}

        function applyLabels(data) {{
            document.getElementById("fps_val").textContent = data.fps;
            document.getElementById("jpeg_quality_val").textContent = data.jpeg_quality;
            const depthMaxVal = document.getElementById("depth_max_val");
            if (depthMaxVal) depthMaxVal.textContent = data.depth_max_m + " m";
        }}

        function fillForm(data) {{
            document.getElementById("fps").value = data.fps;
            document.getElementById("resolution").value = data.resolution;
            document.getElementById("jpeg_quality").value = data.jpeg_quality;
            document.getElementById("show_fps").checked = data.show_fps;
            const depthToggle = document.getElementById("enable_depth");
            if (depthToggle) depthToggle.checked = data.enable_depth;
            const colormap = document.getElementById("colormap");
            if (colormap) colormap.value = data.colormap;
            const depthMax = document.getElementById("depth_max_m");
            if (depthMax) depthMax.value = data.depth_max_m;
            applyLabels(data);
            resizeImages(data.stream_width, data.stream_height, data.enable_depth);
        }}

        function streamUrl(path) {{
            return path + "?t=" + Date.now();
        }}

        function resizeImages(w, h, showDepth) {{
            ["img-left", "img-right"].forEach(id => {{
                const img = document.getElementById(id);
                img.style.width = w + "px";
                img.style.height = h + "px";
            }});
            const depthImg = document.getElementById("img-depth");
            const depthCard = document.getElementById("depth-card");
            const depthControls = document.getElementById("depth-controls");
            if (depthImg) {{
                depthImg.style.width = w + "px";
                depthImg.style.height = h + "px";
            }}
            if (depthCard) depthCard.style.display = showDepth ? "block" : "none";
            if (depthControls) depthControls.style.display = showDepth ? "block" : "none";
        }}

        function reloadStreams(showDepth) {{
            document.getElementById("img-left").src = streamUrl("/left");
            document.getElementById("img-right").src = streamUrl("/right");
            const depthImg = document.getElementById("img-depth");
            if (depthImg && showDepth) depthImg.src = streamUrl("/depth");
        }}

        async function postSettings(data) {{
            const res = await fetch("/api/settings", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(data),
            }});
            return res.json();
        }}

        async function loadSettings() {{
            const res = await fetch("/api/settings");
            const payload = await res.json();
            if (payload.ok) fillForm(payload.settings);
        }}

        async function saveSettings(keys, message) {{
            const data = readForm();
            const partial = {{}};
            keys.forEach(k => {{ if (k in data) partial[k] = data[k]; }});
            try {{
                const payload = await postSettings(partial);
                if (!payload.ok) throw new Error(payload.error || "Save failed");
                fillForm(payload.settings);
                setStatus(message + (payload.restart ? " — camera restarting…" : ""));
                if (payload.restart) {{
                    setTimeout(() => reloadStreams(payload.settings.enable_depth), 2500);
                }}
            }} catch (err) {{
                setStatus(err.message, false);
            }}
        }}

        document.getElementById("apply_pipeline").addEventListener("click", () => {{
            saveSettings(pipelineIds, "Camera settings applied");
        }});

        document.getElementById("reload_streams").addEventListener("click", () => {{
            const data = readForm();
            reloadStreams(data.enable_depth);
            setStatus("Streams reloaded");
        }});

        instantIds.forEach(id => {{
            const el = document.getElementById(id);
            if (!el) return;
            const handler = () => {{
                applyLabels(readForm());
                saveSettings([id], "Updated " + id.replace("_", " "));
            }};
            el.addEventListener("change", handler);
            if (el.type === "range") el.addEventListener("input", () => applyLabels(readForm()));
        }});

        ["fps", "jpeg_quality"].forEach(id => {{
            const el = document.getElementById(id);
            el.addEventListener("input", () => applyLabels(readForm()));
        }});

        loadSettings();
        setInterval(loadSettings, 4000);
    </script>
</body>
</html>"""
            self.wfile.write(html.encode())

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/settings":
            self.send_error(404)
            return
        if runtime_settings is None:
            self._send_json(503, {"ok": False, "error": "Settings unavailable"})
            return
        try:
            data = self._read_json_body()
            result = runtime_settings.update(data)
            payload = {
                "ok": True,
                "restart": result["restart"],
                "settings": runtime_settings.to_dict(),
            }
            if result["restart"]:
                print("🔄 Pipeline restart requested from web UI")
            self._send_json(200, payload)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self._send_json(400, {"ok": False, "error": str(e)})


def start_mjpeg_server(port: int = 8080):
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(("0.0.0.0", port), MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"\n🚀 MJPEG stream launched successfully!")
    print(f"👉 Main Dashboard:  http://localhost:{port}/")
    print(f"👉 Settings API:    http://localhost:{port}/api/settings")
    enable_depth = runtime_settings.enable_depth if runtime_settings else ENABLE_DEPTH
    streams = f"http://localhost:{port}/left | /right"
    if enable_depth:
        streams += " | /depth"
    print(f"👉 Direct Streams: {streams}\n")
    return server


# ── Pipeline ───────────────────────────────────────────────────────────────────


def _is_usb2(speed) -> bool:
    return speed in (
        dai.UsbSpeed.HIGH,
        dai.UsbSpeed.FULL,
        dai.UsbSpeed.LOW,
    )


def _stream_settings(settings: RuntimeSettings) -> tuple[tuple[int, int], int]:
    fps = settings.fps
    preview_size = settings.preview_size
    if settings.usb2_mode:
        if settings.enable_depth:
            capped_size = _cap_resolution(preview_size, USB2_DEPTH_PREVIEW_SIZE)
            return capped_size, min(fps, 15)
        capped_size = _cap_resolution(preview_size, USB2_PREVIEW_SIZE)
        return capped_size, min(fps, USB2_MAX_FPS)
    return preview_size, fps


def _cap_resolution(
    requested: tuple[int, int], maximum: tuple[int, int]
) -> tuple[int, int]:
    if requested[0] * requested[1] <= maximum[0] * maximum[1]:
        return requested
    return maximum


NATIVE_MONO_SIZE = (640, 480)


def build_pipeline(
    preview_size: tuple[int, int], stream_fps: int, enable_depth: bool
):
    pipeline = dai.Pipeline()

    cam_left = pipeline.create(dai.node.MonoCamera)
    cam_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_left.setFps(stream_fps)

    cam_right = pipeline.create(dai.node.MonoCamera)
    cam_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    cam_right.setFps(stream_fps)

    def _xlink_out(name: str):
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName(name)
        xout.input.setBlocking(False)
        xout.input.setQueueSize(2)
        return xout

    def _resize_before_xlink(source, name: str, gray: bool = True):
        manip = pipeline.create(dai.node.ImageManip)
        manip.initialConfig.setResize(*preview_size)
        if gray:
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
        manip.setMaxOutputFrameSize(preview_size[0] * preview_size[1])
        source.link(manip.inputImage)
        xout = _xlink_out(name)
        manip.out.link(xout.input)

    def _mono_to_xlink(source, name: str):
        if preview_size == NATIVE_MONO_SIZE:
            xout = _xlink_out(name)
            source.link(xout.input)
        else:
            _resize_before_xlink(source, name)

    if enable_depth:
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setExtendedDisparity(False)
        stereo.setOutputSize(*preview_size)

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        _resize_before_xlink(stereo.rectifiedLeft, "left")
        _resize_before_xlink(stereo.rectifiedRight, "right")

        xout_depth = _xlink_out("depth")
        stereo.depth.link(xout_depth.input)
    else:
        _mono_to_xlink(cam_left.out, "left")
        _mono_to_xlink(cam_right.out, "right")

    return pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────


def colorize_depth(depth_frame: np.ndarray, colormap: int, max_mm: int) -> np.ndarray:
    clipped = np.clip(depth_frame, 0, max_mm).astype(np.float32)
    normed = (clipped / max_mm * 255).astype(np.uint8)
    return cv2.applyColorMap(normed, colormap)


def draw_fps(frame: np.ndarray, fps: float, label: str = "") -> np.ndarray:
    text = f"{label}  {fps:.1f} FPS" if label else f"{fps:.1f} FPS"
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return frame


def placeholder_frame(
    label: str, message: str, size: tuple[int, int] = PREVIEW_SIZE
) -> np.ndarray:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.putText(
        frame, label, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.putText(
        frame, message, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1
    )
    return frame


def show_reconnecting_streams(
    preview_size: tuple[int, int] = PREVIEW_SIZE, enable_depth: bool = False
):
    msg = "Reconnecting to device..."
    names = ["LEFT", "RIGHT"]
    if enable_depth:
        names.append("DEPTH")
    for name in names:
        _update_stream(name, placeholder_frame(name, msg, preview_size))


def safe_try_get(queue):
    try:
        return queue.tryGet()
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Queue read failed: {e}") from e


def stream_from_device(device, fps_data, settings: RuntimeSettings, preview_size: tuple[int, int]):
    enable_depth = settings.enable_depth
    q_left = device.getOutputQueue(name="left", maxSize=2, blocking=False)
    q_right = device.getOutputQueue(name="right", maxSize=2, blocking=False)
    q_depth = (
        device.getOutputQueue(name="depth", maxSize=2, blocking=False)
        if enable_depth
        else None
    )

    latest = {k: None for k in fps_data}

    queues = [("left", q_left), ("right", q_right)]
    if enable_depth:
        queues.append(("depth", q_depth))

    while True:
        if settings.consume_restart():
            raise PipelineRestart()

        colormap_idx, depth_max, show_fps = settings.read_render_state()

        for name, q in queues:
            pkt = safe_try_get(q)
            if pkt is not None:
                latest[name] = pkt
                tick_fps(fps_data, name)

        if latest["left"] is not None:
            f = latest["left"].getCvFrame()
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            if show_fps:
                f = draw_fps(f, fps_data["left"]["fps"], "LEFT")
            _update_stream("LEFT", f)

        if latest["right"] is not None:
            f = latest["right"].getCvFrame()
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            if show_fps:
                f = draw_fps(f, fps_data["right"]["fps"], "RIGHT")
            _update_stream("RIGHT", f)

        if enable_depth and latest["depth"] is not None:
            raw = latest["depth"].getFrame()
            f = colorize_depth(raw, COLORMAPS[colormap_idx], depth_max)
            if f.shape[1::-1] != preview_size:
                f = cv2.resize(f, preview_size)
            if show_fps:
                lbl = f"DEPTH [{COLORMAP_NAMES[colormap_idx]}] max={depth_max // 1000}m"
                f = draw_fps(f, fps_data["depth"]["fps"], lbl)
            _update_stream("DEPTH", f)

        time.sleep(0.001)


def tick_fps(fps_data, name: str) -> float:
    d = fps_data[name]
    d["count"] += 1
    now = time.monotonic()
    if now - d["t"] >= 1.0:
        d["fps"] = d["count"] / (now - d["t"])
        d["count"] = 0
        d["t"] = now
    return d["fps"]


def connect_and_stream(settings: RuntimeSettings, fps_data: dict):
    global STREAM_SIZE

    preview_size, stream_fps = _stream_settings(settings)
    settings.set_stream_size(preview_size)
    STREAM_SIZE = preview_size
    enable_depth = settings.enable_depth

    print("Building DepthAI v2.29 pipeline …")
    mode = "left + right + depth" if enable_depth else "left + right"
    print(f"📷 Streams: {mode}")
    print(
        f"🖼️  Preview: {preview_size[0]}x{preview_size[1]} @ {stream_fps} FPS "
        f"(JPEG q={settings.jpeg_quality})"
    )
    if settings.usb2_mode:
        print("⚙️  USB2 bandwidth limit active")
    pipeline = build_pipeline(preview_size, stream_fps, enable_depth)
    print("Starting pipeline …")

    with dai.Device(pipeline) as device:
        speed = device.getUsbSpeed()
        print("🔌 Vitesse USB détectée :", speed)
        if _is_usb2(speed) and not settings.usb2_mode:
            settings.usb2_mode = True
            usb2_preview, usb2_fps = _stream_settings(settings)
            if usb2_preview != preview_size or usb2_fps != stream_fps:
                print("⚠️  USB2 détecté — redémarrage avec bande passante réduite …")
                return "usb2"
            print("⚠️  USB2 détecté — qualité conservée (left/right uniquement)")

        print("Connected! Streaming online. Press Ctrl+C in terminal to stop.")
        try:
            stream_from_device(device, fps_data, settings, preview_size)
        except PipelineRestart:
            print("🔄 Restarting pipeline with new settings …")
            return "restart"
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="OAK-D Lite MJPEG camera streams")
    parser.add_argument(
        "fps",
        nargs="?",
        type=int,
        default=20,
        help="Target FPS (default: 20)",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="Enable depth stream (StereoDepth, heavier on USB bandwidth)",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────


def _build_fps_data(settings: RuntimeSettings) -> dict:
    stream_names = ["left", "right"]
    if settings.enable_depth:
        stream_names.append("depth")
    return {
        name: {"count": 0, "t": time.monotonic(), "fps": 0.0}
        for name in stream_names
    }


def main():
    global FPS, ENABLE_DEPTH, runtime_settings

    args = parse_args()
    FPS = args.fps
    ENABLE_DEPTH = args.depth

    runtime_settings = RuntimeSettings(
        fps=args.fps,
        preview_size=PREVIEW_SIZE,
        enable_depth=args.depth,
    )

    start_mjpeg_server(port=PORT)

    while True:
        fps_data = _build_fps_data(runtime_settings)
        preview_size = runtime_settings.stream_size
        try:
            result = connect_and_stream(runtime_settings, fps_data)
            if result in ("usb2", "restart"):
                continue
        except KeyboardInterrupt:
            print("\nShutting down pipeline and server gracefully …")
            break
        except RuntimeError as e:
            print(f"\n⚠️  Device communication error: {e}")
            show_reconnecting_streams(preview_size, runtime_settings.enable_depth)
            print(f"Retrying in {RECONNECT_DELAY_S:.0f}s … (Ctrl+C to stop)")
            time.sleep(RECONNECT_DELAY_S)
            runtime_settings.usb2_mode = True
        except Exception as e:
            print(f"\n⚠️  Unexpected device error: {e}")
            show_reconnecting_streams(preview_size, runtime_settings.enable_depth)
            print(f"Retrying in {RECONNECT_DELAY_S:.0f}s … (Ctrl+C to stop)")
            time.sleep(RECONNECT_DELAY_S)
            runtime_settings.usb2_mode = True


if __name__ == "__main__":
    main()
