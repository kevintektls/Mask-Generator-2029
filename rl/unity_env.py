"""Environnement Gymnasium connecté au simulateur Unity via TCP."""

from __future__ import annotations

import socket
import struct
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.vision import (
    MASK_H,
    MASK_W,
    lateral_error_from_mask,
    rgb_to_observation,
    rgb_to_gray,
    detect_lines,
)

MAGIC_CMD = 0x524C434D  # "RLCM"
MAGIC_RSP = 0x524C5350  # "RLSP"
CMD_RESET = 0
CMD_STEP = 1
CMD_CLOSE = 2

HEADER_CMD = struct.Struct(">IB")          # magic, cmd
HEADER_RSP = struct.Struct(">II")          # magic, payload_len
STEP_ACTION = struct.Struct(">ff")         # steering, throttle
STEP_INFO = struct.Struct(">ffBB")         # lateral_error, speed, on_track, done


class UnityLaneEnv(gym.Env):
    """
    Observation : masque binaire (1, 120, 160) — identique au pilot_model.pth.
    Action      : [steering, throttle] dans [-1, 1] (SB3 Box normalisé).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7777,
        connect_timeout: float = 120.0,
        recv_timeout: float = 120.0,
        fixed_throttle: float | None = 0.35,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.recv_timeout = recv_timeout
        self.fixed_throttle = fixed_throttle

        self.observation_space = spaces.Box(0.0, 1.0, shape=(1, MASK_H, MASK_W), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._sock: socket.socket | None = None
        self._prev_steer_norm = 0.0
        self._steps = 0
        self._last_rgb: np.ndarray | None = None

    def _connect(self) -> None:
        deadline = time.time() + self.connect_timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=5.0)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self.recv_timeout)
                self._sock = sock
                print(f"[RL] Connecté à Unity {self.host}:{self.port}")
                return
            except OSError as exc:
                last_err = exc
                time.sleep(1.0)
        raise ConnectionError(
            f"Impossible de se connecter à Unity sur {self.host}:{self.port}. "
            f"Lance le simulateur en mode RL d'abord. ({last_err})"
        )

    def _send(self, data: bytes) -> None:
        assert self._sock is not None
        self._sock.sendall(data)

    def _recv_exactly(self, n: int) -> bytes:
        assert self._sock is not None
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Unity a fermé la connexion")
            buf += chunk
        return buf

    def _command(self, cmd: int, action: tuple[float, float] | None = None) -> dict[str, Any]:
        if self._sock is None:
            self._connect()

        payload = b""
        if cmd == CMD_STEP:
            assert action is not None
            steer_norm, throttle_norm = action
            steering = float(np.clip(steer_norm * 0.5 + 0.5, 0.0, 1.0))
            if self.fixed_throttle is not None:
                throttle = self.fixed_throttle
            else:
                throttle = float(np.clip(throttle_norm * 0.5 + 0.5, 0.0, 1.0))
            payload = STEP_ACTION.pack(steering, throttle)

        self._send(HEADER_CMD.pack(MAGIC_CMD, cmd) + payload)

        magic, payload_len = HEADER_RSP.unpack(self._recv_exactly(HEADER_RSP.size))
        if magic != MAGIC_RSP:
            raise ValueError(f"Magic response invalide: {magic:#x}")

        raw = self._recv_exactly(payload_len)
        if len(raw) < 8:
            raise ValueError("Payload Unity trop court")

        width, height = struct.unpack(">II", raw[:8])
        rgb_size = width * height * 3
        if len(raw) < 8 + rgb_size + STEP_INFO.size:
            raise ValueError("Payload image incomplet")

        rgb = np.frombuffer(raw[8 : 8 + rgb_size], dtype=np.uint8).reshape(height, width, 3)
        lateral_error, speed, on_track, done = STEP_INFO.unpack(
            raw[8 + rgb_size : 8 + rgb_size + STEP_INFO.size]
        )

        return {
            "rgb": rgb.copy(),
            "lateral_error": lateral_error,
            "speed": speed,
            "on_track": bool(on_track),
            "done": bool(done),
        }

    def _compute_reward(
        self,
        lateral_error: float,
        on_track: bool,
        speed: float,
        steer_norm: float,
        terminated: bool,
    ) -> float:
        if terminated and not on_track:
            return -50.0

        abs_err = abs(lateral_error)
        reward = 0.5  # bonus survie sur piste
        reward += 4.0 * (1.0 - min(1.0, abs_err))

        # Pénalité avant sortie de piste (anticipe les virages serrés)
        if abs_err > 0.3:
            reward -= 4.0 * (abs_err - 0.3)

        # Vitesse récompensée seulement si bien centré
        if abs_err < 0.2:
            reward += 0.1 * speed
        elif abs_err < 0.4:
            reward += 0.03 * speed

        reward -= 0.2 * abs(steer_norm - self._prev_steer_norm)
        return float(reward)

    def _info_from_rgb(self, rgb: np.ndarray) -> float:
        mask = detect_lines(rgb_to_gray(rgb))
        return lateral_error_from_mask(mask)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._prev_steer_norm = 0.0
        self._steps = 0
        state = self._command(CMD_RESET, action=(0.0, 0.0))
        self._last_rgb = state["rgb"]
        obs = rgb_to_observation(state["rgb"])
        info = {
            "lateral_error": state["lateral_error"],
            "speed": state["speed"],
            "on_track": state["on_track"],
        }
        return obs, info

    def step(self, action: np.ndarray):
        self._steps += 1
        steer_norm = float(action[0])
        throttle_norm = float(action[1]) if len(action) > 1 else 0.0

        state = self._command(CMD_STEP, action=(steer_norm, throttle_norm))
        self._last_rgb = state["rgb"]
        obs = rgb_to_observation(state["rgb"])

        terminated = state["done"] or not state["on_track"]
        truncated = self._steps >= 3000

        reward = self._compute_reward(
            state["lateral_error"],
            state["on_track"],
            state["speed"],
            steer_norm,
            terminated,
        )
        self._prev_steer_norm = steer_norm

        info = {
            "lateral_error": state["lateral_error"],
            "speed": state["speed"],
            "on_track": state["on_track"],
            "mask_lateral_error": self._info_from_rgb(state["rgb"]),
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sock is not None:
            try:
                self._send(HEADER_CMD.pack(MAGIC_CMD, CMD_CLOSE))
            except OSError:
                pass
            self._sock.close()
            self._sock = None
