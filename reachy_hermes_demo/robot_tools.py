"""Safe semantic tool layer for Reachy Mini.

The model never receives direct motor APIs. It can call named gestures,
recorded dances/emotions, camera capture, and learned gesture playback.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

DEFAULT_DAEMON_URL = "http://127.0.0.1:8000"
DEFAULT_MOVE_DIR = Path.home() / ".local/share/reachy-hermes-demo/moves"
DEFAULT_PHOTO_DIR = Path.home() / ".local/share/reachy-hermes-demo/photos"

DANCES_DATASET = "pollen-robotics/reachy-mini-dances-library"
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class RobotToolError(RuntimeError):
    """Raised when a robot tool cannot complete."""


@dataclass(frozen=True)
class GestureStep:
    """One safe gesture keyframe."""

    direction: str = "front"
    duration: float = 0.45
    hold: float = 0.0
    antennas: str = "neutral"
    body_yaw: float = 0.0


POSES: dict[str, dict[str, float]] = {
    "front": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "left": {"roll": 0.0, "pitch": 0.0, "yaw": math.radians(28)},
    "right": {"roll": 0.0, "pitch": 0.0, "yaw": math.radians(-28)},
    "up": {"roll": 0.0, "pitch": math.radians(-18), "yaw": 0.0},
    "down": {"roll": 0.0, "pitch": math.radians(18), "yaw": 0.0},
    "tilt_left": {"roll": math.radians(18), "pitch": 0.0, "yaw": 0.0},
    "tilt_right": {"roll": math.radians(-18), "pitch": 0.0, "yaw": 0.0},
    "curious_left": {"roll": math.radians(13), "pitch": math.radians(-8), "yaw": math.radians(20)},
    "curious_right": {"roll": math.radians(-13), "pitch": math.radians(-8), "yaw": math.radians(-20)},
}

ANTENNAS: dict[str, tuple[float, float]] = {
    "neutral": (-0.18, 0.18),
    "perked": (-0.55, 0.55),
    "droop": (0.55, -0.55),
    "opposite": (0.45, 0.45),
    "flat": (0.0, 0.0),
}

BUILTIN_GESTURES: dict[str, list[GestureStep]] = {
    "attentive": [
        GestureStep("front", 0.35, 0.05, "perked"),
    ],
    "nod_yes": [
        GestureStep("down", 0.22, 0.02, "perked"),
        GestureStep("up", 0.26, 0.02, "perked"),
        GestureStep("down", 0.22, 0.02, "perked"),
        GestureStep("front", 0.25, 0.0, "neutral"),
    ],
    "shake_no": [
        GestureStep("left", 0.25, 0.02, "neutral"),
        GestureStep("right", 0.32, 0.02, "neutral"),
        GestureStep("left", 0.32, 0.02, "neutral"),
        GestureStep("front", 0.25, 0.0, "neutral"),
    ],
    "curious": [
        GestureStep("curious_left", 0.45, 0.12, "perked"),
        GestureStep("curious_right", 0.55, 0.12, "perked"),
        GestureStep("front", 0.35, 0.0, "neutral"),
    ],
    "confused": [
        GestureStep("tilt_left", 0.35, 0.10, "opposite"),
        GestureStep("tilt_right", 0.45, 0.10, "opposite"),
        GestureStep("front", 0.30, 0.0, "neutral"),
    ],
    "happy": [
        GestureStep("up", 0.28, 0.02, "perked"),
        GestureStep("tilt_left", 0.25, 0.02, "perked"),
        GestureStep("tilt_right", 0.25, 0.02, "perked"),
        GestureStep("front", 0.30, 0.0, "neutral"),
    ],
}


def _json_request(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except urllib.error.URLError as exc:
        raise RobotToolError(f"Reachy daemon request failed: {method} {url}: {exc}") from exc


def _safe_name(name: str) -> str:
    normalized = name.strip().replace(" ", "_")
    if not NAME_RE.match(normalized):
        raise RobotToolError("name must be 1-64 chars: letters, numbers, dash, underscore")
    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


class ReachyToolbox:
    """High-level operations exposed to an agent."""

    def __init__(
        self,
        daemon_url: str = DEFAULT_DAEMON_URL,
        move_dir: Path = DEFAULT_MOVE_DIR,
        photo_dir: Path = DEFAULT_PHOTO_DIR,
    ) -> None:
        self.daemon_url = daemon_url.rstrip("/")
        self.move_dir = move_dir
        self.photo_dir = photo_dir
        self.move_dir.mkdir(parents=True, exist_ok=True)
        self.photo_dir.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, timeout: float = 10.0) -> Any:
        return _json_request("GET", self.daemon_url + path, timeout=timeout)

    def post(self, path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
        return _json_request("POST", self.daemon_url + path, payload=payload or {}, timeout=timeout)

    def status(self) -> dict[str, Any]:
        daemon = self.get("/api/daemon/status", timeout=5.0)
        motors = self.get("/api/motors/status", timeout=5.0)
        media = self.get("/api/media/status", timeout=5.0)
        backend = daemon.get("backend_status") or {}
        stats = backend.get("control_loop_stats") or {}
        return {
            "daemon": daemon.get("state"),
            "motor_mode": motors.get("mode"),
            "media": media,
            "control_loop_hz": stats.get("mean_control_loop_frequency"),
            "error": backend.get("error") or daemon.get("error"),
        }

    def enable_motors(self) -> dict[str, Any]:
        return self.post("/api/motors/set_mode/enabled", timeout=8.0)

    def wait_move(self, move_id: str | UUID, timeout: float = 12.0) -> None:
        deadline = time.monotonic() + timeout
        target = str(move_id)
        while time.monotonic() < deadline:
            running = self.get("/api/move/running", timeout=3.0)
            if not any(str(item.get("uuid")) == target for item in running):
                return
            time.sleep(0.08)
        raise RobotToolError(f"move {target} did not finish within {timeout:.1f}s")

    def goto_step(self, step: GestureStep) -> dict[str, Any]:
        pose = POSES.get(step.direction)
        if pose is None:
            raise RobotToolError(f"unknown direction {step.direction!r}")
        antennas = ANTENNAS.get(step.antennas)
        if antennas is None:
            raise RobotToolError(f"unknown antenna pose {step.antennas!r}")

        payload = {
            "head_pose": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": _clamp(pose["roll"], -0.7, 0.7),
                "pitch": _clamp(pose["pitch"], -0.7, 0.7),
                "yaw": _clamp(pose["yaw"], -0.9, 0.9),
            },
            "antennas": [_clamp(antennas[0], -1.0, 1.0), _clamp(antennas[1], -1.0, 1.0)],
            "body_yaw": _clamp(step.body_yaw, -0.8, 0.8),
            "duration": _clamp(step.duration, 0.12, 3.0),
            "interpolation": "minjerk",
        }
        result = self.post("/api/move/goto", payload=payload, timeout=8.0)
        if isinstance(result, dict) and "uuid" in result:
            self.wait_move(result["uuid"], timeout=payload["duration"] + 5.0)
        if step.hold > 0:
            time.sleep(_clamp(step.hold, 0.0, 3.0))
        return result

    def look(self, direction: str) -> dict[str, Any]:
        direction = direction.strip().lower()
        self.enable_motors()
        self.goto_step(GestureStep(direction=direction, duration=0.55, antennas="neutral"))
        return {"status": "ok", "look": direction}

    def gesture(self, name: str) -> dict[str, Any]:
        name = _safe_name(name)
        steps = BUILTIN_GESTURES.get(name)
        source = "builtin"
        if steps is None:
            steps = self.load_learned_steps(name)
            source = "learned"
        self.enable_motors()
        for step in steps:
            self.goto_step(step)
        return {"status": "ok", "gesture": name, "source": source, "steps": len(steps)}

    def list_gestures(self) -> dict[str, Any]:
        learned = sorted(path.stem for path in self.move_dir.glob("*.json"))
        return {"builtin": sorted(BUILTIN_GESTURES.keys()), "learned": learned}

    def play_wake(self) -> dict[str, Any]:
        self.enable_motors()
        result = self.post("/api/move/play/wake_up", timeout=8.0)
        if isinstance(result, dict) and "uuid" in result:
            self.wait_move(result["uuid"], timeout=8.0)
        return {"status": "ok", "move": "wake_up"}

    def play_sleep(self) -> dict[str, Any]:
        result = self.post("/api/move/play/goto_sleep", timeout=8.0)
        if isinstance(result, dict) and "uuid" in result:
            self.wait_move(result["uuid"], timeout=10.0)
        return {"status": "ok", "move": "goto_sleep"}

    def list_recorded(self, dataset: str) -> list[str]:
        quoted = urllib.parse.quote(dataset, safe="/")
        return self.get(f"/api/move/recorded-move-datasets/list/{quoted}", timeout=15.0)

    def dance(self, move: str | None = None, repeat: int = 1) -> dict[str, Any]:
        moves = self.list_recorded(DANCES_DATASET)
        selected = move or (moves[0] if moves else None)
        if selected not in moves:
            raise RobotToolError(f"unknown dance {selected!r}; available={moves}")
        repeat = int(_clamp(repeat, 1, 3))
        self.enable_motors()
        for _ in range(repeat):
            path = f"/api/move/play/recorded-move-dataset/{DANCES_DATASET}/{urllib.parse.quote(selected)}"
            result = self.post(path, timeout=12.0)
            if isinstance(result, dict) and "uuid" in result:
                self.wait_move(result["uuid"], timeout=30.0)
        return {"status": "ok", "dance": selected, "repeat": repeat}

    def emotion(self, emotion: str | None = None) -> dict[str, Any]:
        moves = self.list_recorded(EMOTIONS_DATASET)
        selected = emotion or (moves[0] if moves else None)
        if selected not in moves:
            raise RobotToolError(f"unknown emotion {selected!r}; available={moves}")
        self.enable_motors()
        path = f"/api/move/play/recorded-move-dataset/{EMOTIONS_DATASET}/{urllib.parse.quote(selected)}"
        result = self.post(path, timeout=12.0)
        if isinstance(result, dict) and "uuid" in result:
            self.wait_move(result["uuid"], timeout=20.0)
        return {"status": "ok", "emotion": selected}

    def normalize_steps(self, steps: list[dict[str, Any]]) -> list[GestureStep]:
        if not steps or len(steps) > 12:
            raise RobotToolError("learned gesture must contain 1-12 steps")
        normalized: list[GestureStep] = []
        for raw in steps:
            direction = str(raw.get("direction", "front")).strip().lower()
            if direction not in POSES:
                raise RobotToolError(f"unknown direction {direction!r}")
            antennas = str(raw.get("antennas", "neutral")).strip().lower()
            if antennas not in ANTENNAS:
                raise RobotToolError(f"unknown antenna pose {antennas!r}")
            normalized.append(
                GestureStep(
                    direction=direction,
                    duration=_clamp(float(raw.get("duration", 0.45)), 0.12, 3.0),
                    hold=_clamp(float(raw.get("hold", 0.0)), 0.0, 3.0),
                    antennas=antennas,
                    body_yaw=_clamp(float(raw.get("body_yaw", 0.0)), -0.8, 0.8),
                )
            )
        return normalized

    def learn_gesture(self, name: str, steps: list[dict[str, Any]], description: str = "") -> dict[str, Any]:
        name = _safe_name(name)
        normalized = self.normalize_steps(steps)
        path = self.move_dir / f"{name}.json"
        payload = {
            "name": name,
            "description": description.strip()[:300],
            "steps": [step.__dict__ for step in normalized],
            "created_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": "ok", "learned": name, "steps": len(normalized), "path": str(path)}

    def load_learned_steps(self, name: str) -> list[GestureStep]:
        path = self.move_dir / f"{_safe_name(name)}.json"
        if not path.exists():
            raise RobotToolError(f"unknown gesture {name!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.normalize_steps(payload.get("steps") or [])

    def snapshot(self, question: str = "What do you see?") -> dict[str, Any]:
        import numpy as np
        from reachy_mini import ReachyMini

        self.photo_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        jpg_path = self.photo_dir / f"reachy-{stamp}.jpg"

        with ReachyMini(connection_mode="localhost_only", media_backend="default", timeout=8.0) as reachy:
            frame = None
            for _ in range(30):
                frame = reachy.media.get_frame()
                if frame is not None:
                    break
                time.sleep(0.1)
            if frame is None:
                raise RobotToolError("camera did not produce a frame")

        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RobotToolError(f"unexpected camera frame shape {frame.shape}")

        height, width = frame.shape[:2]
        rgb = np.ascontiguousarray(frame[..., ::-1])
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-q:v",
            "3",
            "-f",
            "image2",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, input=rgb.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not proc.stdout:
            raise RobotToolError(f"ffmpeg JPEG encode failed: {proc.stderr.decode('utf-8', 'replace')}")
        jpg_path.write_bytes(proc.stdout)
        return {
            "status": "ok",
            "question": question,
            "path": str(jpg_path),
            "width": width,
            "height": height,
            "jpeg_bytes": len(proc.stdout),
            "image_b64": base64.b64encode(proc.stdout).decode("ascii"),
        }


def build_toolbox() -> ReachyToolbox:
    daemon_url = os.environ.get("REACHY_DAEMON_URL", DEFAULT_DAEMON_URL)
    return ReachyToolbox(daemon_url=daemon_url)
