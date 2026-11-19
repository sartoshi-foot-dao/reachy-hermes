from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reachy_hermes_demo.agent_tools import ToolDispatcher, tool_specs
from reachy_hermes_demo.robot_tools import ANTENNAS, BUILTIN_GESTURES, POSES, ReachyToolbox, RobotToolError


class SemanticToolBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.box = ReachyToolbox(
            daemon_url="http://127.0.0.1:8000",
            move_dir=base / "moves",
            photo_dir=base / "photos",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_tool_schemas_expose_only_semantic_surface(self) -> None:
        names = {item["name"] for item in tool_specs(self.box)}
        self.assertEqual(
            names,
            {
                "robot_status",
                "wake",
                "sleep",
                "look",
                "gesture",
                "dance",
                "emotion",
                "take_photo",
                "learn_gesture",
                "list_gestures",
            },
        )

    def test_learn_gesture_clamps_bounds_and_roundtrips(self) -> None:
        res = self.box.learn_gesture(
            name="hesitant yes",
            description="Looks left, perks antennas, then nods.",
            steps=[
                {"direction": "curious_left", "duration": 99.0, "hold": -2.0, "antennas": "perked", "body_yaw": 5.0},
                {"direction": "down", "duration": 0.01, "antennas": "neutral", "body_yaw": -5.0},
            ],
        )
        self.assertEqual(res["learned"], "hesitant_yes")
        loaded = self.box.load_learned_steps("hesitant_yes")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].duration, 3.0)
        self.assertEqual(loaded[0].hold, 0.0)
        self.assertEqual(loaded[0].body_yaw, 0.8)
        self.assertEqual(loaded[1].duration, 0.12)
        self.assertEqual(loaded[1].body_yaw, -0.8)

    def test_rejects_excess_steps_unknown_pose_and_unsafe_names(self) -> None:
        with self.assertRaises(RobotToolError):
            self.box.learn_gesture("too_long", [{"direction": "front"}] * 13)
        with self.assertRaises(RobotToolError):
            self.box.learn_gesture("bad_pose", [{"direction": "spin_360"}])
        with self.assertRaises(RobotToolError):
            self.box.learn_gesture("../escape", [{"direction": "front"}])

    def test_goto_payload_uses_minjerk_and_clamped_angles(self) -> None:
        sent = []

        def fake_post(path, payload=None, timeout=10.0):
            sent.append((path, payload))
            return {"status": "queued"}

        with patch.object(self.box, "post", side_effect=fake_post):
            dispatcher = ToolDispatcher(self.box)
            out = dispatcher.call("gesture", {"name": "nod_yes"})
        self.assertEqual(out["status"], "ok")
        goto_calls = [payload for path, payload in sent if path == "/api/move/goto"]
        self.assertEqual(len(goto_calls), len(BUILTIN_GESTURES["nod_yes"]))
        for payload in goto_calls:
            self.assertEqual(payload["interpolation"], "minjerk")
            hp = payload["head_pose"]
            self.assertTrue(-0.7 <= hp["roll"] <= 0.7)
            self.assertTrue(-0.7 <= hp["pitch"] <= 0.7)
            self.assertTrue(-0.9 <= hp["yaw"] <= 0.9)
            self.assertEqual(len(POSES) > 0 and len(ANTENNAS) > 0, True)


if __name__ == "__main__":
    unittest.main()
