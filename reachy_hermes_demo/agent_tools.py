"""Tool schemas and dispatcher for the Reachy Hermes demo."""

from __future__ import annotations

import json
from typing import Any

from .robot_tools import POSES, ANTENNAS, BUILTIN_GESTURES, RobotToolError, ReachyToolbox


def tool_specs(toolbox: ReachyToolbox) -> list[dict[str, Any]]:
    gestures = sorted(set(BUILTIN_GESTURES) | set(toolbox.list_gestures()["learned"]))
    return [
        {
            "type": "function",
            "name": "robot_status",
            "description": "Read Reachy Mini daemon, motor, media, and control-loop status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "look",
            "description": "Move Reachy's head to a simple gaze direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": sorted(POSES.keys()),
                    }
                },
                "required": ["direction"],
            },
        },
        {
            "type": "function",
            "name": "gesture",
            "description": "Play a built-in or learned expressive head gesture.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": gestures or sorted(BUILTIN_GESTURES.keys()),
                    }
                },
                "required": ["name"],
            },
        },
        {
            "type": "function",
            "name": "dance",
            "description": "Play a named recorded dance from the Reachy Mini dances dataset. Use sparingly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "move": {
                        "type": "string",
                        "description": "Recorded dance name. If omitted, the first available dance is used.",
                    },
                    "repeat": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                    },
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "emotion",
            "description": "Play a recorded emotion clip from the Reachy Mini emotions dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "Recorded emotion name. If omitted, the first available emotion is used.",
                    }
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "take_photo",
            "description": "Take one camera snapshot. The image is attached to the conversation for visual reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What the agent should answer about the captured image.",
                    }
                },
                "required": ["question"],
            },
        },
        {
            "type": "function",
            "name": "learn_gesture",
            "description": "Create or replace a learned safe gesture from semantic keyframes. Then use gesture(name) to play it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short gesture name using letters, numbers, dash, or underscore.",
                    },
                    "description": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "properties": {
                                "direction": {"type": "string", "enum": sorted(POSES.keys())},
                                "duration": {"type": "number", "minimum": 0.12, "maximum": 3.0},
                                "hold": {"type": "number", "minimum": 0.0, "maximum": 3.0},
                                "antennas": {"type": "string", "enum": sorted(ANTENNAS.keys())},
                                "body_yaw": {"type": "number", "minimum": -0.8, "maximum": 0.8},
                            },
                            "required": ["direction"],
                        },
                    },
                },
                "required": ["name", "steps"],
            },
        },
        {
            "type": "function",
            "name": "list_gestures",
            "description": "List built-in and learned gestures.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "wake",
            "description": "Wake Reachy into the default attentive pose.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "sleep",
            "description": "Put Reachy into its sleep pose.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    ]


class ToolDispatcher:
    """Execute model tool calls against the safe robot toolbox."""

    def __init__(self, toolbox: ReachyToolbox) -> None:
        self.toolbox = toolbox

    def call(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        args = arguments or {}
        try:
            if name == "robot_status":
                return self.toolbox.status()
            if name == "look":
                return self.toolbox.look(str(args["direction"]))
            if name == "gesture":
                return self.toolbox.gesture(str(args["name"]))
            if name == "dance":
                return self.toolbox.dance(args.get("move"), int(args.get("repeat", 1)))
            if name == "emotion":
                return self.toolbox.emotion(args.get("emotion"))
            if name == "take_photo":
                return self.toolbox.snapshot(str(args.get("question") or "What do you see?"))
            if name == "learn_gesture":
                return self.toolbox.learn_gesture(
                    str(args["name"]),
                    list(args["steps"]),
                    str(args.get("description") or ""),
                )
            if name == "list_gestures":
                return self.toolbox.list_gestures()
            if name == "wake":
                return self.toolbox.play_wake()
            if name == "sleep":
                return self.toolbox.play_sleep()
            return {"error": f"unknown tool {name!r}"}
        except (KeyError, ValueError, TypeError, RobotToolError) as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}


def sanitize_for_model(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky transport-only fields from tool output before model echo."""
    if tool_name == "take_photo" and "image_b64" in result:
        cleaned = dict(result)
        cleaned.pop("image_b64", None)
        cleaned["image_attached"] = True
        return cleaned
    return result
