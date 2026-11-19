"""Typed Hermes-style tool loop for Reachy Mini."""

from __future__ import annotations

import argparse
import json
import shlex
from typing import Any

from .agent_tools import ToolDispatcher
from .robot_tools import build_toolbox


HELP = """Commands:
  status
  wake
  sleep
  look <front|left|right|up|down|tilt_left|tilt_right|curious_left|curious_right>
  gesture <attentive|nod_yes|shake_no|curious|confused|happy|learned_name>
  dance [move_name]
  emotion [emotion_name]
  photo [question...]
  list
  learn demo
  quit
"""


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def execute(dispatcher: ToolDispatcher, line: str) -> bool:
    parts = shlex.split(line)
    if not parts:
        return True
    cmd = parts[0].lower()
    if cmd in {"q", "quit", "exit"}:
        return False
    if cmd in {"h", "help", "?"}:
        print(HELP)
        return True
    if cmd == "status":
        print_json(dispatcher.call("robot_status"))
        return True
    if cmd == "wake":
        print_json(dispatcher.call("wake"))
        return True
    if cmd == "sleep":
        print_json(dispatcher.call("sleep"))
        return True
    if cmd == "look":
        print_json(dispatcher.call("look", {"direction": parts[1] if len(parts) > 1 else "front"}))
        return True
    if cmd == "gesture":
        print_json(dispatcher.call("gesture", {"name": parts[1] if len(parts) > 1 else "attentive"}))
        return True
    if cmd == "dance":
        print_json(dispatcher.call("dance", {"move": parts[1] if len(parts) > 1 else None}))
        return True
    if cmd == "emotion":
        print_json(dispatcher.call("emotion", {"emotion": parts[1] if len(parts) > 1 else None}))
        return True
    if cmd == "photo":
        question = " ".join(parts[1:]) or "What do you see?"
        result = dispatcher.call("take_photo", {"question": question})
        result.pop("image_b64", None)
        print_json(result)
        return True
    if cmd == "list":
        print_json(dispatcher.call("list_gestures"))
        return True
    if cmd == "learn" and len(parts) > 1 and parts[1] == "demo":
        print_json(
            dispatcher.call(
                "learn_gesture",
                {
                    "name": "demo_hello",
                    "description": "Looks left, perks antennas, then nods once.",
                    "steps": [
                        {"direction": "curious_left", "duration": 0.4, "hold": 0.1, "antennas": "perked"},
                        {"direction": "down", "duration": 0.22, "hold": 0.03, "antennas": "perked"},
                        {"direction": "up", "duration": 0.24, "hold": 0.03, "antennas": "perked"},
                        {"direction": "front", "duration": 0.25, "antennas": "neutral"},
                    ],
                },
            )
        )
        return True
    print(f"unknown command: {cmd}\n")
    print(HELP)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Typed Reachy Hermes tool demo.")
    parser.add_argument("--command", action="append", help="Run one command and exit; may be repeated.")
    args = parser.parse_args()

    dispatcher = ToolDispatcher(build_toolbox())
    if args.command:
        for command in args.command:
            execute(dispatcher, command)
        return

    print("Reachy Hermes typed demo. Type 'help' for commands.")
    while True:
        try:
            line = input("reachy> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not execute(dispatcher, line):
            break


if __name__ == "__main__":
    main()
