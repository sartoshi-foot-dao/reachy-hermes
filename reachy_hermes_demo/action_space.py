"""Inspect Reachy Mini action-space surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent_tools import tool_specs
from .robot_tools import ANTENNAS, DANCES_DATASET, EMOTIONS_DATASET, POSES, build_toolbox


def summarize_recorded(dataset: str, sample: str | None = None) -> dict[str, Any]:
    from reachy_mini.motion.recorded_move import RecordedMoves

    moves = RecordedMoves(dataset)
    names = moves.list_moves()
    selected = sample if sample in names else (names[0] if names else None)
    summary: dict[str, Any] = {"dataset": dataset, "count": len(names), "names": names}
    if selected:
        move = moves.get(selected)
        first = move.trajectory[0] if move.trajectory else {}
        summary["sample"] = {
            "name": selected,
            "description": move.description,
            "duration_s": move.duration,
            "timestamps": len(move.timestamps),
            "trajectory_points": len(move.trajectory),
            "first_set_target_keys": sorted(first.keys()),
            "first_head_shape": [len(first.get("head", [])), len((first.get("head") or [[]])[0])],
            "first_antennas": first.get("antennas"),
            "first_body_yaw": first.get("body_yaw"),
            "sound_path": str(move.sound_path) if move.sound_path else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Reachy Mini action-space representation.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Print semantic poses, antenna states, built-in gestures, and tool schemas without querying the daemon.",
    )
    args = parser.parse_args()

    toolbox = build_toolbox()
    if args.offline:
        payload = {
            "poses_rad": POSES,
            "antennas_rad": ANTENNAS,
            "semantic_gestures": toolbox.list_gestures(),
            "tools": tool_specs(toolbox),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    payload = {
        "daemon": toolbox.status(),
        "semantic_gestures": toolbox.list_gestures(),
        "dances": summarize_recorded(DANCES_DATASET),
        "emotions": summarize_recorded(EMOTIONS_DATASET),
        "learned_move_dir": str(Path.home() / ".local/share/reachy-hermes-demo/moves"),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Reachy Mini action-space summary")
    print(json.dumps(payload["daemon"], indent=2, sort_keys=True))
    print()
    print("Semantic gestures:")
    print(json.dumps(payload["semantic_gestures"], indent=2, sort_keys=True))
    print()
    for key in ("dances", "emotions"):
        data = payload[key]
        print(f"{key}: {data['count']} recorded moves")
        print("names:", ", ".join(data["names"][:30]))
        sample = data.get("sample")
        if sample:
            print("sample:")
            print(json.dumps(sample, indent=2, sort_keys=True))
        print()


if __name__ == "__main__":
    main()
