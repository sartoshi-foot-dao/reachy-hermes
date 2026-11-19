# Reachy Mini Action Space

## Concrete Layers

Reachy Mini has three useful action layers.

1. Daemon/SDK primitives

These are immediate or interpolated robot controls:

- `set_target`: set head pose, antennas, and body yaw directly.
- `goto`: interpolate to a target over a duration.
- `wake_up` and `goto_sleep`: built-in choreographed moves.
- `play recorded move`: load a named recorded move and execute it.

The daemon REST route for `goto` accepts:

```json
{
  "head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
  "antennas": [-0.18, 0.18],
  "body_yaw": 0,
  "duration": 0.5,
  "interpolation": "minjerk"
}
```

2. `Move` objects

The Python motion interface is centered on a `Move` abstraction:

```python
move.duration
move.evaluate(t) -> (head_pose_4x4, antennas_rad, body_yaw_rad)
```

The daemon samples a `Move` at playback frequency and applies the returned targets.

3. Semantic tools

This demo exposes safe semantic calls to the agent:

- `look("left")`
- `gesture("curious")`
- `dance("simple_nod")`
- `emotion("amazed1")`
- `take_photo("what is on the desk?")`
- `learn_gesture("hesitant_yes", steps=[...])`

The model never writes raw motor currents, raw serial packets, or arbitrary Python.

## Recorded Moves

The built-in recorded move datasets are Hugging Face datasets:

- `pollen-robotics/reachy-mini-dances-library`
- `pollen-robotics/reachy-mini-emotions-library`

On this machine the inspector found:

- 19 dances
- 81 emotions

Each recorded move JSON has this effective shape:

```json
{
  "description": "A move description",
  "time": [0.0, 0.01, 0.02],
  "set_target_data": [
    {
      "head": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "antennas": [-0.18, 0.18],
      "body_yaw": 0,
      "check_collision": false
    }
  ]
}
```

Some emotions also have a same-name `.wav`; the daemon can play sound alongside motion.

## Learned Gestures

For dynamic agent behavior, this demo uses a deliberately smaller learned representation:

```json
{
  "name": "demo_hello",
  "description": "Looks left, perks antennas, then nods once.",
  "steps": [
    {"direction": "curious_left", "duration": 0.4, "hold": 0.1, "antennas": "perked"},
    {"direction": "down", "duration": 0.22, "hold": 0.03, "antennas": "perked"},
    {"direction": "up", "duration": 0.24, "hold": 0.03, "antennas": "perked"},
    {"direction": "front", "duration": 0.25, "antennas": "neutral"}
  ]
}
```

This is less expressive than raw recorded moves, but safer for an LLM to author. The code validates:

- max 12 steps
- bounded durations
- bounded body yaw
- known head directions
- known antenna poses

That gives Hermes a dynamic expression vocabulary without direct low-level motor access.

## Prior Art

Pollen's official conversation app already exposes a similar semantic surface: `move_head`, `camera`, `head_tracking`, `dance`, `play_emotion`, `remember`, and `forget`. Its motion manager queues primary moves while blending speech wobble and tracking offsets. This demo borrows that shape but trims it to a local command-line prototype.

Sources:

- Open-source app: https://github.com/pollen-robotics/reachy_mini_conversation_app
- Reachy Mini app integration docs: https://huggingface.co/docs/reachy_mini/SDK/integration
- Local inspected package: `~/.cache/reachy_mini_py/lib/python3.12/site-packages/reachy_mini/`

