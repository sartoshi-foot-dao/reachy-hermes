# Reachy Hermes

A bounded semantic action-space and realtime voice bridge for running an LLM agent inside a desk [Reachy Mini](https://huggingface.co/docs/reachy_mini/SDK/core-concept) robot without giving the model raw motor access.

Context and write-up: [zidingz.com/semantic-tools-for-a-desk-robot/](https://zidingz.com/semantic-tools-for-a-desk-robot/) · earlier motivation note: [Why I'm putting Hermes into a robot](https://zidingz.com/why-im-putting-hermes-into-a-robot-reachy-mini/).

## Design boundary

The model never writes motor currents, serial packets, arbitrary trajectories, or Python. Instead:

1. **Realtime voice session** (`realtime_voice.py`) handles low-latency listening, speech interruption, and immediate tool dispatch.
2. **Semantic tool contract** (`agent_tools.py`, `robot_tools.py`) exposes bounded operations:
   - `look(direction)` — 9 named head directions
   - `gesture(name)` — 6 built-in multi-step gestures or validated learned gestures
   - `dance(move)` and `emotion(name)` — curated recorded-move datasets
   - `take_photo(question)` — local camera frame capture for multimodal grounding
   - `learn_gesture(name, steps, description)` — lets the agent compose new gestures up to 12 keyframes, clamping roll/pitch to `[-0.7, 0.7]` rad, yaw to `[-0.9, 0.9]` rad, body yaw to `[-0.8, 0.8]` rad, and step duration to `[0.12, 3.0]` s before `minjerk` interpolation on the local Reachy daemon (`127.0.0.1:8000`).

## Quick start (no robot required for inspection and tests)

```bash
python3 -m unittest discover -s tests -v
./scripts/reachy-hermes-action-space --offline
```

With a local Reachy Mini daemon running on `http://127.0.0.1:8000`:

```bash
./scripts/reachy-hermes-demo
OPENAI_API_KEY=... ./scripts/reachy-hermes-voice
```

Set `REACHY_PYTHON` if your `reachy_mini` SDK lives in a dedicated virtualenv.

## Documentation

- [docs/action_space.md](docs/action_space.md) — three motion layers and learned-gesture bounds
- [docs/architecture.md](docs/architecture.md) — split between realtime turn-taking and deeper agent planning
- [docs/system_design.md](docs/system_design.md) — local daemon, audio path, and failure boundaries

## License

MIT.
