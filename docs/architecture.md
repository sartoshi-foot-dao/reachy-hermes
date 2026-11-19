# Reachy Hermes Architecture

## Purpose

This demo explores how to run a Hermes-style agent inside Reachy Mini without
making the robot feel like a slow chatbot. The primary design decision is to
separate realtime turn-taking from deeper agency:

- The realtime voice session owns low-latency listening, interruption, speech
  synthesis, and small immediate tool calls.
- The robot tool layer owns all physical effects and camera access.
- A future Hermes sidecar owns longer planning, memory, and skill evolution, but
  uses the same tool contract as the realtime model.

The result should feel like Reachy is directly responding through voice and
motion while still leaving room for a more capable agent loop behind it.

## Design Constraints

- Local Linux setup talking directly to the Reachy Mini daemon; the desktop GUI app is not in the required path.
- Reachy Mini is attached over USB and controlled through the local daemon on
  `http://127.0.0.1:8000`.
- The OpenAI API key is local configuration. It is not checked in, echoed, or
  embedded in scripts.
- Motion must be bounded. The model does not receive raw serial access, motor
  currents, arbitrary Python execution, or unvalidated movement matrices.
- Camera images can be captured locally, but image bytes are treated as
  transport data and are not echoed back through generic tool output.
- The first useful interface is command-line and voice; no dashboard dependency.

## Runtime Topology

```text
Reachy USB mic
  -> local audio capture process
  -> OpenAI Realtime WebSocket session
      -> fast speech-to-speech response
      -> model-visible function calls
      -> optional image input after camera capture
  -> local tool dispatcher
      -> validated semantic robot tools
      -> Reachy daemon REST API
      -> Reachy Python SDK for camera frames
  -> Reachy USB speaker and motors

Future sidecar:

Realtime session transcripts/tool events
  -> Hermes planner and memory process
  -> same semantic tool dispatcher
  -> Reachy daemon/SDK
```

The realtime session is intentionally not the only place where intelligence can
live. It is the latency-critical voice surface. Hermes should be able to observe
the same transcript and tool-result stream, then emit tool calls for slower
tasks without blocking turn-taking.

## Components

### `realtime_voice.py`

Owns the live voice loop:

- captures PCM from Reachy's USB microphone with `parec`
- sends 24 kHz mono PCM chunks to OpenAI Realtime over WebSocket
- plays returned PCM through Reachy's USB speaker with `pw-play` or `paplay`
- registers the robot tool schema in `session.update`
- handles `response.function_call_arguments.done`
- returns tool outputs as `function_call_output` items
- attaches camera snapshots as `input_image` items after `take_photo`

The default model is `gpt-realtime-2`, with `reasoning.effort` set to `low` so
the voice path remains responsive. The system prompt encourages short speech,
small gestures, and visible tool use only when the tool reports success.

### `agent_tools.py`

Defines the public contract the model can call:

```text
robot_status()
look(direction)
gesture(name)
dance(move, repeat)
emotion(emotion)
take_photo(question)
learn_gesture(name, steps)
list_gestures()
wake()
sleep()
```

It also dispatches tool calls into the robot toolbox and strips transport-only
fields such as base64 image bytes before returning results to the model.

### `robot_tools.py`

Owns the physical boundary:

- talks to the daemon REST API for status, wake/sleep, `goto`, recorded dances,
  and recorded emotions
- uses the Reachy Python SDK for camera frames
- encodes camera frames to JPEG through `ffmpeg`
- stores learned gestures as JSON under
  `~/.local/share/reachy-hermes-demo/moves/`
- stores photos under `~/.local/share/reachy-hermes-demo/photos/`
- clamps durations, head poses, antenna values, body yaw, and step counts

This layer is intentionally semantic. A model can ask for `curious_left` or
`nod_yes`; it cannot write arbitrary transforms or run arbitrary code.

### Operations Scripts

The demo exposes direct scripts and a Taskfile:

```text
reachy-on   -> start daemon with media, wake robot, print status
reachy-off  -> sleep robot, stop daemon
task on     -> reachy-on
task off    -> reachy-off
task sleep  -> sleep only, daemon stays running
task voice  -> realtime voice mode
```

The direct scripts are available even if Go Task is not installed.

## Action Space

The action space has three layers.

1. Daemon and SDK primitives

These are low-level enough to move hardware directly:

- `goto` interpolates to a target head pose, antenna pair, and body yaw.
- `wake_up` and `goto_sleep` play built-in daemon moves.
- recorded move playback loads a named Hugging Face dataset move.
- camera capture streams frames through the SDK media backend.

2. Recorded `Move` objects

Reachy's Python motion layer represents moves with:

```python
move.duration
move.evaluate(t) -> (head_pose_4x4, antennas_rad, body_yaw_rad)
```

Recorded dances and emotions are time series of sampled target data. On this
machine the local inspector found 19 dances and 81 emotions.

3. Agent-safe semantic gestures

The model-facing layer narrows motion to safe named directions and antenna
states:

```json
{
  "name": "hesitant_yes",
  "steps": [
    {"direction": "down", "duration": 0.2, "hold": 0.05, "antennas": "perked"},
    {"direction": "front", "duration": 0.25, "antennas": "neutral"}
  ]
}
```

This is the dynamic-expression path. Hermes can author a new gesture, persist
it, and then reuse it by name.

## Event Flow: Simple Conversation

```text
user speaks
  -> realtime VAD commits a turn
  -> model answers with audio
  -> optional small gesture tool call
  -> dispatcher executes motion
  -> tool result is returned
```

The model is allowed to speak before or around a visible action, but it is told
not to claim the action happened unless the tool result says it worked.

## Event Flow: Visual Question

```text
user asks "what do you see?"
  -> realtime model calls take_photo(question)
  -> robot tool captures one JPEG
  -> tool output returns file metadata, not raw image text
  -> realtime handler attaches the JPEG as input_image
  -> model answers from the image
```

The image handoff is separated from generic tool output so large base64 payloads
do not pollute the tool-result channel.

## Event Flow: Learned Motion

```text
user asks Reachy to invent a small expressive motion
  -> model calls learn_gesture(name, steps)
  -> robot tool validates and saves JSON
  -> model calls gesture(name)
  -> robot tool plays validated keyframes through daemon goto
```

This gives the agent a form of embodied skill learning without giving it
unbounded motion control.

## Hermes Sidecar Plan

The future Hermes process should be a local sidecar, not a replacement for the
Realtime session. Its inputs:

- user transcripts
- assistant transcripts
- tool calls
- tool results
- image metadata
- learned gesture definitions

Its outputs:

- calls to the same semantic tools
- memory writes
- suggestions for new learned gestures
- longer multi-step plans

The sidecar can run a more capable, slower model because it is not responsible
for every speech token. For example, the realtime voice can answer "I'll take a
look" immediately, while Hermes decides whether to compare multiple observations
or remember an object location.

## Safety Model

The current safety boundary is code, not prompting:

- fixed enum of head directions
- fixed enum of antenna poses
- max 12 learned gesture steps
- duration clamp per step
- body yaw clamp
- explicit name validation
- no shell/tool execution exposed to the model
- no arbitrary REST paths exposed to the model
- no raw camera byte echo in ordinary model-visible results

Failures are returned as structured tool errors and should be spoken plainly by
the agent.

## Current Status

Verified locally:

- daemon starts without the desktop app
- motors can be enabled
- sleep and wake moves work
- action-space inspector sees 19 dances and 81 emotions
- typed tool loop can learn and play a gesture
- camera snapshot saves a 1920x1080 JPEG
- voice wrapper starts and validates key presence

Not yet complete:

- live Realtime call has not been run with a fresh local key in this shell
- no durable Hermes memory bridge yet
- no transcript/event log feeding a Hermes sidecar yet
- no autonomous idle behavior
- no calibration UI for learned gestures

## Operational Commands

```bash
reachy-on
reachy-off
reachy-hermes-demo
reachy-hermes-action-space
reachy-hermes-voice
```

If Go Task is installed:

```bash
task on
task off
task status
task action-space
task voice
```

