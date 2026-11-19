# System Design

## Problem

A good embodied Reachy agent needs two different qualities:

- Fluidity: speech should respond quickly, interrupt naturally, and avoid long dead air.
- Agency: the system should still have durable state, tool calls, camera use, learned gestures, and eventually Hermes-style planning.

A single slow agent loop should not sit directly in the audio path.

## Proposed Runtime

```text
Reachy mic
  -> realtime voice session
      -> immediate speech/audio responses
      -> function calls for robot/camera/memory tools
  -> tool dispatcher
      -> safe semantic tools
      -> motion/camera workers
      -> Reachy daemon + SDK
  -> optional mature Hermes loop
      -> slower planning, memory, multi-step tasks
      -> emits the same tool schema
```

The realtime model is the fast front-of-house voice. Hermes is the slower deliberative back office. Both target the same tool bus.

## Why This Split

OpenAI's voice-agent docs describe two voice architectures:

- speech-to-speech live sessions for natural, low-latency interaction
- chained voice pipelines for stronger explicit control over STT, agent workflow, and TTS

For Reachy, the right first design is a hybrid:

- Use speech-to-speech Realtime for the live conversational loop.
- Put robot tools and business logic server-side.
- Let a deeper Hermes agent take over only for tasks that need planning or longer memory.

OpenAI also documents server-side/sideband controls for Realtime sessions. That maps well to Reachy: the robot tool executor should live locally next to the daemon, not in the browser or inside the model.

## Demo Architecture

This repo implements:

```text
realtime_voice.py
  - captures Reachy USB mic via parec
  - streams PCM to OpenAI Realtime over WebSocket
  - plays returned PCM through Reachy speaker via pw-play/paplay
  - receives function_call events

agent_tools.py
  - declares model-visible tool schemas
  - dispatches tool calls
  - strips bulky image base64 before echoing tool output

robot_tools.py
  - talks to Reachy daemon REST and Python SDK
  - validates semantic motion keyframes
  - stores learned gestures as JSON
  - captures camera frames and encodes JPEG with ffmpeg
```

Camera calls are special. The tool captures a JPEG and returns metadata; the realtime handler then attaches the image as an `input_image` conversation item so the model can answer visually.

## Latency Policy

The realtime voice model should:

- answer directly for simple conversational turns
- speak a short preamble before visible tool use
- call background robot tools without waiting to narrate every motor step
- use `reasoning.effort=low` by default

The deeper Hermes loop should:

- handle long plans and memory consolidation
- propose learned gestures
- decide when to use camera or repeated observation
- never own low-level motor commands

## Tool Contract

Tools are semantic and bounded:

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

This is intentionally close to Pollen's conversation app tool surface but with `learn_gesture` added as the dynamic-expression path.

## What Is Still Missing

This demo is enough to try the design, but not a full Hermes integration yet:

- no durable Hermes memory adapter
- no remote Hermes process bridge
- no UI
- no VAD diagnostics panel
- no autonomous idle behavior beyond the current safe gestures
- no calibration UI for custom learned moves

The next step is to make the mature Hermes loop a sidecar that can subscribe to transcripts/tool results and emit the same tool calls as the realtime model.

## Sources

- OpenAI voice agents: https://developers.openai.com/api/docs/guides/voice-agents
- OpenAI Realtime overview: https://developers.openai.com/api/docs/guides/realtime
- OpenAI Realtime WebSocket: https://developers.openai.com/api/docs/guides/realtime-websocket
- OpenAI Realtime WebRTC: https://developers.openai.com/api/docs/guides/realtime-webrtc
- OpenAI server-side controls: https://developers.openai.com/api/docs/guides/realtime-server-controls
- OpenAI realtime prompting: https://developers.openai.com/api/docs/guides/realtime-models-prompting
- Pollen conversation app: https://github.com/pollen-robotics/reachy_mini_conversation_app

