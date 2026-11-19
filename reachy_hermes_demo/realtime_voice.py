"""Realtime voice demo for Reachy Mini using OpenAI Realtime over WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from typing import Any

import websockets

from .agent_tools import ToolDispatcher, sanitize_for_model, tool_specs
from .robot_tools import build_toolbox


MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "marin")
SAMPLE_RATE = 24000

DEFAULT_SOURCE = "alsa_input.usb-Pollen_Robotics_Reachy_Mini_Audio_100025004261401204-00.analog-stereo"
DEFAULT_SINK = "alsa_output.usb-Pollen_Robotics_Reachy_Mini_Audio_100025004261401204-00.analog-stereo"


INSTRUCTIONS = """You are Hermes running inside Reachy Mini.

You are a fast, embodied voice agent. Keep speech short and responsive. Use natural
preambles only when useful: say what you are doing in a few words, then call the
tool. Never claim you moved, danced, saw, or learned something unless the tool
result says it worked.

Embodiment policy:
- Use small gestures often: attentive, curious, nod_yes, shake_no, confused, happy.
- Use dance only when the user asks for something showy.
- Use take_photo when the user asks what you can see, asks about the room, points to
  an object, or wants visual grounding.
- You may create learned gestures with learn_gesture. Keep them safe and short:
  1-6 steps, durations under 1s, no extreme body_yaw. Then play the gesture.
- If background audio or side conversation is not addressed to you, stay quiet.

You are not a dashboard. You are the conversational body and tool-using agent loop
for Reachy Mini.
"""


def _session_update(model: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    realtime_tools = [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
        }
        for tool in tools
    ]
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": model,
            "output_modalities": ["audio"],
            "instructions": INSTRUCTIONS,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 450,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": VOICE,
                },
            },
            "tools": realtime_tools,
            "tool_choice": "auto",
            "reasoning": {"effort": "low"},
        },
    }


def _open_recorder(source: str | None) -> subprocess.Popen[bytes]:
    cmd = [
        "parec",
        "--raw",
        f"--rate={SAMPLE_RATE}",
        "--format=s16le",
        "--channels=1",
        "--latency-msec=40",
    ]
    if source:
        cmd.append(f"--device={source}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def _open_player(sink: str | None) -> subprocess.Popen[bytes]:
    if shutil.which("pw-play"):
        cmd = [
            "pw-play",
            "--raw",
            f"--rate={SAMPLE_RATE}",
            "--format=s16",
            "--channels=1",
            "--latency=40ms",
        ]
        if sink:
            cmd.extend(["--target", sink])
        cmd.append("-")
    else:
        cmd = [
            "paplay",
            "--raw",
            f"--rate={SAMPLE_RATE}",
            "--format=s16le",
            "--channels=1",
            "--latency-msec=40",
        ]
        if sink:
            cmd.append(f"--device={sink}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


async def _send_audio(ws: Any, recorder: subprocess.Popen[bytes], stop: asyncio.Event) -> None:
    assert recorder.stdout is not None
    chunk_bytes = 4800  # 100ms of 24kHz int16 mono
    while not stop.is_set():
        chunk = await asyncio.to_thread(recorder.stdout.read, chunk_bytes)
        if not chunk:
            await asyncio.sleep(0.01)
            continue
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )
        )


async def _handle_tool(
    ws: Any,
    dispatcher: ToolDispatcher,
    tool_name: str,
    call_id: str,
    args_json: str,
) -> None:
    try:
        args = json.loads(args_json or "{}")
    except json.JSONDecodeError as exc:
        result = {"error": f"invalid JSON arguments: {exc}"}
    else:
        result = await asyncio.to_thread(dispatcher.call, tool_name, args)

    model_result = sanitize_for_model(tool_name, result if isinstance(result, dict) else {"result": result})
    await ws.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(model_result),
                },
            }
        )
    )

    if tool_name == "take_photo" and isinstance(result, dict) and result.get("image_b64"):
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{result['image_b64']}",
                            }
                        ],
                    },
                }
            )
        )
    await ws.send(json.dumps({"type": "response.create"}))


async def _receive_events(
    ws: Any,
    dispatcher: ToolDispatcher,
    player: subprocess.Popen[bytes],
    stop: asyncio.Event,
) -> None:
    assert player.stdin is not None
    async for raw in ws:
        event = json.loads(raw)
        event_type = event.get("type", "")

        if event_type in {"conversation.item.input_audio_transcription.completed"}:
            transcript = (event.get("transcript") or "").strip()
            if transcript:
                print(f"user: {transcript}", flush=True)

        if event_type in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
            transcript = (event.get("transcript") or "").strip()
            if transcript:
                print(f"hermes: {transcript}", flush=True)

        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            delta = event.get("delta")
            if isinstance(delta, str):
                audio = base64.b64decode(delta)
                try:
                    player.stdin.write(audio)
                    player.stdin.flush()
                except BrokenPipeError:
                    pass

        if event_type == "response.function_call_arguments.done":
            tool_name = event.get("name")
            call_id = event.get("call_id")
            arguments = event.get("arguments") or "{}"
            if isinstance(tool_name, str) and isinstance(call_id, str):
                print(f"tool: {tool_name}({arguments})", flush=True)
                asyncio.create_task(_handle_tool(ws, dispatcher, tool_name, call_id, arguments))

        if event_type == "error":
            print(f"realtime error: {json.dumps(event.get('error', event), sort_keys=True)}", flush=True)

        if stop.is_set():
            return


async def run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your shell or source a local env file.")

    toolbox = build_toolbox()
    dispatcher = ToolDispatcher(toolbox)
    tools = tool_specs(toolbox)
    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("OpenAI-Safety-Identifier", "reachy-mini-local-demo"),
    ]
    url = f"wss://api.openai.com/v1/realtime?model={args.model}"

    recorder = _open_recorder(args.source)
    player = _open_player(args.sink)
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    with suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, stop.set)
        loop.add_signal_handler(signal.SIGTERM, stop.set)

    try:
        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            await ws.send(json.dumps(_session_update(args.model, tools)))
            print("Connected. Speak to Reachy. Press Ctrl-C to stop.", flush=True)
            audio_task = asyncio.create_task(_send_audio(ws, recorder, stop))
            receive_task = asyncio.create_task(_receive_events(ws, dispatcher, player, stop))
            await stop.wait()
            audio_task.cancel()
            receive_task.cancel()
            with suppress(asyncio.CancelledError):
                await audio_task
            with suppress(asyncio.CancelledError):
                await receive_task
    finally:
        for proc in (recorder, player):
            with suppress(Exception):
                proc.terminate()
            with suppress(Exception):
                proc.wait(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime voice Hermes demo for Reachy Mini.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--source", default=os.environ.get("REACHY_AUDIO_SOURCE", DEFAULT_SOURCE))
    parser.add_argument("--sink", default=os.environ.get("REACHY_AUDIO_SINK", DEFAULT_SINK))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
