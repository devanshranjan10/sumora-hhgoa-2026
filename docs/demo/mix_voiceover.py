"""Add timed voice-over to the continuous Playwright screen recording.

Requires macOS `say`, ffmpeg, ffprobe, and NumPy. Run after record_demo.cjs.
"""

import json
import os
import subprocess
import wave
from pathlib import Path

import numpy as np


OUT = Path(os.environ.get("SUMORA_VIDEO_OUT", "/tmp/sumora-video"))
RATE = 48_000


def run(*args: str) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main() -> None:
    cues = json.loads((OUT / "cues.json").read_text())
    duration = float(
        run(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(OUT / "continuous.webm"),
        )
    )
    samples = np.zeros(round(duration * RATE), dtype=np.float32)
    previous_end = 0.0

    for cue in cues:
        name = cue["name"]
        aiff = OUT / f"voice-{name}.aiff"
        wav = OUT / f"voice-{name}.wav"
        speed = "138" if name == "results" else "125"
        run("say", "-v", "Aman", "-r", speed, "-o", str(aiff), cue["speech"])
        run(
            "ffmpeg", "-y", "-v", "error", "-i", str(aiff),
            "-ar", str(RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav),
        )
        with wave.open(str(wav), "rb") as audio:
            clip = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
        start = max(cue["atMs"] / 1000 + 0.4, previous_end + 0.2)
        end = start + len(clip) / RATE
        print(f"{name}: {start:.1f}–{end:.1f}s / scene cue {cue['atMs'] / 1000:.1f}s", flush=True)
        begin = round(start * RATE)
        if begin >= len(samples):
            raise RuntimeError(f"Voice cue {name} falls after video")
        count = min(len(clip), len(samples) - begin)
        samples[begin : begin + count] += clip[:count].astype(np.float32) / 32768
        previous_end = end

    peak = float(np.max(np.abs(samples)))
    if peak == 0:
        raise RuntimeError("Voice track is silent")
    samples *= min(0.85 / peak, 1.2)
    with wave.open(str(OUT / "voiceover.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(RATE)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())

    run(
        "ffmpeg", "-y", "-v", "error", "-i", str(OUT / "continuous.webm"),
        "-i", str(OUT / "voiceover.wav"), "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "25", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "112k", "-movflags", "+faststart",
        "-t", str(duration), str(OUT / "sumora-hhgoa-demo.mp4"),
    )
    print(f"Finished {OUT / 'sumora-hhgoa-demo.mp4'} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
