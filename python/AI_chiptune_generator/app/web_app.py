#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)

import torch
import chiptune_core as core

# -------------------------
# CONFIG
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Folder where chiptune_core writes MIDI files
MIDI_DIR = PROJECT_ROOT / "midi_data"

# Folder where we will (optionally) put rendered audio
AUDIO_DIR = PROJECT_ROOT / "audio_data"
AUDIO_DIR.mkdir(exist_ok=True)

# If you want automatic MIDI->audio rendering, set this to a valid
# SoundFont path and make sure `fluidsynth` and `ffmpeg` are installed
# and on PATH.
SOUNDFONT_PATH = PROJECT_ROOT / "The_Nes_Soundfont.sf2"

FFMPEG_PATH = "ffmpeg"  # or r"C:\Path\to\ffmpeg.exe" if needed
# -------------------------
# APP + MODEL INIT
# -------------------------

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Flask app using device:", device)

# Load LSTM model + vocabs once at startup
MODEL, CHORD_TO_IDX, STYLE_TO_IDX, CONSOLE_TO_IDX = core.load_model(
    path=str(PROJECT_ROOT / "models" /"melody_lstm.pt"),
    device=device,
)


# -------------------------
# MIDI -> AUDIO (optional)
# -------------------------

def render_midi_to_audio(midi_path: Path):
    """
    Given a MIDI file path, try to render a WAV (and optional MP3) into AUDIO_DIR.
    Returns (wav_path, mp3_path) or (None, None) on failure.
    """
    if not SOUNDFONT_PATH.exists():
        print("SoundFont missing at", SOUNDFONT_PATH)
        return None, None

    wav_name = midi_path.stem + ".wav"
    wav_path = AUDIO_DIR / wav_name

    # 1) MIDI -> WAV via FluidSynth
    # NOTE: options FIRST, then soundfont, then MIDI file.
    cmd_fs = [
        "fluidsynth",
        "-ni",
        "-F", str(wav_path),   # write audio to this WAV
        "-r", "44100",         # sample rate
        "-g", "2.0",
        str(SOUNDFONT_PATH),   # soundfont
        str(midi_path),        # midi file
    ]

    try:
        print("Rendering MIDI to WAV with:", " ".join(cmd_fs))
        # don't rely only on return code; also check that the WAV was created
        subprocess.run(cmd_fs, check=False)
        if not wav_path.exists():
            print("Warning: FluidSynth did not create WAV at", wav_path)
            return None, None
    except Exception as e:
        print("Warning: fluidsynth rendering failed:", e)
        return None, None

    # 2) WAV -> MP3 via ffmpeg (optional, but nicer for browsers / file size)
    mp3_name = midi_path.stem + ".mp3"
    mp3_path = AUDIO_DIR / mp3_name

    try:
        cmd_ff = [
            FFMPEG_PATH,
            "-y",
            "-i", str(wav_path),
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            str(mp3_path),
        ]
        print("Converting WAV to MP3 with:", " ".join(cmd_ff))
        subprocess.run(cmd_ff, check=True)
    except Exception as e:
        print("Warning: ffmpeg MP3 conversion failed:", e)
        # fall back to just WAV
        mp3_path = None

    return wav_path, mp3_path


# -------------------------
# ROUTES
# -------------------------

@app.route("/")
def index():
    # Just render the HTML page; all logic is frontend + /generate
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """
    Endpoint called via JS fetch. Expects JSON:
    {
        "title": "...",
        "style": "overworld",
        "console": "nes",
        "speaker_mode": "stereo" | "mono_crt",
        "bars": 8,
        "tempo_bpm": 120,
        "temperature": 1.1,
        "key_name": "C",
        "progression_name": "POP"
    }
    Returns JSON with:
    - status: "ok" or "error"
    - log_text (if ok)
    - midi_url (if ok)
    - audio_url (if ok and audio render succeeded, else null)
    - title (if ok)
    - message (if error)
    """
    import traceback  # debug helper

    try:
        data = request.get_json(force=True) or {}

        title = (data.get("title") or "").strip()
        style_name = data.get("style") or "overworld"
        console_name = data.get("console") or "nes"
        speaker_mode = data.get("speaker_mode") or "stereo"
        key_name = (data.get("key_name") or "C").strip()
        progression_name = (data.get("progression_name") or "POP").strip()

        try:
            num_bars = int(data.get("bars") or 8)
        except ValueError:
            num_bars = 8
        num_bars = max(1, min(16, num_bars))

        try:
            tempo_bpm = int(data.get("tempo_bpm") or 120)
        except ValueError:
            tempo_bpm = 120
        tempo_bpm = max(40, min(220, tempo_bpm))

        try:
            temperature = float(data.get("temperature") or 1.1)
        except ValueError:
            temperature = 1.1
        temperature = max(0.1, min(3.0, temperature))

        if not title:
            title = f"{style_name}_{console_name}_{num_bars}bars"

        # --- Call the high-level core wrapper ---
        result = core.generate_track(
            model=MODEL,
            chord_to_idx=CHORD_TO_IDX,
            style_to_idx=STYLE_TO_IDX,
            console_to_idx=CONSOLE_TO_IDX,
            track_title=title,
            style_name=style_name,
            console_name=console_name,
            speaker_mode=speaker_mode,
            num_bars=num_bars,
            tempo_bpm=tempo_bpm,
            temperature=temperature,
            key_name=key_name,
            progression_name=progression_name,
            device=device,
        )

        midi_path = Path(result["midi_path"])
        log_text = result["log_text"]

        # Best-effort audio rendering; may return WAV and/or MP3
        wav_path, mp3_path = render_midi_to_audio(midi_path)

        midi_url = f"/midi/{midi_path.name}"
        if mp3_path is not None:
            audio_url = f"/audio/{mp3_path.name}"
        elif wav_path is not None:
            audio_url = f"/audio/{wav_path.name}"
        else:
            audio_url = None

        return jsonify(
            {
                "status": "ok",
                "title": title,
                "log_text": log_text,
                "midi_url": midi_url,
                "audio_url": audio_url,
            }
        )

    except Exception as e:
        # Print the full traceback to the Flask console so we can diagnose
        print("ERROR in /generate:", e)
        traceback.print_exc()
        return jsonify(
            {
                "status": "error",
                "message": str(e),
            }
        ), 500


@app.route("/midi/<path:filename>")
def serve_midi(filename):
    """
    Serve MIDI files from the midi_data folder so the user can download them.
    """
    return send_from_directory(MIDI_DIR, filename, as_attachment=False)


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """
    Serve rendered audio (WAV) files from audio_data.
    """
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)


# -------------------------
# ENTRYPOINT
# -------------------------

if __name__ == "__main__":
    # Run Flask in debug mode on localhost
    app.run(host="127.0.0.1", port=5000, debug=True)