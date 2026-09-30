#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import re
import time
from torch.utils.data import TensorDataset, DataLoader

########################################
# CONFIG
########################################

note_min = 48      # C3
note_max = 72      # C5
rest_token = 0     # special rest ID

# Major scales in a single octave within [note_min, note_max]
MAJOR_SCALES = {
    # C major: C D E F G A B
    "C":  [48, 50, 52, 53, 55, 57, 59],
    # F major: F G A Bb C D E
    "F":  [53, 55, 57, 58, 60, 62, 64],
    # G major: G A B C D E F#
    "G":  [55, 57, 59, 60, 62, 64, 66],
    # D major: D E F# G A B C#
    "D":  [50, 52, 54, 55, 57, 59, 61],
}

# Natural minor scales in a single octave within [note_min, note_max]
MINOR_SCALES = {
    # A natural minor: A B C D E F G
    "Am": [57, 59, 60, 62, 64, 65, 67],
    # D natural minor: D E F G A Bb C
    "Dm": [50, 52, 53, 55, 57, 58, 60],
    # E natural minor: E F# G A B C D
    "Em": [52, 54, 55, 57, 59, 60, 62],
    # G natural minor: G A Bb C D Eb F
    "Gm": [55, 57, 58, 60, 62, 63, 65],
}

def get_scale_and_mode(key_name: str):
    """
    Return (scale_pitches, mode_str) for a given key_name.
    mode_str is 'major' or 'minor'.
    Raises ValueError if key_name is unknown.
    """
    if key_name in MAJOR_SCALES:
        return MAJOR_SCALES[key_name], "major"
    if key_name in MINOR_SCALES:
        return MINOR_SCALES[key_name], "minor"
    raise ValueError("Unknown key_name " + str(key_name))


PROGRESSIONS = {
    # Original pop loop
    "POP":   ["I", "V", "vi", "IV"],
    # Variant pop loop: vi - IV - I - V
    "POP2":  ["vi", "IV", "I", "V"],
    # Simple I-IV-V-IV loop
    "LOOP":  ["I", "IV", "V", "IV"],
    # Simple loop variant I-V-I-IV loop
    "LOOP2":  ["I", "V", "I", "IV"],
}

ROMAN_FUNCTIONS = {
    "I":   {"degrees": [1, 3, 5]},
    "V":   {"degrees": [5, 7, 2]},
    "vi":  {"degrees": [6, 1, 3]},
    "IV":  {"degrees": [4, 6, 1]},
}

STYLE_LABELS = [
    "overworld",
    "battle",
    "dungeon",
    "town",
]

CONSOLE_LABELS = [
    "nes",
    "snes",
    "genesis",
    "gameboy",
]

########################################
# SYNTHETIC DATA GENERATION
########################################

def generate_sequence_for_key_and_progression(
    key_name,
    progression_name,
    style_name=None,
    console_name=None,
    num_bars=4,
    steps_per_bar=16,
    rng=None,
):
    """
    Generate a monophonic note sequence for a given key and chord progression,
    with behavior strongly influenced by style_name and console_name.

    All notes remain in-scale and chord-correct; style/console only affect:
    - rest probability (sparsity / density)
    - repetition tendency
    - pitch bias within the chord tones (low vs high)
    """
    if progression_name not in PROGRESSIONS:
        raise ValueError("Unknown progression_name " + str(progression_name))

    # Use provided RNG for variability across sequences; fall back to fixed seed if None
    if rng is None:
        rng = np.random.RandomState(0)

    # Get scale (major or minor) for this key
    scale, mode = get_scale_and_mode(key_name)
    roman_prog = PROGRESSIONS[progression_name]

    # Local mapping of roman numerals to chord indices (not used globally, but kept for completeness)
    roman_to_idx_local = {}
    roman_keys_sorted = sorted(ROMAN_FUNCTIONS.keys())
    for i, r in enumerate(roman_keys_sorted):
        roman_to_idx_local[r] = i

    note_seq = []
    chord_idx_seq = []

    # -----------------------------
    # Base behavioral parameters
    # -----------------------------
    rest_prob = 0.15
    repeat_prob = 0.25
    pitch_bias = 0.0

    # -----------------------------
    # Style-based adjustments
    # -----------------------------
    if style_name == "overworld":
        rest_prob *= 0.8
        repeat_prob *= 0.8
        pitch_bias += 0.25
    elif style_name == "battle":
        rest_prob *= 0.3
        repeat_prob *= 0.6
        pitch_bias += 0.35
    elif style_name == "dungeon":
        rest_prob *= 2.2
        repeat_prob *= 1.8
        pitch_bias += 0.1
    elif style_name == "town":
        rest_prob *= 1.1
        repeat_prob *= 1.3
        pitch_bias += 0.05

    # -----------------------------
    # Console-based adjustments
    # -----------------------------
    if console_name == "nes":
        rest_prob *= 0.8
        repeat_prob *= 0.9
        pitch_bias += 0.05
    elif console_name == "snes":
        rest_prob *= 1.0
        repeat_prob *= 1.1
        pitch_bias += 0.0
    elif console_name == "genesis":
        rest_prob *= 0.85
        repeat_prob *= 0.8
        pitch_bias += 0.15
    elif console_name == "gameboy":
        rest_prob *= 1.3
        repeat_prob *= 1.4
        pitch_bias -= 0.05

    rest_prob = float(min(max(rest_prob, 0.02), 0.70))
    repeat_prob = float(min(max(repeat_prob, 0.05), 0.90))
    pitch_bias = float(min(max(pitch_bias, -0.45), 0.45))

    prev_note = None

    # For dungeon style, we want longer held notes and less “busy” motion.
    dungeon_mode = (style_name == "dungeon")
    sustain_remaining = 0  # how many more steps to keep holding the current note

    for bar in range(num_bars):
        roman = roman_prog[bar % len(roman_prog)]
        chord_info = ROMAN_FUNCTIONS[roman]
        chord_degrees = chord_info["degrees"]

        # Compute chord tones from scale degrees (in key), clamped
        chord_tones = []
        for deg in chord_degrees:
            pitch = scale[(deg - 1) % len(scale)]
            if pitch < note_min:
                pitch = note_min
            if pitch > note_max:
                pitch = note_max
            chord_tones.append(pitch)

        chord_tones_sorted = sorted(chord_tones)
        chord_idx_local = roman_to_idx_local[roman]

        for _ in range(steps_per_bar):
            # -----------------------------
            # DUNGEON MODE: sustained notes
            # -----------------------------
            if dungeon_mode and sustain_remaining > 0 and prev_note is not None and prev_note != rest_token:
                # Keep holding the previous note for longer, no new decision this step.
                note = prev_note
                sustain_remaining -= 1
            else:
                # Decide rest vs note
                if rng.rand() < rest_prob:
                    note = rest_token
                    sustain_remaining = 0  # break any existing sustain
                else:
                    # Decide whether to repeat the previous note
                    if (
                            prev_note is not None
                            and prev_note != rest_token
                            and rng.rand() < repeat_prob
                    ):
                        note = prev_note
                        # If we are in dungeon mode, we can optionally still
                        # let this repetition count as sustaining.
                        if dungeon_mode and sustain_remaining > 0:
                            sustain_remaining -= 1
                    else:
                        # Choose a chord tone with bias toward low/mid/high
                        u = rng.rand()
                        u = min(max(u + pitch_bias, 0.0), 0.999)

                        if u < 1.0 / 3.0:
                            note = chord_tones_sorted[0]
                        elif u < 2.0 / 3.0 and len(chord_tones_sorted) > 1:
                            note = chord_tones_sorted[1]
                        else:
                            note = chord_tones_sorted[-1]

                        # In dungeon mode, when we pick a *new* note, decide
                        # how long to sustain it over future steps.
                        if dungeon_mode and note != rest_token:
                            # Hold for 1–4 extra steps (tweak range to taste)
                            sustain_remaining = rng.randint(1, 5)
                        else:
                            sustain_remaining = 0

            note_seq.append(int(note))
            chord_idx_seq.append(chord_idx_local)
            prev_note = note

    return note_seq, chord_idx_seq, roman_to_idx_local

def build_synthetic_dataset(
    num_sequences=800,
    num_bars=4,
    steps_per_bar=16,
    context_len=16,
):
    """
    Build a synthetic dataset of (note, chord, style, console) sequences
    using the style/console-aware sequence generator.
    """
    rng = np.random.RandomState(123)

    x_notes_list = []
    x_chords_list = []
    x_styles_list = []
    x_consoles_list = []
    y_notes_list = []

    # Global vocabularies
    chord_to_idx = {}
    style_to_idx = {s: i for i, s in enumerate(STYLE_LABELS)}
    console_to_idx = {c: i for i, c in enumerate(CONSOLE_LABELS)}

    # Precompute list of all available keys and progressions
    all_major_keys = list(MAJOR_SCALES.keys())
    all_minor_keys = list(MINOR_SCALES.keys())
    all_keys = all_major_keys + all_minor_keys
    all_progressions = list(PROGRESSIONS.keys())

    for _ in range(num_sequences):
        # Randomly choose key and progression for this sequence
        key_name = rng.choice(all_keys)
        progression_name = rng.choice(all_progressions)

        # Choose style and console for this sequence
        style_name = rng.choice(STYLE_LABELS)
        console_name = rng.choice(CONSOLE_LABELS)

        # Generate a full-sequence note line with style/console behavior
        note_seq, chord_idx_seq_local, local_chord_to_idx = generate_sequence_for_key_and_progression(
            key_name=key_name,
            progression_name=progression_name,
            style_name=style_name,
            console_name=console_name,
            num_bars=num_bars,
            steps_per_bar=steps_per_bar,
            rng=rng,
        )

        # Ensure global chord_to_idx has entries for all romans in this progression
        for roman_name in PROGRESSIONS[progression_name]:
            if roman_name not in chord_to_idx:
                chord_to_idx[roman_name] = len(chord_to_idx)

        # Build a global chord index sequence (one chord per bar, repeated over steps_per_bar)
        global_chord_idx_seq = []
        for bar_i in range(num_bars):
            roman = PROGRESSIONS[progression_name][bar_i % len(PROGRESSIONS[progression_name])]
            global_idx = chord_to_idx[roman]
            for _ in range(steps_per_bar):
                global_chord_idx_seq.append(global_idx)

        assert len(global_chord_idx_seq) == len(note_seq)

        style_idx = style_to_idx[style_name]
        console_idx = console_to_idx[console_name]

        # Sliding window to create (context_len -> next note) training samples
        for i in range(len(note_seq) - context_len):
            ctx_notes = note_seq[i:i + context_len]
            tgt_note = note_seq[i + context_len]

            ctx_chords = global_chord_idx_seq[i:i + context_len]

            x_notes_list.append(ctx_notes)
            x_chords_list.append(ctx_chords)
            x_styles_list.append([style_idx] * context_len)
            x_consoles_list.append([console_idx] * context_len)
            y_notes_list.append(tgt_note)

    x_notes = np.array(x_notes_list, dtype=np.int64)
    x_chords = np.array(x_chords_list, dtype=np.int64)
    x_styles = np.array(x_styles_list, dtype=np.int64)
    x_consoles = np.array(x_consoles_list, dtype=np.int64)
    y_notes = np.array(y_notes_list, dtype=np.int64)

    return (
        x_notes,
        x_chords,
        x_styles,
        x_consoles,
        y_notes,
        chord_to_idx,
        style_to_idx,
        console_to_idx,
    )

########################################
# MODEL
########################################

class MelodyLSTM(nn.Module):
    def __init__(
        self,
        note_vocab_size,
        chord_vocab_size,
        style_vocab_size,
        console_vocab_size,
        note_emb_dim=32,
        chord_emb_dim=32,
        style_emb_dim=8,
        console_emb_dim=8,
        hidden_size=128,
        num_layers=2,
    ):
        super().__init__()

        self.note_emb = nn.Embedding(note_vocab_size, note_emb_dim)
        self.chord_emb = nn.Embedding(chord_vocab_size, chord_emb_dim)
        self.style_emb = nn.Embedding(style_vocab_size, style_emb_dim)
        self.console_emb = nn.Embedding(console_vocab_size, console_emb_dim)

        self.input_proj = nn.Linear(
            note_emb_dim + chord_emb_dim + style_emb_dim + console_emb_dim,
            hidden_size,
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_size, note_vocab_size)

    def forward(self, notes, chords, styles, consoles, hidden=None):
        note_e = self.note_emb(notes)
        chord_e = self.chord_emb(chords)
        style_e = self.style_emb(styles)
        console_e = self.console_emb(consoles)

        x = torch.cat([note_e, chord_e, style_e, console_e], dim=-1)
        x = self.input_proj(x)

        out, hidden_out = self.lstm(x, hidden)

        logits = self.output_proj(out)
        return logits, hidden_out

########################################
# TRAINING UTILITIES
########################################

def train_model(
    x_notes,
    x_chords,
    x_styles,
    x_consoles,
    y_notes,
    num_epochs=5,
    batch_size=64,
    lr=1e-3,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    note_vocab_size = int(max(x_notes.max(), note_max) + 2)
    chord_vocab_size = int(x_chords.max() + 1)
    style_vocab_size = int(x_styles.max() + 1)
    console_vocab_size = int(x_consoles.max() + 1)

    model = MelodyLSTM(
        note_vocab_size=note_vocab_size,
        chord_vocab_size=chord_vocab_size,
        style_vocab_size=style_vocab_size,
        console_vocab_size=console_vocab_size,
    ).to(device)

    ds = TensorDataset(
        torch.from_numpy(x_notes),
        torch.from_numpy(x_chords),
        torch.from_numpy(x_styles),
        torch.from_numpy(x_consoles),
        torch.from_numpy(y_notes),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dl:
            b_notes, b_chords, b_styles, b_consoles, b_targets = [
                t.to(device) for t in batch
            ]

            optimizer.zero_grad()
            logits, _ = model(b_notes, b_chords, b_styles, b_consoles)

            logits_last = logits[:, -1, :]

            loss = criterion(logits_last, b_targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * b_notes.size(0)

        avg_loss = total_loss / len(ds)
        print("Epoch " + str(epoch + 1) + "/" + str(num_epochs) + " - loss: " + str(round(avg_loss, 4)))

    return model

########################################
# SAVE / LOAD MODEL + VOCABS
########################################

def save_model(model, path, chord_to_idx, style_to_idx, console_to_idx):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "chord_to_idx": chord_to_idx,
        "style_to_idx": style_to_idx,
        "console_to_idx": console_to_idx,
    }
    torch.save(checkpoint, path)
    print("Saved model checkpoint to" , path)


def load_model(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"]
    chord_to_idx = checkpoint["chord_to_idx"]
    style_to_idx = checkpoint["style_to_idx"]
    console_to_idx = checkpoint["console_to_idx"]

    note_vocab_size = state["note_emb.weight"].shape[0]
    chord_vocab_size = state["chord_emb.weight"].shape[0]
    style_vocab_size = state["style_emb.weight"].shape[0]
    console_vocab_size = state["console_emb.weight"].shape[0]

    model = MelodyLSTM(
        note_vocab_size=note_vocab_size,
        chord_vocab_size=chord_vocab_size,
        style_vocab_size=style_vocab_size,
        console_vocab_size=console_vocab_size,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    print("Loaded model from" , path)
    return model, chord_to_idx, style_to_idx, console_to_idx

########################################
# GENERATION
########################################

def sample_from_logits(logits, temperature=1.0, deterministic=False):
    if deterministic:
        return int(torch.argmax(logits).item())

    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    probs = probs.reshape(-1)
    probs = probs / probs.sum()
    idx = int(np.random.choice(len(probs), p=probs))
    return idx


def generate_melody(
    model,
    chord_to_idx,
    style_to_idx,
    console_to_idx,
    key_name="C",
    progression_name="POP",
    style_name="overworld",
    console_name="nes",
    num_bars=4,
    steps_per_bar=16,
    context_len=16,
    device=None,
    deterministic=False,
    temperature=1.0,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate key and progression; support both major and minor keys
    scale, mode = get_scale_and_mode(key_name)
    if progression_name not in PROGRESSIONS:
        raise ValueError("Unknown progression_name " + str(progression_name))

    roman_prog = PROGRESSIONS[progression_name]

    chord_idx_seq = []
    for bar in range(num_bars):
        roman = roman_prog[bar % len(roman_prog)]
        chord_idx = chord_to_idx[roman]
        for _ in range(steps_per_bar):
            chord_idx_seq.append(chord_idx)

    total_steps = num_bars * steps_per_bar

    start_pitch = scale[0]
    start_note = max(min(start_pitch, note_max), note_min)

    notes = [start_note] * context_len

    style_idx = style_to_idx[style_name]
    console_idx = console_to_idx[console_name]

    style_seq = [style_idx] * total_steps
    console_seq = [console_idx] * total_steps

    model.eval()

    with torch.no_grad():
        for t in range(context_len, total_steps):
            ctx_notes = notes[t - context_len:t]
            ctx_chords = chord_idx_seq[t - context_len:t]
            ctx_styles = style_seq[t - context_len:t]
            ctx_consoles = console_seq[t - context_len:t]

            x_notes = torch.tensor([ctx_notes], dtype=torch.long, device=device)
            x_chords = torch.tensor([ctx_chords], dtype=torch.long, device=device)
            x_styles = torch.tensor([ctx_styles], dtype=torch.long, device=device)
            x_consoles = torch.tensor([ctx_consoles], dtype=torch.long, device=device)

            logits, _ = model(x_notes, x_chords, x_styles, x_consoles)

            logits_last = logits[0, -1, :]

            next_idx = sample_from_logits(
                logits_last,
                temperature=temperature,
                deterministic=deterministic,
            )

            notes.append(next_idx)

    return notes, {
        "style_name": style_name,
        "console_name": console_name,
        "key_name": key_name,
        "progression_name": progression_name,
    }

########################################
# MIDI EXPORT
########################################

try:
    import mido
except ImportError:
    mido = None

# General MIDI program numbers are 0-based here (0..127).
# Comments show the standard 1-based GM numbers for reference.

STYLE_INSTRUMENTS = {
    "overworld": {
        "melody": 80,   # 81: Lead 1 (square)
        "harmony": 89,  # 90: Pad 2 (warm)
        "bass": 33,     # 34: Electric Bass (finger)
    },
    "battle": {
        "melody": 81,   # 82: Lead 2 (sawtooth)
        "harmony": 62,  # 63: Synth Brass 1
        "bass": 38,     # 39: Synth Bass 1
    },
    "dungeon": {
        "melody": 50,   # 51: Synth Strings
        "harmony": 91,  # 92: Pad 4 (choir)
        "bass": 35,     # 36: Fretless Bass
    },
    "town": {
        "melody": 74,   # 75: Pan Flute
        "harmony": 12,  # 13: Marimba
        "bass": 32,     # 33: Acoustic Bass
    },
}

# Very rough GM approximations of each console’s sound chip.
# These override the STYLE_INSTRUMENTS defaults to give each console
# a more authentic “voice”.
CONSOLE_INSTRUMENT_OVERRIDES = {
    "nes": {
        # NES: 2 pulse, 1 triangle, 1 noise. Use square-ish leads + simple bass.
        "melody": 80,   # Lead 1 (square)
        "harmony": 80,  # Another square-ish voice
        "bass": 33,     # Electric Bass (finger) as triangle stand-in
    },
    "snes": {
        # SNES: sample-based, airy / reverby.
        "melody": 50,   # Synth Strings 1
        "harmony": 91,  # Pad 4 (choir)
        "bass": 34,     # Electric Bass (finger)
    },
    "genesis": {
        # Genesis: FM-heavy, bright / metallic.
        "melody": 82,   # Lead 3 (calliope) as FM-ish
        "harmony": 91,  # Pad 4 (choir) or similar
        "bass": 38,     # Synth Bass 1
    },
    "gameboy": {
        # Game Boy: 2 pulse + 1 wave + 1 noise. Very simple, dry.
        "melody": 80,   # Square lead
        "harmony": 80,  # Same square, lower velocity in mix
        "bass": 32,     # Acoustic Bass as simple tone
    },
}


def get_instruments(style_name: str, console_name: str | None = None):
    """
    Return a (melody_prog, harmony_prog, bass_prog) tuple of GM program numbers.

    We start from STYLE_INSTRUMENTS for the requested style, then (optionally)
    apply console-specific overrides to approximate NES/SNES/Genesis/GameBoy.
    """
    # Base style palette
    base = STYLE_INSTRUMENTS.get("overworld").copy()
    base.update(STYLE_INSTRUMENTS.get(style_name, {}))

    # Console overrides (if any)
    if console_name is not None:
        overrides = CONSOLE_INSTRUMENT_OVERRIDES.get(console_name)
        if overrides:
            base.update(overrides)

    return base["melody"], base["harmony"], base["bass"]


def notes_to_midi(notes, output_path="output.mid", tempo_bpm=120):
    if mido is None:
        print("mido is not installed; cannot write MIDI.")
        return

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    ticks_per_beat = mid.ticks_per_beat
    step_ticks = ticks_per_beat // 4

    current_note = None

    for n in notes:
        if n == rest_token:
            if current_note is not None:
                track.append(mido.Message("note_off", note=current_note, velocity=64, time=step_ticks))
                current_note = None
            else:
                track.append(mido.Message("note_off", note=0, velocity=0, time=step_ticks))
        else:
            if current_note is not None and current_note != n:
                track.append(mido.Message("note_off", note=current_note, velocity=64, time=0))
                track.append(mido.Message("note_on", note=n, velocity=80, time=0))
                current_note = n
                track.append(mido.Message("note_off", note=current_note, velocity=64, time=step_ticks))
                current_note = None
            elif current_note is None:
                track.append(mido.Message("note_on", note=n, velocity=80, time=0))
                current_note = n
                track.append(mido.Message("note_off", note=current_note, velocity=64, time=step_ticks))
                current_note = None
            else:
                track.append(mido.Message("note_off", note=current_note, velocity=64, time=step_ticks))
                current_note = None

    if current_note is not None:
        track.append(mido.Message("note_off", note=current_note, velocity=64, time=0))

    mid.save(output_path)
    print("Wrote MIDI to" , output_path)

def build_harmony_and_bass(meta, num_bars: int, steps_per_bar: int = 16):
    """
    Build simple rule-based harmony and bass lines aligned to the same
    bar/step grid as the melody, using the global key/progression info.
    Returns:
        harmony_notes: list[int] of length num_bars * steps_per_bar
        bass_notes:    list[int] of length num_bars * steps_per_bar
    """
    key_name = meta.get("key_name", "C")
    progression_name = meta.get("progression_name", "POP")

    scale, mode = get_scale_and_mode(key_name)
    if progression_name not in PROGRESSIONS:
        raise ValueError("Unknown progression_name " + str(progression_name))

    roman_prog = PROGRESSIONS[progression_name]

    total_steps = num_bars * steps_per_bar

    harmony_notes = [rest_token] * total_steps
    bass_notes = [rest_token] * total_steps

    for bar in range(num_bars):
        roman = roman_prog[bar % len(roman_prog)]
        chord_info = ROMAN_FUNCTIONS[roman]
        chord_degrees = chord_info["degrees"]

        # Build chord tones in key
        chord_pitches = []
        for deg in chord_degrees:
            p = scale[(deg - 1) % len(scale)]
            chord_pitches.append(p)

        # Root / third / fifth (fallbacks if chord has fewer than 3 degrees)
        root_pitch = chord_pitches[0]
        third_pitch = chord_pitches[1] if len(chord_pitches) > 1 else root_pitch
        fifth_pitch = chord_pitches[2] if len(chord_pitches) > 2 else root_pitch

        # Clamp harmony to the melody range
        def clamp_to_melody_range(p):
            p = int(p)
            if p < note_min:
                while p + 12 <= note_max:
                    p += 12
            if p > note_max:
                while p - 12 >= note_min:
                    p -= 12
            return p

        root_h = clamp_to_melody_range(root_pitch)
        third_h = clamp_to_melody_range(third_pitch)
        fifth_h = clamp_to_melody_range(fifth_pitch)

        # Bass: push root down an octave or two, keep in a reasonable low range
        bass_pitch = int(root_pitch) - 24
        if bass_pitch < 28:  # avoid going too low
            bass_pitch = root_pitch - 12
        if bass_pitch < 24:
            bass_pitch = root_pitch
        if bass_pitch > 60:
            bass_pitch = 60

        # Fill this bar
        bar_start = bar * steps_per_bar
        bar_end = bar_start + steps_per_bar

        for step in range(steps_per_bar):
            global_idx = bar_start + step
            beat = step // 4       # 0..3 for 16 steps/bar
            inner = step % 4       # 0..3 within each beat

            # Harmony pattern: quarter-note arpeggio root -> third -> fifth -> third
            if inner == 0:
                if beat == 0:
                    harmony_notes[global_idx] = root_h
                elif beat == 1:
                    harmony_notes[global_idx] = third_h
                elif beat == 2:
                    harmony_notes[global_idx] = fifth_h
                else:
                    harmony_notes[global_idx] = third_h
            else:
                harmony_notes[global_idx] = rest_token

            # Bass pattern: root on beats 1 and 3 (step 0 and 8)
            if inner == 0 and beat in (0, 2):
                bass_notes[global_idx] = bass_pitch
            else:
                bass_notes[global_idx] = bass_notes[global_idx]

    return harmony_notes, bass_notes

def build_drum_track(style_name: str, num_bars: int, steps_per_bar: int = 16):
    """
    Build a simple, deterministic drum pattern for the given style.
    Returns a list of lists: drum_notes[step] = [drum_midi_notes...].
    Uses GM standard drum mapping on channel 9.

    Patterns (which steps are hit) are the same across versions;
    only the drum instruments (notes) differ per style.
    """
    # GM drum notes
    BD1 = 36          # Bass Drum 1
    ABD = 35          # Acoustic Bass Drum
    SNARE = 38        # Acoustic Snare
    HAND_CLAP = 39    # Hand Clap
    LOW_FLOOR_TOM = 41
    LOW_TOM = 45
    CLOSED_HH = 42    # Closed Hi-Hat
    LOW_CONGA = 64
    MARACAS = 70
    TAMBOURINE = 54

    total_steps = num_bars * steps_per_bar
    drum_notes = [[] for _ in range(total_steps)]

    for bar in range(num_bars):
        bar_start = bar * steps_per_bar

        # Default patterns (kick/snare/hat steps)
        if style_name == "overworld":
            # Pop-ish: K: 1 & 3, S: 2 & 4, hats on 8ths
            kick_steps = [0, 8]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]

            kick_note = BD1
            snare_note = SNARE
            hat_notes = [CLOSED_HH]

        elif style_name == "battle":
            # Busier pattern: more kicks, hats on every 16th
            kick_steps = [0, 6, 8, 14]
            snare_steps = [4, 12]
            hat_steps = list(range(0, steps_per_bar))  # every step

            kick_note = BD1
            snare_note = SNARE
            hat_notes = [CLOSED_HH]

        elif style_name == "dungeon":
            # Sparse, minimal: low thumps, no bright cymbals
            kick_steps = [0, 8]
            snare_steps = [12]        # low floor tom as "snare"
            hat_steps = [0, 8]        # low conga pulses

            kick_note = ABD                 # deeper kick
            snare_note = LOW_FLOOR_TOM      # tom instead of snare
            hat_notes = [LOW_CONGA]         # low conga instead of hat

        elif style_name == "town":
            # Simple, relaxed: kick + hand clap + hand percussion
            kick_steps = [0, 8]
            snare_steps = [4, 12]    # hand clap on 2 & 4
            hat_steps = [0, 4, 8, 12]

            kick_note = BD1
            snare_note = HAND_CLAP
            # Alternate maracas & tambourine on hat steps
            hat_notes = [MARACAS, TAMBOURINE]

        else:
            # Fallback: simple pop groove with standard kit
            kick_steps = [0, 8]
            snare_steps = [4, 12]
            hat_steps = [0, 2, 4, 6, 8, 10, 12, 14]

            kick_note = BD1
            snare_note = SNARE
            hat_notes = [CLOSED_HH]

        # Apply to global step indices
        for s in kick_steps:
            idx = bar_start + s
            if 0 <= idx < total_steps:
                drum_notes[idx].append(kick_note)

        for s in snare_steps:
            idx = bar_start + s
            if 0 <= idx < total_steps:
                drum_notes[idx].append(snare_note)

        # Hat/hand-percussion: may have 1 or 2 alternating sounds
        for i, s in enumerate(hat_steps):
            idx = bar_start + s
            if 0 <= idx < total_steps:
                if len(hat_notes) == 1:
                    drum_notes[idx].append(hat_notes[0])
                else:
                    # Alternate between the two hat/perc sounds
                    drum_notes[idx].append(hat_notes[i % len(hat_notes)])

    return drum_notes

def emit_sustained_track(
    track,
    notes,
    channel: int,
    step_ticks: int,
    program: int = None,
    tempo: int = None,
    set_tempo_here: bool = False,
):
    """
    Emit a MIDI track where consecutive identical notes are merged
    into sustained notes, instead of one note per step.

    - notes: list[int] using rest_token for silence
    - channel: MIDI channel (0-15)
    - step_ticks: ticks per 1 step (we use 16 steps per bar)
    - program: optional GM program number (0-based). If not None, send program_change.
    - tempo: optional microseconds-per-beat. If set_tempo_here, send set_tempo meta here.
    """
    import mido  # in case we’re inside another module

    # Optional tempo on this track
    if set_tempo_here and tempo is not None:
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Optional instrument program
    if program is not None:
        track.append(
            mido.Message(
                "program_change",
                program=int(program),
                channel=channel,
                time=0,
            )
        )

    if not notes:
        return

    # Compress notes into (value, length_steps) segments
    segments = []
    prev = notes[0]
    length = 1
    for n in notes[1:]:
        if n == prev:
            length += 1
        else:
            segments.append((prev, length))
            prev = n
            length = 1
    segments.append((prev, length))

    pending_time = 0

    for value, seg_len in segments:
        duration_ticks = seg_len * step_ticks

        if value == rest_token:
            # Just accumulate rest time
            pending_time += duration_ticks
        else:
            # Emit a sustained note of seg_len steps
            track.append(
                mido.Message(
                    "note_on",
                    note=int(value),
                    velocity=80,
                    time=pending_time,
                    channel=channel,
                )
            )
            track.append(
                mido.Message(
                    "note_off",
                    note=int(value),
                    velocity=64,
                    time=duration_ticks,
                    channel=channel,
                )
            )
            pending_time = 0

    # Any remaining pending_time is just trailing silence; no need to emit anything

def write_multitrack_midi(
    melody_notes,
    harmony_notes,
    bass_notes,
    style_name: str,
    tempo_bpm: int,
    output_path: str,
    drum_notes=None,
    console_name: str | None = None,
):
    """
    Write a multi-track MIDI file:
      - Track 0: melody (LSTM output, sustained)
      - Track 1: harmony (rule-based, sustained)
      - Track 2: bass    (rule-based, sustained)
      - Track 3: drums   (style-based patterns on channel 9, step-based)

    Instruments for melody/harmony/bass are chosen based on both style_name
    and console_name, so selecting NES vs SNES vs Genesis vs Game Boy
    actually changes the timbre and keeps us roughly inside each chip's
    "voice" family.
    """
    if mido is None:
        print("mido is not installed; cannot write multi-track MIDI.")
        return

    # Ensure track lengths match (clip to shortest if not)
    lengths = [len(melody_notes), len(harmony_notes), len(bass_notes)]
    if drum_notes is not None:
        lengths.append(len(drum_notes))

    min_len = min(lengths)
    if len(melody_notes) != min_len:
        melody_notes = melody_notes[:min_len]
    if len(harmony_notes) != min_len:
        harmony_notes = harmony_notes[:min_len]
    if len(bass_notes) != min_len:
        bass_notes = bass_notes[:min_len]
    if drum_notes is not None and len(drum_notes) != min_len:
        drum_notes = drum_notes[:min_len]

    mid = mido.MidiFile()
    ticks_per_beat = mid.ticks_per_beat
    step_ticks = ticks_per_beat // 4  # 16 steps per 4/4 bar

    tempo = mido.bpm2tempo(tempo_bpm)

    # Console-aware instrument selection
    mel_prog, harm_prog, bass_prog = get_instruments(
        style_name=style_name,
        console_name=console_name,
    )

    # Simple per-track velocity “mix”: make melody a bit more forward,
    # harmony a bit softer, bass in between.
    if style_name == "dungeon":
        melody_vel = 104
        harmony_vel = 72
        bass_vel = 84
    else:
        melody_vel = 96
        harmony_vel = 80
        bass_vel = 88

    # Melody, harmony, bass tracks (sustained)
    for i, (notes, channel, program, velocity) in enumerate(
        [
            (melody_notes, 0, mel_prog, melody_vel),
            (harmony_notes, 1, harm_prog, harmony_vel),
            (bass_notes, 2, bass_prog, bass_vel),
        ]
    ):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Set tempo once on the first track
        emit_sustained_track(
            track=track,
            notes=notes,
            channel=channel,
            step_ticks=step_ticks,
            program=program,
            tempo=tempo,
            set_tempo_here=(i == 0),
        )

        # Apply a simple track name meta to record style/console
        track_name = f"{style_name}_{console_name or 'default'}_track{channel}"
        track.append(
            mido.MetaMessage("track_name", name=track_name, time=0)
        )

    # Drum track on channel 9 (GM standard)
    if drum_notes is not None:
        drum_track = mido.MidiTrack()
        mid.tracks.append(drum_track)

        channel = 9  # channel 10 in 1-based terms

        for step_hits in drum_notes:
            # step_hits is a list of drum note numbers at this step
            if not step_hits:
                # No drums here; just advance time
                drum_track.append(
                    mido.Message(
                        "note_off",
                        note=0,
                        velocity=0,
                        time=step_ticks,
                        channel=channel,
                    )
                )
            else:
                first_on = True
                for dn in step_hits:
                    drum_track.append(
                        mido.Message(
                            "note_on",
                            note=int(dn),
                            velocity=90,
                            time=0 if not first_on else 0,
                            channel=channel,
                        )
                    )
                    first_on = False

                # Turn them all off at the end of the step
                first_off = True
                for dn in step_hits:
                    drum_track.append(
                        mido.Message(
                            "note_off",
                            note=int(dn),
                            velocity=64,
                            time=step_ticks if first_off else 0,
                            channel=channel,
                        )
                    )
                    first_off = False

    mid.save(output_path)
    print("Wrote multi-track MIDI to", output_path)

########################################
# HIGH-LEVEL GENERATION WRAPPER FOR WEB/CLI
########################################

def generate_track(
    model,
    chord_to_idx,
    style_to_idx,
    console_to_idx,
    track_title: str,
    style_name: str,
    console_name: str,
    speaker_mode: str,
    num_bars: int,
    tempo_bpm: int,
    temperature: float,
    key_name: str = "C",
    progression_name: str = "POP",
    device=None,
):
    """
    High-level wrapper that:
      - Generates melody via LSTM
      - Builds harmony, bass, drum tracks
      - Writes multi-track MIDI
      - Saves composer log
      - Returns paths + log text for UI/Flask

    Returns dict:
    {
        "midi_path": "...",
        "log_path": "...",
        "log_text": "...",
        "meta": {...}
    }
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure safe filename
    safe_title = "".join(ch for ch in track_title if ch.isalnum() or ch in "_-").rstrip()
    if not safe_title:
        safe_title = "untitled_track"

    midi_filename = f"{safe_title}.mid"

    # Build full path inside midi_data/
    midi_path = get_midi_output_path(midi_filename)

    # ---- 1. Melody generation ----
    t_model_start = time.perf_counter()
    notes, meta = generate_melody(
        model,
        chord_to_idx,
        style_to_idx,
        console_to_idx,
        key_name=key_name,
        progression_name=progression_name,
        style_name=style_name,
        console_name=console_name,
        num_bars=num_bars,
        steps_per_bar=16,
        context_len=16,
        device=device,
        deterministic=False,
        temperature=temperature,
    )
    t_model_end = time.perf_counter()

    # Store timing in meta so analyze_melody() can log it
    meta["melody_generation_time_sec"] = t_model_end - t_model_start

    # Include speaker mode in meta (not used in analysis, but good for logging)
    meta["speaker_mode"] = speaker_mode

    # ---- 2. Harmony + Bass ----
    harmony_notes, bass_notes = build_harmony_and_bass(
        meta,
        num_bars=num_bars,
        steps_per_bar=16,
    )

    # ---- 3. Drums ----
    drum_notes = build_drum_track(
        style_name=style_name,
        num_bars=num_bars,
        steps_per_bar=16,
    )

    # ---- 4. Write MIDI ----
    write_multitrack_midi(
        melody_notes=notes,
        harmony_notes=harmony_notes,
        bass_notes=bass_notes,
        style_name=style_name,
        tempo_bpm=tempo_bpm,
        output_path=midi_path,
        drum_notes=drum_notes,
        console_name=console_name,
    )

    # ---- 5. Composer log ----
    log_text, metrics = analyze_melody(notes, meta, steps_per_bar=16)

    log_path = midi_path.replace(".mid", "_composer_log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_text)
    except Exception as e:
        print(f"Warning: Could not write composer log: {e}")

    return {
        "midi_path": midi_path,
        "log_path": log_path,
        "log_text": log_text,
        "meta": meta,
    }

########################################
# INTERACTIVE LOOP HELPERS
########################################

def prompt_choice(prompt, options):
    while True:
        print(prompt)
        for i, opt in enumerate(options):
            print("  " + str(i + 1) + ") " + opt)
        choice = input("Enter choice: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("Invalid choice, try again.")

def sanitize_filename(name: str) -> str:
    """
    Turn an arbitrary track title into a safe filename.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name or "track"

def get_audio_output_path(filename: str) -> str:
    """
    Return an absolute path to save an audio file in the local 'audio_data' folder,
    creating the folder if it does not exist.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    audio_dir = os.path.join(project_root, "audio_data")

    os.makedirs(audio_dir, exist_ok=True)

    return os.path.join(audio_dir, filename)

def get_midi_output_path(filename: str) -> str:
    """
    Return an absolute path to save a MIDI file in the local 'midi_data' folder,
    creating the folder if it does not exist.
    """
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(app_dir)
    midi_dir = os.path.join(project_root, "midi_data")

    os.makedirs(midi_dir, exist_ok=True)

    return os.path.join(midi_dir, filename)

########################################
# COMPOSER LOG & EVALUATION
########################################

def analyze_melody(notes, meta, steps_per_bar=16):
    """
    Compute basic music-theory-based metrics (scale adherence, chord fit, rests, etc.)
    and return a human-readable composer log string plus a metrics dict.
    """
    key_name = meta.get("key_name", "C")
    progression_name = meta.get("progression_name", "POP")
    style_name = meta.get("style_name", "unknown")
    console_name = meta.get("console_name", "unknown")

    try:
        scale, mode = get_scale_and_mode(key_name)
    except ValueError:
        scale, mode = [], "unknown"

    roman_prog = PROGRESSIONS.get(progression_name, [])

    total_steps = len(notes)
    num_bars = total_steps // steps_per_bar if steps_per_bar > 0 else 0

    # Basic counts
    num_rest = sum(1 for n in notes if n == rest_token)
    non_rest_notes = [n for n in notes if n != rest_token]

    # Scale adherence: % of non-rest notes that lie in the scale
    if non_rest_notes and scale:
        scale_hits = sum(1 for n in non_rest_notes if n in scale)
        scale_adherence = 100.0 * scale_hits / len(non_rest_notes)
    else:
        scale_hits = 0
        scale_adherence = 0.0

    # Chord fit: % of non-rest notes that match the chord tones for the active chord
    chord_hits = 0
    chord_checked = 0
    if roman_prog and scale:
        for t, n in enumerate(notes):
            if n == rest_token:
                continue

            bar_idx = t // steps_per_bar
            roman = roman_prog[bar_idx % len(roman_prog)]
            chord_info = ROMAN_FUNCTIONS.get(roman)
            if chord_info is None:
                continue

            chord_degrees = chord_info["degrees"]
            chord_tones = []
            for deg in chord_degrees:
                pitch = scale[(deg - 1) % len(scale)]
                if pitch < note_min:
                    pitch = note_min
                if pitch > note_max:
                    pitch = note_max
                chord_tones.append(pitch)

            chord_checked += 1
            if n in chord_tones:
                chord_hits += 1

        if chord_checked > 0:
            chord_fit = 100.0 * chord_hits / chord_checked
        else:
            chord_fit = 0.0
    else:
        chord_fit = 0.0
        chord_checked = 0

    # Rest ratio
    rest_ratio = 100.0 * num_rest / total_steps if total_steps > 0 else 0.0

    # Simple bar-by-bar summary: roman function, average pitch, density
    bar_summaries = []
    for b in range(num_bars):
        start = b * steps_per_bar
        end = start + steps_per_bar
        bar_notes = [n for n in notes[start:end] if n != rest_token]

        if bar_notes:
            avg_pitch = sum(bar_notes) / len(bar_notes)
            density = 100.0 * len(bar_notes) / steps_per_bar
        else:
            avg_pitch = None
            density = 0.0

        roman = roman_prog[b % len(roman_prog)] if roman_prog else "?"
        bar_summaries.append((b + 1, roman, avg_pitch, density))

    # Build human-readable log
    lines = []
    lines.append("=== Composer Log ===")
    lines.append(f"Style: {style_name} | Console: {console_name}")
    if roman_prog:
        prog_str = "-".join(roman_prog)
    else:
        prog_str = "N/A"
    lines.append(f"Key: {key_name} {mode} | Progression: {progression_name} ({prog_str})")
    lines.append("")
    lines.append(f"Total steps: {total_steps}")
    lines.append(f"Bars (assuming {steps_per_bar} steps per bar): {num_bars}")
    lines.append(f"Rest ratio: {rest_ratio:.1f}% ({num_rest} of {total_steps} steps)")

    if non_rest_notes and scale:
        lines.append(
            f"Scale adherence: {scale_adherence:.1f}% "
            f"({scale_hits} of {len(non_rest_notes)} non-rest notes in scale)"
        )
    else:
        lines.append("Scale adherence: N/A (no non-rest notes or no scale info)")

    if chord_checked > 0:
        lines.append(
            f"Chord fit: {chord_fit:.1f}% "
            f"({chord_hits} of {chord_checked} chord-checked notes)"
        )
    else:
        lines.append("Chord fit: N/A (no chord-checked notes)")

    lines.append("")
    lines.append("Bar-by-bar summary:")
    if bar_summaries:
        for bar_index, roman, avg_pitch, density in bar_summaries:
            pitch_str = f"{avg_pitch:.1f}" if avg_pitch is not None else "N/A"
            lines.append(
                f"  Bar {bar_index}: Roman {roman}, "
                f"avg pitch: {pitch_str}, note density: {density:.1f}%"
            )
    else:
        lines.append("  (No bar summary available)")

    log_text = "\n".join(lines)

    # ---- Performance / timing info (if available) ----
    melody_time = meta.get("melody_generation_time_sec")
    total_time = meta.get("total_generation_time_sec")

    if melody_time is not None or total_time is not None:
        lines.append("")
        lines.append("==== Performance ====")
        if melody_time is not None:
            lines.append(f"Model melody generation time: {melody_time:.3f} seconds")
        if total_time is not None:
            lines.append(f"End-to-end track generation time: {total_time:.3f} seconds")

    metrics = {
        "total_steps": total_steps,
        "num_bars": num_bars,
        "num_rest": num_rest,
        "rest_ratio": rest_ratio,
        "scale_adherence": scale_adherence,
        "scale_hits": scale_hits,
        "non_rest_count": len(non_rest_notes),
        "chord_fit": chord_fit,
        "chord_hits": chord_hits,
        "chord_checked": chord_checked,
        # New: performance metrics (may be None if not recorded)
        "melody_generation_time_sec": float(melody_time) if melody_time is not None else None,
    }

    return log_text, metrics

def console_style_loop(model, chord_to_idx, style_to_idx, console_to_idx, device):
    print("Entering interactive generation loop. Type q at any prompt to quit.")

    while True:
        style_names = list(style_to_idx.keys())
        console_names = list(console_to_idx.keys())

        style = prompt_choice("Select a soundscape / style:", style_names)
        console = prompt_choice("Select a console:", console_names)

        # Speaker mode: logical flag so we know if the user wants CRT/mono.
        speaker_mode = prompt_choice(
            "Select output speaker mode:",
            ["stereo", "mono_crt"],
        )

        num_bars_input = input("Number of bars (default 4): ").strip()
        if num_bars_input.lower() == "q":
            break
        num_bars = 4
        if num_bars_input.isdigit():
            num_bars = max(1, min(16, int(num_bars_input)))

        temp_input = input("Sampling temperature (e.g. 1.0, 1.2; default 1.1): ").strip()
        if temp_input.lower() == "q":
            break
        temperature = 1.1
        try:
            if temp_input:
                temperature = float(temp_input)
        except ValueError:
            print("Invalid temperature, using default 1.1")

        # Default BPM based on style
        if style == "overworld":
            default_bpm = 120
        elif style == "battle":
            default_bpm = 140
        elif style == "dungeon":
            default_bpm = 90
        elif style == "town":
            default_bpm = 110
        else:
            default_bpm = 120

        bpm_input = input(f"Tempo in BPM (default {default_bpm}): ").strip()
        if bpm_input.lower() == "q":
            break

        try:
            if bpm_input:
                tempo_bpm = max(40, min(220, int(bpm_input)))
            else:
                tempo_bpm = default_bpm
        except ValueError:
            print(f"Invalid BPM, using default {default_bpm}")
            tempo_bpm = default_bpm

        print("Generating... this should be quick.")

        notes, meta = generate_melody(
            model,
            chord_to_idx,
            style_to_idx,
            console_to_idx,
            key_name="C",
            progression_name="POP",
            style_name=style,
            console_name=console,
            num_bars=num_bars,
            steps_per_bar=16,
            context_len=16,
            device=device,
            deterministic=False,
            temperature=temperature,
        )

        # Attach speaker mode to meta so the composer log knows what we targeted.
        meta["speaker_mode"] = speaker_mode

        safe_style = style.replace(" ", "_")
        safe_console = console.replace(" ", "_")
        safe_speaker = speaker_mode.replace(" ", "_")

        out_name = f"melody_{safe_style}_{safe_console}_{safe_speaker}_{num_bars}bars.mid"

        # Build full absolute path inside midi_data/
        output_path = get_midi_output_path(out_name)

        harmony_notes, bass_notes = build_harmony_and_bass(
            meta,
            num_bars=num_bars,
            steps_per_bar=16,
        )

        # Build drum track based on style
        drum_notes = build_drum_track(
            style_name=style,
            num_bars=num_bars,
            steps_per_bar=16,
        )

        # Write multi-track MIDI (melody + harmony + bass + drums),
        # console-aware instruments.
        write_multitrack_midi(
            melody_notes=notes,
            harmony_notes=harmony_notes,
            bass_notes=bass_notes,
            style_name=style,
            tempo_bpm=tempo_bpm,
            output_path=output_path,
            drum_notes=drum_notes,
            console_name=console,
        )

        print(f"Saved multi-track MIDI to: {output_path}")
        print("Done. Wrote", out_name)

        # --- Composer log & evaluation ---
        log_text, metrics = analyze_melody(
            notes,
            meta,
            steps_per_bar=16,
        )

        # Print composer log to console
        print("\n" + log_text + "\n")

        # Save composer log alongside the MIDI
        log_path = os.path.splitext(output_path)[0] + "_composer_log.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(log_text)
            print(f"Saved composer log to: {log_path}")
        except Exception as e:
            print("Warning: could not save composer log:", e)

        again = input("Generate another? (y/n): ").strip().lower()
        if again != "y":
            break

########################################
# MAIN
########################################

def main():
    # Set up device once, reuse for all modes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    while True:
        print("\nSelect mode:")
        print("  1) Train model on synthetic data")
        print("  2) Load existing model and enter console/style generation loop")
        print("  3) Train on synthetic data, then enter generation loop")
        print("  4) Train model on real MIDI data (experimental - placeholder)")
        print("  5) Placeholder option 5 (future feature)")
        print("  6) Placeholder option 6 (future feature)")
        print("  7) Placeholder option 7 (future feature)")
        print("  8) Placeholder option 8 (future feature)")
        print("  9) Quit")
        mode = input("Enter a number (1-9): ").strip()

        if mode == "1" or mode == "3":
            print("Building synthetic dataset...")

            x_notes, x_chords, x_styles, x_consoles, y_notes, chord_to_idx, style_to_idx, console_to_idx = (
                build_synthetic_dataset(
                    num_sequences=200,
                    num_bars=4,
                    steps_per_bar=16,
                    context_len=16,
                )
            )

            print(
                f"Synthetic dataset built: X_notes shape={x_notes.shape}, y_notes shape={y_notes.shape}"
            )

            model = train_model(
                x_notes=x_notes,
                x_chords=x_chords,
                x_styles=x_styles,
                x_consoles=x_consoles,
                y_notes=y_notes,
                device=device,
                num_epochs=5,
                batch_size=32,
                lr=1e-3,
            )

            # Save the trained model checkpoint so it can be loaded in mode 2 later
            save_model(
                model,
                path="melody_lstm.pt",
                chord_to_idx=chord_to_idx,
                style_to_idx=style_to_idx,
                console_to_idx=console_to_idx,
            )

            print("Training complete.")

            if mode == "1":
                # Just training; return to menu instead of exiting program
                print("Returning to main menu.")
                continue

            print("Entering interactive console/style loop...")
            console_style_loop(model, chord_to_idx, style_to_idx, console_to_idx, device)
            print("Exiting interactive loop. Returning to main menu.")

        elif mode == "2":
            print("Loading trained model...")

            model, chord_to_idx, style_to_idx, console_to_idx = load_model(
                path="melody_lstm.pt",
                device=device,
            )

            console_style_loop(model, chord_to_idx, style_to_idx, console_to_idx, device)
            print("Exiting interactive loop. Returning to main menu.")

        elif mode == "4":
            print("Mode 4 (real MIDI) is currently a placeholder.")
            print("You can wire this up later with your own real MIDI loader.")
            print("Returning to main menu.")

        elif mode == "5":
            print("Option 5 is a placeholder for a future feature.")
            print("Returning to main menu.")

        elif mode == "6":
            print("Option 6 is a placeholder for a future feature.")
            print("Returning to main menu.")

        elif mode == "7":
            print("Option 7 is a placeholder for a future feature.")
            print("Returning to main menu.")

        elif mode == "8":
            print("Option 8 is a placeholder for a future feature.")
            print("Returning to main menu.")

        elif mode == "9":
            print("Exiting program. Goodbye!")
            break

        else:
            print("Invalid mode. Please enter a number between 1 and 9.")


