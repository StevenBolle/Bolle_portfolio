#!/usr/bin/env python3
"""
Test script for melody_lstm_real.pt

- Loads the trained real-melody LSTM checkpoint.
- Generates an autoregressive melody sequence.
- Wraps it in a simple multi-track chiptune-like arrangement:
  * Melody track (from model)
  * Chord track (rule-based I-V-vi-IV in C)
  * Bass track (roots of chords)
  * Drum track (simple pattern)

Outputs a multi-track MIDI file for inspection / listening.
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Dict

import torch
from torch import nn
import mido


# -----------------------------
# Model definition (must match training)
# -----------------------------


class MelodyLSTMReal(nn.Module):
    """
    Simple LSTM model for melody next-token prediction.

    Input:  [batch_size, seq_len] int indices
    Output: [batch_size, vocab_size] logits for next token
    """

    def __init__(
            self,
            note_vocab_size: int,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.note_vocab_size = note_vocab_size

        self.embedding = nn.Embedding(note_vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, note_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] integer indices

        Returns:
            logits: [batch_size, vocab_size]
        """
        emb = self.embedding(x)  # [B, T, E]
        output, _ = self.lstm(emb)  # [B, T, H]
        last_hidden = output[:, -1, :]  # [B, H]
        logits = self.fc_out(last_hidden)  # [B, V]
        return logits


# -----------------------------
# Melody generation
# -----------------------------


def load_model_checkpoint(
        model_path: Path,
        device: str = "cpu",
) -> (MelodyLSTMReal, Dict):
    """
    Load the trained real-melody LSTM from checkpoint.

    Returns:
        model: MelodyLSTMReal in eval() mode
        meta: dict containing mapping_info, seq_len, etc.
    """
    ckpt = torch.load(model_path, map_location=device)

    note_vocab_size = ckpt["note_vocab_size"]
    config = ckpt.get("config", {})
    mapping_info = ckpt.get("mapping_info", {})
    seq_len = ckpt.get("seq_len", 64)

    model = MelodyLSTMReal(
        note_vocab_size=note_vocab_size,
        embed_dim=config.get("embed_dim", 64),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    meta = {
        "note_vocab_size": note_vocab_size,
        "mapping_info": mapping_info,
        "seq_len": seq_len,
        "config": config,
    }

    print(f"Loaded model from {model_path}")
    print(f"Vocab size: {note_vocab_size}, seq_len: {seq_len}")
    print(f"Mapping info: {mapping_info}")

    return model, meta


def sample_next_token(
        logits: torch.Tensor,
        temperature: float = 1.0,
) -> int:
    """
    Sample next token from logits with optional temperature scaling.

    Args:
        logits: [vocab_size] tensor
        temperature: >0, 1.0 = no scaling, <1.0 = more peaked, >1.0 = more random

    Returns:
        int token index
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1).item()
    return token


def generate_melody_tokens(
        model: MelodyLSTMReal,
        mapping_info: Dict,
        seq_len: int,
        total_steps: int,
        temperature: float,
        device: str = "cpu",
) -> List[int]:
    """
    Autoregressively generate a sequence of token indices using the model.

    Args:
        model: trained MelodyLSTMReal
        mapping_info: dict with rest_index, pitch_offset, etc.
        seq_len: context length used during training
        total_steps: number of time steps to generate
        temperature: sampling temperature
        device: 'cpu' or 'cuda'

    Returns:
        List of length total_steps of integer token indices.
    """
    rest_index = mapping_info.get("rest_index", 0)
    pitch_offset = mapping_info.get("pitch_offset", 1)

    # Initialize context with REST tokens
    context = [rest_index] * seq_len

    generated: List[int] = []

    with torch.no_grad():
        for step in range(total_steps):
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
            logits = model(x).squeeze(0)  # [vocab_size]
            token = sample_next_token(logits, temperature=temperature)

            generated.append(token)

            # Slide the window: drop oldest, append new token
            context = context[1:] + [token]

    return generated


def tokens_to_pitches(
        tokens: List[int],
        mapping_info: Dict,
) -> List[Optional[int]]:
    """
    Map model token indices back to "original" pitch representation.

    Assumes training used:
        -1 (original rest)  -> 0 (rest_index)
        x>=0 (original)     -> x + pitch_offset

    So:
        if token == rest_index: rest (None)
        else: pitch = token - pitch_offset

    Returns:
        List of pitches (MIDI note numbers) or None for rest.
    """
    rest_index = mapping_info.get("rest_index", 0)
    pitch_offset = mapping_info.get("pitch_offset", 1)

    result: List[Optional[int]] = []
    for t in tokens:
        if t == rest_index:
            result.append(None)  # REST
        else:
            pitch = t - pitch_offset
            result.append(pitch)

    return result


# -----------------------------
# MIDI construction helpers
# -----------------------------


def create_multitrack_midi(
        melody_pitches: List[Optional[int]],
        bars: int,
        tempo_bpm: int,
        output_path: Path,
) -> None:
    """
    Build a 4-track MIDI file:
      - Track 0: global meta (tempo, time signature)
      - Track 1: melody
      - Track 2: chords
      - Track 3: bass
      - Track 4: drums

    Assumes each element of melody_pitches is a 16th note.
    """
    # Basic timing
    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Convert tempo
    microseconds_per_beat = mido.bpm2tempo(tempo_bpm)
    sixteenth_ticks = ticks_per_beat // 4  # 1/16 note

    # ------------- Track 0: Meta -------------
    meta_track = mido.MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(mido.MetaMessage("set_tempo", tempo=microseconds_per_beat, time=0))
    meta_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24,
                                       notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(mido.MetaMessage("key_signature", key="C", time=0))

    # ------------- Track 1: Melody -------------
    melody_track = mido.MidiTrack()
    mid.tracks.append(melody_track)

    # Set instrument for melody (e.g., lead)
    melody_channel = 0
    melody_program = 80  # Lead 1 (square) or similar
    melody_track.append(mido.Message("program_change", program=melody_program, channel=melody_channel, time=0))

    time_since_last = 0
    note_length_ticks = sixteenth_ticks  # each step = 1/16 note

    for pitch in melody_pitches:
        if pitch is None:
            # REST: just advance time
            time_since_last += sixteenth_ticks
        else:
            # Note ON
            melody_track.append(
                mido.Message(
                    "note_on",
                    note=int(pitch),
                    velocity=96,
                    channel=melody_channel,
                    time=time_since_last,
                )
            )
            # Note OFF after note_length_ticks
            melody_track.append(
                mido.Message(
                    "note_off",
                    note=int(pitch),
                    velocity=0,
                    channel=melody_channel,
                    time=note_length_ticks,
                )
            )
            # Reset time_since_last for the next event
            time_since_last = 0

    # ------------- Track 2: Chords -------------
    chord_track = mido.MidiTrack()
    mid.tracks.append(chord_track)

    chord_channel = 1
    chord_program = 48  # Strings or warm pad
    chord_track.append(mido.Message("program_change", program=chord_program, channel=chord_channel, time=0))

    # Simple I-V-vi-IV progression in C major (one chord per bar)
    bar_chord_progression = ["I", "V", "vi", "IV"]
    # Map degrees to chord tones (MIDI notes)
    # Using C4 = 60 as tonic
    chord_map = {
        "I": [60, 64, 67],  # C major
        "V": [67, 71, 74],  # G major
        "vi": [69, 72, 76],  # A minor
        "IV": [65, 69, 72],  # F major
    }

    chord_duration_ticks = sixteenth_ticks * 16  # 1 bar

    # For each bar, place a chord lasting 1 bar
    for bar_idx in range(bars):
        degree = bar_chord_progression[bar_idx % len(bar_chord_progression)]
        chord_notes = chord_map[degree]

        # Note ONs: first note has the gap encoded in previous note_off's time;
        # here we use time=0 for the first chord in track, which is fine.
        first_note = True
        for n in chord_notes:
            chord_track.append(
                mido.Message(
                    "note_on",
                    note=n,
                    velocity=70,
                    channel=chord_channel,
                    time=0 if not first_note else 0,
                )
            )
            first_note = False

        # Note OFFs after chord_duration_ticks; encode duration in the first off
        first_off = True
        for n in chord_notes:
            chord_track.append(
                mido.Message(
                    "note_off",
                    note=n,
                    velocity=0,
                    channel=chord_channel,
                    time=chord_duration_ticks if first_off else 0,
                )
            )
            first_off = False

    # ------------- Track 3: Bass -------------
    bass_track = mido.MidiTrack()
    mid.tracks.append(bass_track)

    bass_channel = 2
    bass_program = 33  # Acoustic bass / synth bass
    bass_track.append(mido.Message("program_change", program=bass_program, channel=bass_channel, time=0))

    # Bass roots approx one octave below chords
    bass_root_map = {
        "I": 48,  # C3
        "V": 55,  # G3
        "vi": 57,  # A3
        "IV": 53,  # F3
    }

    quarter_ticks = ticks_per_beat
    bass_note_length = quarter_ticks  # each bass note = quarter note

    # 4 quarter notes per bar, all root
    for bar_idx in range(bars):
        degree = bar_chord_progression[bar_idx % len(bar_chord_progression)]
        root_pitch = bass_root_map[degree]

        for beat in range(4):
            # ON at each quarter
            bass_track.append(
                mido.Message(
                    "note_on",
                    note=root_pitch,
                    velocity=80,
                    channel=bass_channel,
                    time=0 if (bar_idx == 0 and beat == 0) else 0,
                )
            )
            # OFF after bass_note_length
            bass_track.append(
                mido.Message(
                    "note_off",
                    note=root_pitch,
                    velocity=0,
                    channel=bass_channel,
                    time=bass_note_length,
                )
            )

    # ------------- Track 4: Drums -------------
    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)

    drum_channel = 9  # GM drum channel
    # Basic pattern:
    # - Kick: 36
    # - Snare: 38
    # - Closed hat: 42

    for bar_idx in range(bars):
        # We'll build 16th-note resolution, but pattern is 8ths/quarters
        # We'll do:
        #  Beat 1: kick + hat
        #  Beat 2: snare + hat
        #  Beat 3: kick + hat
        #  Beat 4: snare + hat
        #  Hats on every 8th (twice per beat)

        # In ticks, 8th note = half a beat
        eighth_ticks = ticks_per_beat // 2

        # For each beat:
        for beat in range(4):
            beat_start_time = 0 if (bar_idx == 0 and beat == 0) else 0

            # Kick / Snare on the beat
            if beat in (0, 2):
                # Kick
                drum_track.append(
                    mido.Message(
                        "note_on",
                        note=36,
                        velocity=100,
                        channel=drum_channel,
                        time=beat_start_time,
                    )
                )
                drum_track.append(
                    mido.Message(
                        "note_off",
                        note=36,
                        velocity=0,
                        channel=drum_channel,
                        time=eighth_ticks,
                    )
                )
                hat_time = 0
            else:
                # Snare
                drum_track.append(
                    mido.Message(
                        "note_on",
                        note=38,
                        velocity=100,
                        channel=drum_channel,
                        time=beat_start_time,
                    )
                )
                drum_track.append(
                    mido.Message(
                        "note_off",
                        note=38,
                        velocity=0,
                        channel=drum_channel,
                        time=eighth_ticks,
                    )
                )
                hat_time = 0

            # Hi-hat on the beat (after kick/snare off) and on the off-beat
            # First hat: right after kick/snare off
            drum_track.append(
                mido.Message(
                    "note_on",
                    note=42,
                    velocity=60,
                    channel=drum_channel,
                    time=hat_time,
                )
            )
            drum_track.append(
                mido.Message(
                    "note_off",
                    note=42,
                    velocity=0,
                    channel=drum_channel,
                    time=eighth_ticks // 2,
                )
            )

            # Second hat on off-beat (another 8th later)
            drum_track.append(
                mido.Message(
                    "note_on",
                    note=42,
                    velocity=60,
                    channel=drum_channel,
                    time=eighth_ticks // 2,
                )
            )
            drum_track.append(
                mido.Message(
                    "note_off",
                    note=42,
                    velocity=0,
                    channel=drum_channel,
                    time=eighth_ticks // 2,
                )
            )

    # ------------- Save file -------------
    mid.save(output_path)
    print(f"Saved multi-track MIDI to: {output_path}")


# -----------------------------
# CLI plumbing
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a multi-track MIDI using melody_lstm_real.pt"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="melody_lstm_real.pt",
        help="Path to the trained melody LSTM checkpoint",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="test_real_model_multitrack.mid",
        help="Where to save the generated MIDI",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=16,
        help="Number of bars to generate (melody: 16th-note resolution)",
    )
    parser.add_argument(
        "--tempo-bpm",
        type=int,
        default=120,
        help="Tempo in BPM",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (>0). Lower = more deterministic, higher = more random.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu', 'cuda', or None for auto-detect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        torch.manual_seed(args.seed)

    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    model, meta = load_model_checkpoint(model_path, device=device)
    mapping_info = meta["mapping_info"]
    seq_len = meta["seq_len"]

    # Total melody steps: bars * 16 (16th notes per bar)
    total_steps = args.bars * 16
    print(f"Generating melody: {total_steps} steps ({args.bars} bars @ 16th-note resolution)")

    tokens = generate_melody_tokens(
        model=model,
        mapping_info=mapping_info,
        seq_len=seq_len,
        total_steps=total_steps,
        temperature=args.temperature,
        device=device,
    )

    melody_pitches = tokens_to_pitches(tokens, mapping_info=mapping_info)

    # Build multi-track MIDI around the melody
    create_multitrack_midi(
        melody_pitches=melody_pitches,
        bars=args.bars,
        tempo_bpm=args.tempo_bpm,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
