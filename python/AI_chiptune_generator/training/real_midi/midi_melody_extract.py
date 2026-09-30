import sys
import os
from collections import defaultdict

from mido import MidiFile


def pick_melody_track(mid: MidiFile) -> int:
    """
    Heuristic to pick a melody track:
    - Ignore drum channel (channel 9 in 0-based)
    - Choose track with the most note-on events (velocity > 0)
    """
    track_scores = []
    for i, track in enumerate(mid.tracks):
        note_count = 0
        total_pitch = 0
        pitch_events = 0

        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                # Some messages might not have channel; be defensive
                ch = getattr(msg, "channel", None)
                if ch == 9:  # GM drum channel, skip
                    continue
                note_count += 1
                total_pitch += msg.note
                pitch_events += 1

        avg_pitch = total_pitch / pitch_events if pitch_events > 0 else 0
        track_scores.append((note_count, avg_pitch, i))

    # Pick by highest note_count; if tie, pick higher avg pitch
    track_scores.sort(reverse=True)  # sort by (note_count, avg_pitch) descending
    if not track_scores or track_scores[0][0] == 0:
        # fallback to track 0 if everything is empty
        return 0
    return track_scores[0][2]


def build_note_intervals(track, ticks_per_beat: int):
    """
    Build a list of (start_tick, end_tick, pitch) intervals
    for the chosen melody track.
    """
    abs_ticks = 0
    note_starts = defaultdict(list)  # note -> list of start_ticks
    intervals = []

    for msg in track:
        abs_ticks += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            ch = getattr(msg, "channel", None)
            if ch == 9:
                continue  # skip drums
            note_starts[msg.note].append(abs_ticks)
        elif msg.type in ("note_off", "note_on") and (
            getattr(msg, "velocity", 0) == 0
        ):
            ch = getattr(msg, "channel", None)
            if ch == 9:
                continue
            if note_starts[msg.note]:
                start = note_starts[msg.note].pop(0)
                intervals.append((start, abs_ticks, msg.note))

    # Close any hanging notes at the final time
    final_tick = abs_ticks
    for note, starts in note_starts.items():
        for start in starts:
            intervals.append((start, final_tick, note))

    return intervals


def intervals_to_quantized_sequence(intervals, ticks_per_beat: int):
    """
    Convert note intervals into a 16th-note quantized sequence.
    For each time-step we store the *highest* active pitch,
    or -1 for REST.
    """
    if not intervals:
        return []

    ticks_per_step = ticks_per_beat / 4.0  # 16th note
    max_end = max(end for (_, end, _) in intervals)
    num_steps = int(round(max_end / ticks_per_step)) + 1

    seq = [None] * num_steps

    for start, end, pitch in intervals:
        start_step = int(round(start / ticks_per_step))
        end_step = int(round(end / ticks_per_step))
        if end_step <= start_step:
            end_step = start_step + 1  # ensure at least 1 step

        for s in range(start_step, min(end_step, num_steps)):
            if seq[s] is None or pitch > seq[s]:
                seq[s] = pitch

    # Replace None with REST marker
    return [p if p is not None else -1 for p in seq]


def detect_tonic_pc(pitch_sequence):
    """
    Very crude key detection:
    - Build histogram of pitch classes (0–11)
    - Pick the most common as tonic
    """
    counts = [0] * 12
    for p in pitch_sequence:
        if p >= 0:
            counts[p % 12] += 1
    tonic_pc = max(range(12), key=lambda i: counts[i])
    return tonic_pc


def transpose_to_c_major(pitch_sequence):
    """
    Transpose so that the detected tonic maps to C (0 mod 12).
    Returns (transposed_sequence, offset).
    """
    if not any(p >= 0 for p in pitch_sequence):
        return pitch_sequence, 0  # nothing to do

    tonic_pc = detect_tonic_pc(pitch_sequence)
    # We want tonic_pc + offset ≡ 0 (mod 12)
    offset = (0 - tonic_pc) % 12

    transposed = []
    for p in pitch_sequence:
        if p < 0:
            transposed.append(p)
        else:
            new_p = p + offset
            # clamp just to be safe
            new_p = max(0, min(127, new_p))
            transposed.append(new_p)

    return transposed, offset


def main(path):
    mid = MidiFile(path)
    print(f"Loaded MIDI: {path}")
    print(f"Ticks per beat: {mid.ticks_per_beat}")

    melody_idx = pick_melody_track(mid)
    print(f"Selected melody track index: {melody_idx}")
    track = mid.tracks[melody_idx]

    intervals = build_note_intervals(track, mid.ticks_per_beat)
    print(f"Extracted {len(intervals)} note intervals from melody track.")

    seq = intervals_to_quantized_sequence(intervals, mid.ticks_per_beat)
    print(f"Quantized sequence length (16th notes): {len(seq)}")

    transposed_seq, offset = transpose_to_c_major(seq)
    print(f"Transposed to C by offset (semitones): {offset}")

    # Show a small preview
    preview_len = min(64, len(transposed_seq))
    print("First 64 steps (pitch, -1 = REST):")
    print(transposed_seq[:preview_len])

    # Optionally save to a simple text file for inspection
    out_path = path + ".melody_tokens.txt"
    with open(out_path, "w") as f:
        f.write(" ".join(str(x) for x in transposed_seq))
    print(f"Saved token sequence to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python midi_melody_extract.py <midi_or_folder>")
        sys.exit(1)

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        print(f"Processing folder: {input_path}")
        all_sequences = []
        for fname in os.listdir(input_path):
            if fname.lower().endswith(".mid") or fname.lower().endswith(".midi"):
                full_path = os.path.join(input_path, fname)
                print("\n--------------------------------")
                print(f"Processing: {full_path}")
                try:
                    mid = MidiFile(full_path)
                    melody_idx = pick_melody_track(mid)
                    track = mid.tracks[melody_idx]
                    intervals = build_note_intervals(track, mid.ticks_per_beat)
                    seq = intervals_to_quantized_sequence(intervals, mid.ticks_per_beat)
                    transposed, offset = transpose_to_c_major(seq)

                    # store sequence
                    all_sequences.append(transposed)

                    # Save a per-file tokens (optional)
                    out_path = full_path + ".melody_tokens.txt"
                    with open(out_path, "w") as f:
                        f.write(" ".join(str(x) for x in transposed))
                    print(f"Saved: {out_path}")

                except Exception as e:
                    print(f"Error processing {full_path}: {e}")

        # After all files, save a combined training file
        combined_path = os.path.join(input_path, "combined_training_sequences.txt")
        with open(combined_path, "w") as f:
            for seq in all_sequences:
                f.write(" ".join(str(x) for x in seq) + "\n")
        print("\n========================================")
        print(f"Saved combined training file: {combined_path}")

    else:
        # Single file mode
        main(input_path)
