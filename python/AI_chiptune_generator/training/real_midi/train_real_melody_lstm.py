#!/usr/bin/env python3
"""
Standalone training script for a melody-focused LSTM model
using combined_training_sequences.txt as the dataset.

- Expects each line of the data file to be a space-separated
  list of integer tokens (pitches), where -1 represents REST.
- Maps:
    -1  -> 0          (REST)
    x>=0 -> x + 1     (pitches shifted up by 1)
  So the final vocabulary is: 0 = REST, 1..N = pitches.

- Trains a next-token prediction LSTM on sliding windows.
- Saves checkpoint to melody_lstm_real.pt with metadata.
"""

import argparse
import json
import os
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset
# -----------------------------


class RealMelodyDataset(Dataset):
    """
    Dataset for real MIDI-derived melody sequences.

    - Loads lines from a text file (one sequence per line).
    - Applies REST/pitch mapping so all tokens are >= 0.
    - Builds (input, target) sliding windows of length seq_len.
    """

    def __init__(
        self,
        path: str,
        seq_len: int = 64,
        min_seq_len: int = 65,
        max_lines: int = None,
    ):
        """
        Args:
            path: Path to combined_training_sequences.txt
            seq_len: Number of time steps in input window
            min_seq_len: Minimum tokens in a line to use
            max_lines: Optional cap on number of lines to load (for quick tests)
        """
        super().__init__()
        self.path = path
        self.seq_len = seq_len

        # 1) Load raw sequences as lists of ints
        raw_sequences: List[List[int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                tokens = [int(tok) for tok in line.split()]
                if len(tokens) >= min_seq_len:
                    raw_sequences.append(tokens)

        if not raw_sequences:
            raise ValueError(
                f"No usable sequences found in {path}. "
                f"Check that it has at least one line with >= {min_seq_len} tokens."
            )

        # 2) Compute mapping: -1 -> 0 (REST), x>=0 -> x + 1
        #    This guarantees all indices are >= 0 and reserves 0 for REST.
        self.mapping_info: Dict[str, int] = {
            "rest_original": -1,
            "rest_index": 0,
            "pitch_offset": 1,
        }

        mapped_sequences: List[List[int]] = []
        max_token = 0

        for seq in raw_sequences:
            mapped_seq = []
            for t in seq:
                if t == -1:
                    mapped = 0
                else:
                    mapped = t + self.mapping_info["pitch_offset"]
                mapped_seq.append(mapped)
                if mapped > max_token:
                    max_token = mapped
            mapped_sequences.append(mapped_seq)

        self.note_vocab_size = max_token + 1  # vocab indices: [0, ..., max_token]

        # 3) Build sliding windows
        #    For each sequence of length L, create (L - seq_len) samples:
        #      X = seq[i : i+seq_len], y = seq[i+seq_len]
        inputs: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for seq in mapped_sequences:
            L = len(seq)
            if L <= seq_len:
                continue
            for i in range(L - seq_len):
                x = seq[i : i + seq_len]
                y = seq[i + seq_len]
                inputs.append(torch.tensor(x, dtype=torch.long))
                targets.append(torch.tensor(y, dtype=torch.long))

        if not inputs:
            raise ValueError(
                f"After sliding-window construction, no samples were created. "
                f"Try reducing seq_len (currently {seq_len}) or checking your data."
            )

        self.inputs = torch.stack(inputs, dim=0)   # [N, seq_len]
        self.targets = torch.stack(targets, dim=0) # [N]

        print(f"Loaded {len(raw_sequences)} sequences from {path}")
        print(f"Constructed {len(self.inputs)} training samples")
        print(f"Note vocab size: {self.note_vocab_size}")

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# -----------------------------
# Model
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
        emb = self.embedding(x)            # [B, T, E]
        output, _ = self.lstm(emb)         # [B, T, H]
        last_hidden = output[:, -1, :]     # [B, H]
        logits = self.fc_out(last_hidden)  # [B, V]
        return logits


# -----------------------------
# Training Loop
# -----------------------------


def train(
    data_path: str,
    output_path: str = "melody_lstm_real.pt",
    seq_len: int = 64,
    batch_size: int = 64,
    num_epochs: int = 5,
    lr: float = 1e-3,
    max_lines: int = None,
    device: str = None,
):
    """
    Main training routine.

    Args:
        data_path: Path to combined_training_sequences.txt
        output_path: Where to save the trained model checkpoint
        seq_len: Input sequence length (time steps)
        batch_size: Training batch size
        num_epochs: Number of epochs
        lr: Learning rate
        max_lines: Optional limit on number of lines (for debugging/quick runs)
        device: 'cpu', 'cuda', or None (auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    dataset = RealMelodyDataset(
        path=data_path,
        seq_len=seq_len,
        min_seq_len=seq_len + 1,
        max_lines=max_lines,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MelodyLSTMReal(
        note_vocab_size=dataset.note_vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)  # [B, T]
            y = y.to(device)  # [B]

            optimizer.zero_grad()
            logits = model(x)  # [B, V]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")

    print(f"Training complete. Saving model to {output_path}")

    # Save checkpoint with metadata you can use later for integration
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "note_vocab_size": dataset.note_vocab_size,
        "seq_len": seq_len,
        "mapping_info": dataset.mapping_info,
        "config": {
            "embed_dim": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
        },
    }

    torch.save(checkpoint, output_path)
    print("Checkpoint saved.")


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LSTM melody model on combined_training_sequences.txt"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="combined_training_sequences.txt",
        help="Path to combined_training_sequences.txt",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="melody_lstm_real.pt",
        help="Where to save the trained model checkpoint",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Input sequence length (time steps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of lines to load (for quick tests)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cpu', 'cuda', or None (auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        data_path=args.data_path,
        output_path=args.output_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        max_lines=args.max_lines,
        device=args.device,
    )


if __name__ == "__main__":
    main()
