"""
Offline VQAD training script.

Trains the four-qubit VQC on synthetic benign/Byzantine gradient data
and saves the learned weights to models/vqad_weights.npy.

Usage:
    python experiments/train_vqad.py [--epochs 100] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vqad import VQADetector


def generate_training_data(n_each: int, model_size: int, seed: int):
    rng = np.random.default_rng(seed)
    global_model = rng.normal(0, 0.1, model_size)
    benign = [rng.normal(0, 0.05, model_size) for _ in range(n_each)]
    byzantine_mm = [rng.normal(0, 0.05, model_size) * 10.0 for _ in range(n_each // 3)]
    byzantine_lf = [-rng.normal(0, 0.05, model_size) for _ in range(n_each // 3)]
    byzantine_sf = [-rng.normal(0, 0.05, model_size) for _ in range(n_each // 3)]
    byzantine = byzantine_mm + byzantine_lf + byzantine_sf
    return benign, byzantine, global_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=150)
    parser.add_argument("--model_size", type=int, default=1000)
    parser.add_argument("--output", default="models/vqad_weights.npy")
    args = parser.parse_args()

    print(f"Generating training data ({args.n_samples} benign + ~{args.n_samples} Byzantine)...")
    benign, byzantine, global_model = generate_training_data(
        args.n_samples, args.model_size, args.seed
    )

    detector = VQADetector(seed=args.seed)
    print(f"Training VQAD for {args.epochs} epochs...")
    losses = detector.train(benign, byzantine, global_model, args.epochs, args.lr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector.save_weights(args.output)
    print(f"Weights saved to {args.output}")
    print(f"Final loss: {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
