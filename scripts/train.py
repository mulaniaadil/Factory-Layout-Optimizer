"""
CLI script to train PyTorch factory equipment detector.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.train_detector import train_detector
from src.config import DATA_DIR, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch factory equipment detector")
    parser.add_argument("--dataset_dir", type=str, default=str(DATA_DIR / "synthetic"), help="Path to synthetic dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--arch", type=str, default="fasterrcnn_mobilenet_v3_large_fpn", help="Architecture")
    parser.add_argument("--output_model", type=str, default=str(MODELS_DIR / "best_detector.pth"), help="Output path")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")

    args = parser.parse_args()

    print(f"Starting detector training ({args.arch}) for {args.epochs} epochs...")
    result = train_detector(
        dataset_dir=Path(args.dataset_dir),
        output_model_path=Path(args.output_model),
        architecture=args.arch,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device
    )

    print("Training finished successfully!")
    print(f"Model saved to: {result['model_path']}")
    print(f"Best Loss: {result['best_loss']:.4f}")
    print(f"Total training time: {result['total_time']:.2f} seconds")


if __name__ == "__main__":
    main()
