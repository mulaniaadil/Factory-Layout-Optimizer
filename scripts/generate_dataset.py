"""
CLI script to generate synthetic factory floor images with COCO annotations.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.data_generation import SyntheticFactoryGenerator, generate_sample_factory_image
from src.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic factory dataset")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR / "synthetic"), help="Dataset destination path")
    parser.add_argument("--train_count", type=int, default=15, help="Number of training images")
    parser.add_argument("--val_count", type=int, default=4, help="Number of validation images")
    parser.add_argument("--test_count", type=int, default=4, help="Number of test images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Generate small smoke test dataset (5 train, 2 val, 2 test)")

    args = parser.parse_args()

    train_c = 5 if args.quick else args.train_count
    val_c = 2 if args.quick else args.val_count
    test_c = 2 if args.quick else args.test_count

    output_path = Path(args.output_dir)
    print(f"Generating synthetic factory dataset at: {output_path}")
    print(f"Splits -> Train: {train_c}, Val: {val_c}, Test: {test_c} (Seed: {args.seed})")

    gen = SyntheticFactoryGenerator(image_size=(800, 600), seed=args.seed)
    paths = gen.generate_coco_dataset(
        output_dir=output_path,
        split_counts={"train": train_c, "val": val_c, "test": test_c}
    )

    sample_img_path = DATA_DIR / "sample_factory_floor.png"
    generate_sample_factory_image(sample_img_path, seed=101)
    print(f"Generated standalone demo sample image at: {sample_img_path}")
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
