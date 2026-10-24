"""
CLI script to evaluate detector checkpoint on test split.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from src.evaluate_detector import evaluate_detector
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate factory equipment detector")
    parser.add_argument("--weights", type=str, default=str(MODELS_DIR / "best_detector.pth"), help="Path to weights")
    parser.add_argument("--test_dir", type=str, default=str(DATA_DIR / "synthetic" / "test"), help="Test split path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--arch", type=str, default="fasterrcnn_mobilenet_v3_large_fpn", help="Architecture")
    parser.add_argument("--report", type=str, default=str(REPORTS_DIR / "eval_report.json"), help="Output JSON report")

    args = parser.parse_args()

    print(f"Evaluating detector on {args.test_dir}...")
    metrics = evaluate_detector(
        weights_path=Path(args.weights),
        test_split_dir=Path(args.test_dir),
        architecture=args.arch,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        output_report_path=Path(args.report)
    )

    if metrics.get("status") == "SUCCESS":
        print("\n--- EVALUATION RESULTS ---")
        print(f"Mean Average Precision (mAP@0.50): {metrics['mean_average_precision_50']:.4f}")
        print(f"Overall Precision: {metrics['overall_precision']:.4f}")
        print(f"Overall Recall: {metrics['overall_recall']:.4f}")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        print(f"Detailed report saved to: {args.report}")
    else:
        print(f"\nEvaluation status: {metrics.get('status', 'ERROR')}")
        print(f"Message: {metrics.get('message', 'Unknown issue')}")


if __name__ == "__main__":
    main()
