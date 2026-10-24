"""
Evaluation script for factory equipment object detector.
Computes IoU, Precision, Recall, Per-class AP, and mAP@0.50 on held-out test set.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

from src.dataset import FactoryEquipmentDataset, detection_collate_fn
from src.train_detector import build_detection_model
from src.config import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, PROJECT_ROOT


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(1e-6, union)


def evaluate_detector(
    weights_path: Path,
    test_split_dir: Path,
    architecture: str = "fasterrcnn_mobilenet_v3_large_fpn",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    output_report_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Evaluates detector performance on held-out test split.
    Saves metrics JSON and human-readable evaluation summary.
    """
    if torch is None:
        return {"error": "PyTorch is not available."}

    if not weights_path.exists():
        return {
            "status": "CHECKPOINT_MISSING",
            "message": f"Weights checkpoint not found at {weights_path}. Train model first or use synthetic mode."
        }

    ann_path = test_split_dir / "_annotations.coco.json"
    if not ann_path.exists():
        return {
            "status": "DATASET_MISSING",
            "message": f"Test split annotations not found at {ann_path}."
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(architecture=architecture, num_classes=7, pretrained=False)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    test_dataset = FactoryEquipmentDataset(test_split_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    # Accumulators per class (class_id: 1..6)
    stats: Dict[int, Dict[str, Any]] = {
        cid: {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "ious": []}
        for cid in range(1, 7)
    }

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                pred_boxes = output["boxes"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                # Filter by confidence threshold
                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                for cid in range(1, 7):
                    c_gt_idx = np.where(gt_labels == cid)[0]
                    c_pred_idx = np.where(pred_labels == cid)[0]
                    stats[cid]["gt_count"] += len(c_gt_idx)

                    matched_gt = set()
                    # Sort preds by score descending
                    c_preds = sorted(c_pred_idx, key=lambda idx: pred_scores[idx], reverse=True)

                    for p_idx in c_preds:
                        pb = pred_boxes[p_idx]
                        best_iou = 0.0
                        best_gt = -1

                        for g_idx in c_gt_idx:
                            if g_idx not in matched_gt:
                                iou = box_iou(pb, gt_boxes[g_idx])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt = g_idx

                        if best_iou >= iou_threshold and best_gt != -1:
                            stats[cid]["tp"] += 1
                            stats[cid]["ious"].append(float(best_iou))
                            matched_gt.add(best_gt)
                        else:
                            stats[cid]["fp"] += 1

                    stats[cid]["fn"] += (len(c_gt_idx) - len(matched_gt))

    # Compile metrics
    per_class_results = {}
    aps = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    for cid in range(1, 7):
        cname = CLASS_ID_TO_NAME[cid]
        tp = stats[cid]["tp"]
        fp = stats[cid]["fp"]
        fn = stats[cid]["fn"]
        gt = stats[cid]["gt_count"]
        ious = stats[cid]["ious"]

        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, gt)
        # Approximate AP@0.50
        ap = prec * rec
        aps.append(ap)

        per_class_results[cname] = {
            "class_id": cid,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "ap50": round(ap, 4),
            "ground_truth_count": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0
        }

    overall_prec = total_tp / max(1, total_tp + total_fp)
    overall_rec = total_tp / max(1, total_tp + total_fn)
    mAP50 = float(np.mean(aps))
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    report = {
        "status": "SUCCESS",
        "eval_dataset": str(test_split_dir),
        "test_images_count": len(test_dataset),
        "confidence_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "mean_average_precision_50": round(mAP50, 4),
        "mean_iou": round(mean_iou, 4),
        "per_class": per_class_results
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
