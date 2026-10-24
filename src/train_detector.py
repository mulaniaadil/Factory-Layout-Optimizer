"""
PyTorch Detector Training Pipeline for Factory Equipment Object Detection.
Supports Faster R-CNN with ResNet-50 FPN or MobileNetV3 FPN backbones.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader
except ImportError:
    torch = None

from src.dataset import FactoryEquipmentDataset, detection_collate_fn
from src.config import CLASS_ID_TO_NAME, MODELS_DIR


def build_detection_model(
    architecture: str = "fasterrcnn_mobilenet_v3_large_fpn",
    num_classes: int = 7,
    pretrained: bool = True
):
    """
    Constructs Faster R-CNN and replaces classification/box head for factory equipment.
    num_classes = 7 (0=background, 1..6=equipment classes).
    """
    if torch is None:
        raise RuntimeError("PyTorch is not yet installed in the current environment.")

    if architecture == "fasterrcnn_resnet50_fpn":
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        # Default to lighter, highly CPU-friendly MobileNetV3-Large FPN
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def train_detector(
    dataset_dir: Path,
    output_model_path: Optional[Path] = None,
    architecture: str = "fasterrcnn_mobilenet_v3_large_fpn",
    num_epochs: int = 5,
    batch_size: int = 2,
    lr: float = 0.0005,
    device_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes complete training loop on factory floor dataset.
    Logs loss history and saves checkpoint + configuration.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for training.")

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    if not (train_dir / "_annotations.coco.json").exists():
        raise FileNotFoundError(f"Missing training annotations in {train_dir}")

    train_dataset = FactoryEquipmentDataset(train_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn
    )

    has_val = (val_dir / "_annotations.coco.json").exists()
    val_loader = None
    if has_val:
        val_dataset = FactoryEquipmentDataset(val_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=detection_collate_fn
        )

    model = build_detection_model(architecture=architecture, num_classes=7, pretrained=True)
    model.to(device)

    # Optimizer with slight weight decay
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    history = []
    best_loss = float("inf")
    save_path = output_model_path or (MODELS_DIR / "best_detector.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_train_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            num_batches += 1

        lr_scheduler.step()
        avg_train_loss = epoch_loss / max(1, num_batches)

        # Validation phase
        val_loss = 0.0
        if val_loader:
            # Note: TorchVision detection model outputs loss in train mode when passed targets
            with torch.no_grad():
                val_batches = 0
                for images, targets in val_loader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    val_batches += 1
                val_loss = val_loss / max(1, val_batches)
        else:
            val_loss = avg_train_loss

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": architecture,
                    "num_classes": 7,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "class_id_to_name": CLASS_ID_TO_NAME
                },
                save_path
            )

    total_time = time.time() - start_train_time

    # Save training metadata configuration
    meta_path = save_path.parent / "detector_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": architecture,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "device": str(device),
                "total_train_time_sec": round(total_time, 2),
                "best_loss": round(best_loss, 4),
                "saved_weights": str(save_path.name),
                "class_mapping": CLASS_ID_TO_NAME,
                "history": history
            },
            f,
            indent=2
        )

    return {
        "model_path": save_path,
        "metadata_path": meta_path,
        "history": history,
        "total_time": total_time,
        "best_loss": best_loss
    }
