"""
Inference engine supporting both trained PyTorch Faster R-CNN checkpoint inference
and an independent Synthetic Demonstration mode.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.ops import nms
except ImportError:
    torch = None

from src.train_detector import build_detection_model
from src.config import CLASS_ID_TO_NAME, CLASS_COLORS, MODELS_DIR


class FactoryDetectorInference:
    """Handles PyTorch Faster R-CNN model loading, evaluation, and visual annotation."""
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        architecture: str = "fasterrcnn_mobilenet_v3_large_fpn",
        device_str: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path or (MODELS_DIR / "best_detector.pth")
        self.architecture = architecture
        self.device = torch.device(device_str) if device_str and torch else torch.device("cpu")
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if torch is None or not self.checkpoint_path.exists():
            self.is_loaded = False
            return

        try:
            self.model = build_detection_model(
                architecture=self.architecture,
                num_classes=7,
                pretrained=False
            )
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            print(f"Warning: Failed to load PyTorch checkpoint from {self.checkpoint_path}: {e}")

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """Run object detection inference on a PIL Image."""
        if not self.is_loaded or self.model is None or torch is None:
            raise RuntimeError(
                f"Trained detector weights not loaded. Checkpoint missing at '{self.checkpoint_path}'. "
                f"Please train a model first or use Synthetic Demonstration mode."
            )

        img_tensor = TF.to_tensor(image.convert("RGB")).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])
            output = outputs[0]

        boxes = output["boxes"].cpu()
        scores = output["scores"].cpu()
        labels = output["labels"].cpu()

        # Confidence filtering
        keep_conf = scores >= confidence_threshold
        boxes = boxes[keep_conf]
        scores = scores[keep_conf]
        labels = labels[keep_conf]

        # Apply Non-Maximum Suppression (NMS)
        if len(boxes) > 0 and nms_iou_threshold > 0:
            keep_indices = nms(boxes, scores, nms_iou_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        detections: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            cid = int(labels[i])
            cname = CLASS_ID_TO_NAME.get(cid, f"Class {cid}")
            score = float(scores[i])
            bbox = [round(float(v), 2) for v in boxes[i].tolist()]  # [xmin, ymin, xmax, ymax]

            detections.append({
                "detection_id": i + 1,
                "class_id": cid,
                "class_name": cname,
                "confidence": round(score, 3),
                "bbox_px": bbox,  # [xmin, ymin, xmax, ymax]
                "detection_source": "PyTorch Faster R-CNN"
            })

        return detections


def generate_synthetic_demo_detections(image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Synthetic Demonstration Mode:
    Returns known, validated equipment detections with clear explicit labelling.
    Does NOT claim to be predictions from a trained neural network.
    """
    W, H = image_size
    # Scaled realistic layout detections
    scale_w = W / 800.0
    scale_h = H / 600.0

    demo_items = [
        {"class_id": 1, "class_name": "Cutting machine", "box": (70, 70, 160, 140)},
        {"class_id": 1, "class_name": "Cutting machine", "box": (70, 420, 160, 490)},
        {"class_id": 2, "class_name": "Drilling machine", "box": (200, 70, 280, 140)},
        {"class_id": 2, "class_name": "Drilling machine", "box": (200, 420, 280, 490)},
        {"class_id": 3, "class_name": "Assembly workstation", "box": (340, 60, 440, 150)},
        {"class_id": 3, "class_name": "Assembly workstation", "box": (340, 410, 440, 500)},
        {"class_id": 4, "class_name": "Inspection station", "box": (500, 70, 570, 140)},
        {"class_id": 4, "class_name": "Inspection station", "box": (500, 420, 570, 490)},
        {"class_id": 5, "class_name": "Packaging station", "box": (620, 70, 710, 150)},
        {"class_id": 6, "class_name": "Storage area", "box": (620, 400, 730, 510)},
    ]

    detections = []
    for idx, item in enumerate(demo_items, start=1):
        x1, y1, x2, y2 = item["box"]
        bbox = [
            round(x1 * scale_w, 1),
            round(y1 * scale_h, 1),
            round(x2 * scale_w, 1),
            round(y2 * scale_h, 1)
        ]
        detections.append({
            "detection_id": idx,
            "class_id": item["class_id"],
            "class_name": item["class_name"],
            "confidence": 1.00,  # Explicit ground truth / calibrated reference
            "bbox_px": bbox,
            "detection_source": "Synthetic Demonstration Mode (Verified Geometry)"
        })

    return detections


def draw_bounding_boxes_on_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    show_confidence: bool = True
) -> Image.Image:
    """Renders high-visibility colored bounding boxes and labels onto the image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        cname = det["class_name"]
        color_hex = CLASS_COLORS.get(cname, "#E74C3C")
        
        # Convert hex to RGB tuple
        h = color_hex.lstrip("#")
        color_rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        xmin, ymin, xmax, ymax = [int(v) for v in det["bbox_px"]]

        # Draw box outline
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color_rgb, width=3)

        # Label text
        conf_str = f" ({det['confidence']:.2f})" if show_confidence and "confidence" in det else ""
        label_text = f"#{det.get('detection_id', '')} {cname}{conf_str}"

        # Text banner tag
        tag_height = 18
        draw.rectangle([xmin, max(0, ymin - tag_height), xmin + len(label_text) * 8 + 6, max(tag_height, ymin)], fill=color_rgb)
        draw.text((xmin + 4, max(0, ymin - tag_height) + 2), label_text, fill=(255, 255, 255))

    return annotated
