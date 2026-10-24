"""
PyTorch Dataset for Factory Equipment Detection with COCO format compatibility.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF
except ImportError:
    # Graceful fallback before PyTorch finishes installing
    torch = None
    Dataset = object


class FactoryEquipmentDataset(Dataset):
    """
    TorchVision-compatible Dataset for factory floor plan images.
    Loads COCO JSON annotations and formats tensors for Faster R-CNN.
    """
    def __init__(self, split_dir: Path, transforms=None):
        self.split_dir = Path(split_dir)
        self.transforms = transforms
        self.ann_file = self.split_dir / "_annotations.coco.json"

        if not self.ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")

        with open(self.ann_file, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.images = self.coco.get("images", [])
        self.img_id_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.split_dir / img_info["file_name"]
        
        image = Image.open(img_path).convert("RGB")
        anns = self.img_id_to_anns.get(img_info["id"], [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # Convert to TorchVision format: [xmin, ymin, xmax, ymax]
            xmin = float(x)
            ymin = float(y)
            xmax = float(x + w)
            ymax = float(y + h)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if torch is not None:
            if boxes:
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
                area_tensor = torch.as_tensor(areas, dtype=torch.float32)
                iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
                area_tensor = torch.zeros((0,), dtype=torch.float32)
                iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

            target = {
                "boxes": boxes_tensor,
                "labels": labels_tensor,
                "image_id": torch.tensor([img_info["id"]], dtype=torch.int64),
                "area": area_tensor,
                "iscrowd": iscrowd_tensor
            }

            # Basic image tensor conversion
            img_tensor = TF.to_tensor(image)

            if self.transforms is not None:
                img_tensor, target = self.transforms(img_tensor, target)

            return img_tensor, target

        return image, {"boxes": boxes, "labels": labels, "image_id": img_info["id"]}


def detection_collate_fn(batch):
    """Custom collate function for detection batches of variable bounding box counts."""
    return tuple(zip(*batch))
