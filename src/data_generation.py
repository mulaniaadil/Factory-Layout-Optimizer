"""
Synthetic Factory Floor Plan Image and COCO Annotation Generator.
Produces photorealistic, top-down 2D floor plans with structural walls,
concrete tiling, aisle markings, machine details, and valid bounding boxes.
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.config import CLASS_ID_TO_NAME, CLASS_NAME_TO_ID, CLASS_COLORS, DATA_DIR


# Visual styling constants
EQUIPMENT_STYLE_CONFIG = {
    1: {"name": "Cutting machine", "base_color": (231, 76, 60), "detail_color": (192, 57, 43), "type": "laser"},
    2: {"name": "Drilling machine", "base_color": (230, 126, 34), "detail_color": (211, 84, 0), "type": "drill"},
    3: {"name": "Assembly workstation", "base_color": (46, 204, 113), "detail_color": (39, 174, 96), "type": "bench"},
    4: {"name": "Inspection station", "base_color": (155, 89, 182), "detail_color": (142, 68, 173), "type": "optics"},
    5: {"name": "Packaging station", "base_color": (243, 156, 18), "detail_color": (230, 126, 34), "type": "roller"},
    6: {"name": "Storage area", "base_color": (52, 152, 219), "detail_color": (41, 128, 185), "type": "racks"},
}


class SyntheticFactoryGenerator:
    """Generates synthetic factory floor images with COCO-compliant bounding-box annotations."""
    def __init__(
        self,
        image_size: Tuple[int, int] = (800, 600),
        min_machines: int = 6,
        max_machines: int = 12,
        seed: int = 42
    ):
        self.width, self.height = image_size
        self.min_machines = min_machines
        self.max_machines = max_machines
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _draw_floor_texture(self, draw: ImageDraw.ImageDraw):
        """Draw realistic industrial concrete epoxy floor with subtle grid lines."""
        # Base background: industrial light warm grey
        draw.rectangle([0, 0, self.width, self.height], fill=(235, 237, 240))

        # Fine grid lines (simulating concrete expansion joints / floor tiles)
        grid_step = 40
        for x in range(0, self.width, grid_step):
            draw.line([(x, 0), (x, self.height)], fill=(222, 225, 230), width=1)
        for y in range(0, self.height, grid_step):
            draw.line([(0, y), (self.width, y)], fill=(222, 225, 230), width=1)

        # Outer factory walls (dark structural border)
        wall_thick = 10
        draw.rectangle([0, 0, self.width - 1, self.height - 1], outline=(52, 73, 94), width=wall_thick)

        # Main central forklift aisle lines (safety yellow dashed lines)
        aisle_y = self.height // 2
        for x in range(20, self.width - 20, 30):
            draw.line([(x, aisle_y - 25), (x + 15, aisle_y - 25)], fill=(241, 196, 15), width=2)
            draw.line([(x, aisle_y + 25), (x + 15, aisle_y + 25)], fill=(241, 196, 15), width=2)

    def _draw_machine_graphics(
        self,
        draw: ImageDraw.ImageDraw,
        box: Tuple[int, int, int, int],
        class_id: int,
        label: str
    ):
        """Draw realistic mechanical top-down details for each machine type."""
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        cfg = EQUIPMENT_STYLE_CONFIG.get(class_id, EQUIPMENT_STYLE_CONFIG[1])
        base_col = cfg["base_color"]
        det_col = cfg["detail_color"]

        # 1. Safety perimeter boundary line (yellow hatched or solid zone)
        margin = 6
        draw.rectangle(
            [xmin - margin, ymin - margin, xmax + margin, ymax + margin],
            outline=(243, 156, 18),
            width=1
        )

        # 2. Main physical equipment footprint body
        draw.rectangle([xmin, ymin, xmax, ymax], fill=base_col, outline=(44, 62, 80), width=2)

        # 3. Machine-specific internal components
        m_type = cfg["type"]
        if m_type == "laser":  # Cutting: cutting bed grid + laser gantry bar
            for gx in range(xmin + 8, xmax - 8, 12):
                draw.line([(gx, ymin + 8), (gx, ymax - 8)], fill=det_col, width=1)
            # Gantry rail
            draw.line([(xmin + 6, ymin + h // 2), (xmax - 6, ymin + h // 2)], fill=(255, 255, 255), width=3)
            draw.ellipse([(xmin + w // 2 - 4, ymin + h // 2 - 4), (xmin + w // 2 + 4, ymin + h // 2 + 4)], fill=(255, 255, 0))

        elif m_type == "drill":  # Drilling: spindle rings and drill table
            draw.rectangle([xmin + 6, ymin + 6, xmax - 6, ymax - 6], outline=det_col, width=2)
            draw.ellipse([(xmin + w // 2 - 10, ymin + h // 2 - 10), (xmin + w // 2 + 10, ymin + h // 2 + 10)], outline=(255, 255, 255), width=2)
            draw.ellipse([(xmin + w // 2 - 4, ymin + h // 2 - 4), (xmin + w // 2 + 4, ymin + h // 2 + 4)], fill=(255, 255, 255))

        elif m_type == "bench":  # Assembly workstation: work table + bin trays
            draw.rectangle([xmin + 4, ymin + 4, xmin + 14, ymax - 4], fill=det_col)
            draw.rectangle([xmax - 14, ymin + 4, xmax - 4, ymax - 4], fill=det_col)
            draw.rectangle([xmin + 18, ymin + 6, xmax - 18, ymax - 6], outline=(255, 255, 255), width=1)

        elif m_type == "optics":  # Inspection: CMM optical lens & granite surface plate
            draw.rectangle([xmin + 6, ymin + 6, xmax - 6, ymax - 6], fill=(52, 73, 94))
            draw.ellipse([(xmin + w // 2 - 8, ymin + h // 2 - 8), (xmin + w // 2 + 8, ymin + h // 2 + 8)], fill=(174, 214, 241), outline=(255, 255, 255), width=1)

        elif m_type == "roller":  # Packaging: roller conveyor slats
            for ry in range(ymin + 6, ymax - 6, 8):
                draw.line([(xmin + 4, ry), (xmax - 4, ry)], fill=(236, 240, 241), width=2)

        elif m_type == "racks":  # Storage: pallet racking shelf dividers
            draw.line([(xmin + w // 2, ymin + 4), (xmin + w // 2, ymax - 4)], fill=(255, 255, 255), width=2)
            for py in range(ymin + 8, ymax - 8, 14):
                draw.rectangle([xmin + 4, py, xmin + w // 2 - 4, py + 8], fill=(245, 176, 65))
                draw.rectangle([xmin + w // 2 + 4, py, xmax - 4, py + 8], fill=(245, 176, 65))

        # 4. Small operator terminal / emergency stop button
        draw.rectangle([xmin + 2, ymin + 2, xmin + 8, ymin + 8], fill=(231, 76, 60))

        # 5. Label text on top of machine
        draw.text((xmin + 4, ymax - 14), label[:10], fill=(255, 255, 255))

    def generate_single_layout(self) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        """Generates one factory floor image and its ground truth bounding boxes."""
        img = Image.new("RGB", (self.width, self.height), color=(235, 237, 240))
        draw = ImageDraw.Draw(img)

        self._draw_floor_texture(draw)

        num_machines = self.rng.randint(self.min_machines, self.max_machines)
        placed_boxes: List[Tuple[int, int, int, int]] = []
        annotations: List[Dict[str, Any]] = []

        # Margin from perimeter walls and docks
        pad = 25
        min_w, max_w = 60, 110
        min_h, max_h = 50, 90

        # Try placing machines without physical overlap
        max_attempts = 200
        for m_idx in range(num_machines):
            class_id = self.rng.randint(1, 6)
            class_name = CLASS_ID_TO_NAME[class_id]

            placed = False
            for _ in range(max_attempts):
                mw = self.rng.randint(min_w, max_w)
                mh = self.rng.randint(min_h, max_h)
                x = self.rng.randint(pad, self.width - pad - mw)
                y = self.rng.randint(pad, self.height - pad - mh)

                # Avoid center forklift main thoroughfare
                aisle_y = self.height // 2
                if abs((y + mh // 2) - aisle_y) < 35:
                    continue

                # Check overlap against existing machines (with 15px clearance)
                box_cand = (x, y, x + mw, y + mh)
                clearance_buffer = 15
                cand_buffered = (
                    x - clearance_buffer,
                    y - clearance_buffer,
                    x + mw + clearance_buffer,
                    y + mh + clearance_buffer
                )

                collision = False
                for pb in placed_boxes:
                    if not (cand_buffered[2] <= pb[0] or cand_buffered[0] >= pb[2] or
                            cand_buffered[3] <= pb[1] or cand_buffered[1] >= pb[3]):
                        collision = True
                        break

                if not collision:
                    placed_boxes.append(box_cand)
                    self._draw_machine_graphics(draw, box_cand, class_id, f"M{m_idx+1}")

                    annotations.append({
                        "id": m_idx + 1,
                        "class_id": class_id,
                        "class_name": class_name,
                        "bbox_px": [box_cand[0], box_cand[1], box_cand[2], box_cand[3]],  # [xmin, ymin, xmax, ymax]
                        "coco_bbox": [box_cand[0], box_cand[1], mw, mh],  # [x, y, w, h]
                        "area": mw * mh
                    })
                    placed = True
                    break

        return img, annotations

    def generate_coco_dataset(
        self,
        output_dir: Path,
        split_counts: Dict[str, int] = {"train": 20, "val": 5, "test": 5}
    ) -> Dict[str, Path]:
        """
        Creates train, val, test splits in standard COCO format.
        Avoids duplicate or near-identical images across splits.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        split_paths: Dict[str, Path] = {}

        categories = [
            {"id": cid, "name": cname, "supercategory": "factory_equipment"}
            for cid, cname in CLASS_ID_TO_NAME.items()
            if cid > 0
        ]

        global_img_id = 1
        global_ann_id = 1

        for split_name, count in split_counts.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            images_coco = []
            annotations_coco = []

            for i in range(count):
                img, anns = self.generate_single_layout()
                img_filename = f"{split_name}_{i:04d}.png"
                img_filepath = split_dir / img_filename
                img.save(img_filepath)

                img_info = {
                    "id": global_img_id,
                    "file_name": img_filename,
                    "width": self.width,
                    "height": self.height
                }
                images_coco.append(img_info)

                for ann in anns:
                    coco_ann = {
                        "id": global_ann_id,
                        "image_id": global_img_id,
                        "category_id": ann["class_id"],
                        "bbox": ann["coco_bbox"],
                        "area": float(ann["area"]),
                        "iscrowd": 0
                    }
                    annotations_coco.append(coco_ann)
                    global_ann_id += 1

                global_img_id += 1

            coco_data = {
                "categories": categories,
                "images": images_coco,
                "annotations": annotations_coco
            }

            ann_file = split_dir / "_annotations.coco.json"
            with open(ann_file, "w", encoding="utf-8") as f:
                json.dump(coco_data, f, indent=2)

            split_paths[split_name] = split_dir

        return split_paths


def generate_sample_factory_image(output_path: Path, seed: int = 101) -> Path:
    """Generates a high-quality demonstration factory floor image bundled with the app."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gen = SyntheticFactoryGenerator(image_size=(960, 640), min_machines=8, max_machines=10, seed=seed)
    img, _ = gen.generate_single_layout()
    img.save(output_path)
    return output_path
