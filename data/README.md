# Factory Layout Optimizer — Dataset Specifications

## Equipment Classes
The dataset uses a fixed mapping between class names and numeric IDs throughout all annotations, models, and UI components:

| Class ID | Class Name | Default Dimensions (W × H) | Visual Color Hex | Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `__background__` | — | — | Reserved for Faster R-CNN background class |
| **1** | `Cutting machine` | 3.0m × 2.0m | `#E74C3C` (Red) | CNC laser cutters, plasma tables, band saws |
| **2** | `Drilling machine` | 2.5m × 2.0m | `#E67E22` (Orange) | Radial drill presses, automated drill banks |
| **3** | `Assembly workstation`| 3.5m × 2.5m | `#2ECC71` (Green) | Manual assembly benches, sub-assembly cells |
| **4** | `Inspection station` | 2.0m × 2.0m | `#9B59B6` (Purple) | Optical CMMs, manual QA benches, test gauges |
| **5** | `Packaging station` | 3.0m × 2.5m | `#F39C12` (Amber) | Roller conveyors, palletizing & stretch wrappers |
| **6** | `Storage area` | 4.0m × 3.0m | `#3498DB` (Blue) | Raw stock racks, WIP staging buffers, bins |

## Dataset Generation
To generate train, val, and test splits with COCO JSON annotations:
```bash
python scripts/generate_dataset.py --train_count 20 --val_count 5 --test_count 5 --seed 42
```
Or for a quick smoke test:
```bash
python scripts/generate_dataset.py --quick
```

Outputs will be saved in `data/synthetic/train/`, `data/synthetic/val/`, and `data/synthetic/test/`.
Each split contains image files and an `_annotations.coco.json` file.
