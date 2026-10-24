# Factory Equipment Detection Models

## Supported Architectures
1. **`fasterrcnn_mobilenet_v3_large_fpn`** (Default for local CPU):
   - Fast, lightweight inference.
   - Ideal for laptops and desktop test environments.
2. **`fasterrcnn_resnet50_fpn`** (Supported for GPU / Colab):
   - Higher capacity ResNet-50 backbone.
   - Recommended for GPU training in Google Colab.

## Training Checkpoint Location
When trained via `scripts/train.py` or the Colab notebook, the best weights are saved to:
`models/best_detector.pth`

Accompanying training metadata is stored in:
`models/detector_metadata.json`

## Note on Synthetic Demonstration Mode
If `best_detector.pth` is absent, the Streamlit application seamlessly runs in **Synthetic Demonstration Mode** using verified geometric reference boxes. The application will never crash due to a missing checkpoint.
