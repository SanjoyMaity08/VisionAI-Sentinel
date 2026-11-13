# Third-party notices

VisionAI Sentinel itself is provided as-is without an added commercial license
grant. Model weights are not distributed inside this repository; fetch them
with `python download_models.py`. Review the licensing terms of every
downloaded component before sale, redistribution, or deployment.

## Ultralytics models (YOLO)

- `yolo26n.pt`, `yolo26n-pose.pt`, `yolo11n.pt`, `yolov8s-worldv2.pt`
- Downloaded automatically by Ultralytics from its official assets.
- Ultralytics software is licensed under AGPL-3.0; a commercial license is
  available. Model weight terms are published by Ultralytics.

## CLIP ViT-B/32

- `weights/clip/ViT-B-32.pt`
- Downloaded automatically by the `clip` Python package (OpenAI CLIP).
- CLIP code and weights are distributed under the MIT license
  (https://github.com/openai/CLIP).

## License-plate detector

- `weights/license_plate_yolo11n.pt`
- Model: YOLO11n license-plate detector (`weights/best.pt`) from
  https://github.com/Mrvikas06/License-Plate-Detection
- Pinned source revision: `335cb92d92facf873c5b670a9dc5456e504757aa`
- Source repository license: MIT.
- The source repository describes its training dataset as Roboflow Universe
  `rayans-lab/license-plate-detection-9vgdz/11`. Dataset terms may differ from
  the repository software license and were not independently verified.
- Plate localization and OCR are fallible; verify recognized characters
  against the uploaded footage before taking action.
