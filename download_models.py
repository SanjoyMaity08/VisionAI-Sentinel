"""Download the model weights VisionAI Sentinel needs.

The repository intentionally contains no model binaries. Run this script once
after cloning (or after copying the project to a new machine) to fetch every
weight file from its official source into the expected local paths.

    python download_models.py            # fetch only missing files
    python download_models.py --force    # re-download and overwrite existing files

Requires internet access. The CLIP weights are approximately 338 MB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
CLIP_DIR = WEIGHTS_DIR / "clip"

PLATE_MODEL_SOURCE = (
    "https://github.com/Mrvikas06/License-Plate-Detection/raw/"
    "335cb92d92facf873c5b670a9dc5456e504757aa/weights/best.pt"
)
PLATE_MODEL_ATTRIBUTION = """License-plate detector weights attribution

Model: YOLO11n license-plate detector (weights/best.pt)
Source: https://github.com/Mrvikas06/License-Plate-Detection
Pinned source revision: 335cb92d92facf873c5b670a9dc5456e504757aa
Source repository license: MIT (see the repository LICENSE file).
The repository describes the training dataset as Roboflow Universe:
rayans-lab/license-plate-detection-9vgdz/11. Dataset terms may differ from
the repository software license and were not independently verified.

The model is provided as-is. Plate localization and OCR are fallible; verify
recognized characters against the uploaded footage before taking action.
See also THIRD_PARTY_NOTICES.md in the project root.
"""

# (name, destination, required, description)
ULTRALYTICS_MODELS = (
    ("yolo26n.pt", ROOT / "yolo26n.pt", True, "Primary object detection"),
    ("yolo11n.pt", ROOT / "yolo11n.pt", True, "Uploaded-video detection"),
    ("yolo26n-pose.pt", ROOT / "yolo26n-pose.pt", False, "Pose screening for possible falls and raised arms"),
    ("yolov8s-worldv2.pt", ROOT / "yolov8s-worldv2.pt", False, "Open-vocabulary object detection and helmet screening"),
)


def _status(message: str) -> None:
    print(message, flush=True)


def fetch_ultralytics(name: str, destination: Path, force: bool) -> bool:
    if destination.is_file() and not force:
        _status(f"[skip] {destination.name} already present")
        return True
    try:
        from ultralytics import YOLO

        _status(f"[download] {name} (Ultralytics assets)")
        YOLO(name)
        downloaded = Path.cwd() / name
        if not downloaded.is_file():
            raise FileNotFoundError(f"Ultralytics did not create {name} in the working directory")
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)
        _status(f"[ok] {destination}")
        return True
    except Exception as exc:
        _status(f"[failed] {name}: {exc}")
        return False


def fetch_clip(destination: Path, force: bool) -> bool:
    if destination.is_file() and not force:
        _status(f"[skip] {destination.name} already present")
        return True
    try:
        import clip

        _status(f"[download] ViT-B-32 CLIP weights to {destination.parent} (about 338 MB)")
        CLIP_DIR.mkdir(parents=True, exist_ok=True)
        clip.load("ViT-B/32", device="cpu", download_root=str(CLIP_DIR))
        if not destination.is_file():
            raise FileNotFoundError("CLIP did not create ViT-B-32.pt in the weights/clip folder")
        _status(f"[ok] {destination}")
        return True
    except Exception as exc:
        _status(f"[failed] ViT-B-32.pt: {exc}")
        return False


def fetch_plate_model(destination: Path, force: bool) -> bool:
    if destination.is_file() and not force:
        _status(f"[skip] {destination.name} already present")
        return True
    try:
        import requests

        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        _status(f"[download] {destination.name} from {PLATE_MODEL_SOURCE}")
        with requests.get(PLATE_MODEL_SOURCE, stream=True, timeout=120) as response:
            response.raise_for_status()
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
            temporary.replace(destination)
        (WEIGHTS_DIR / "LICENSE-PLATE-MODEL.txt").write_text(
            PLATE_MODEL_ATTRIBUTION, encoding="utf-8"
        )
        _status(f"[ok] {destination}")
        return True
    except Exception as exc:
        _status(f"[failed] {destination.name}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch VisionAI Sentinel model weights.")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    arguments = parser.parse_args()

    failures = 0
    for name, destination, _required, description in ULTRALYTICS_MODELS:
        _status(f"== {description} ==")
        if not fetch_ultralytics(name, destination, arguments.force):
            failures += 1
    _status("== CLIP photo-reference matching ==")
    if not fetch_clip(CLIP_DIR / "ViT-B-32.pt", arguments.force):
        failures += 1
    _status("== Specialised license-plate detection ==")
    if not fetch_plate_model(WEIGHTS_DIR / "license_plate_yolo11n.pt", arguments.force):
        failures += 1

    _status("")
    _status("Summary:")
    for name, destination, required, _description in ULTRALYTICS_MODELS:
        state = "present" if destination.is_file() else "MISSING"
        _status(f"  {destination.relative_to(ROOT):<36} {state}{' (required)' if required else ''}")
    for label, path, required in (
        ("CLIP weights", CLIP_DIR / "ViT-B-32.pt", False),
        ("Plate detector", WEIGHTS_DIR / "license_plate_yolo11n.pt", True),
    ):
        state = "present" if path.is_file() else "MISSING"
        _status(f"  {path.relative_to(ROOT):<36} {state}{' (required)' if required else ''}")

    if failures:
        _status("")
        _status(f"{failures} item(s) failed. Re-run to retry; existing files were kept.")
        return 1
    _status("")
    _status("All models are ready. Start the dashboard with: python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
