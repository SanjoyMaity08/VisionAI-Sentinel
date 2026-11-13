"""High-detail detection helpers for small objects in webcam frames."""
from __future__ import annotations

from collections import deque
import os
import platform
import sys
from typing import Any

import numpy as np


def prime_windows_platform_cache() -> None:
    """Fill platform.uname() from env data so torch's import never blocks on WMI.

    torch calls platform.machine() while loading its DLLs, and on some Windows
    machines the underlying WMI query stalls for minutes. These values are what
    the same query would report, so the answer is identical without the wait.
    """
    if not sys.platform.startswith("win"):
        return
    if getattr(platform, "_uname_cache", None) is not None:
        return
    try:
        windows = sys.getwindowsversion()
        arch = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64")
        cached = platform.uname_result(
            "Windows",
            os.environ.get("COMPUTERNAME", ""),
            f"{windows.major}.{windows.minor}",
            f"{windows.major}.{windows.minor}.{windows.build}",
            arch,
        )
        cached.__dict__["processor"] = os.environ.get("PROCESSOR_IDENTIFIER", "")
        platform._uname_cache = cached
    except (AttributeError, TypeError):
        pass


prime_windows_platform_cache()


def configure_torch_threads(predictor: Any | None = None) -> None:
    """Restore the CPU thread cap after Ultralytics resets it during predictor setup."""
    if predictor is not None and predictor.device.type != "cpu":
        return
    try:
        import torch

        desired = int(os.environ.get("VISIONAI_TORCH_THREADS", "0") or 0)
        if desired <= 0:
            desired = max(1, min(2, (os.cpu_count() or 2) // 2))
        if torch.get_num_threads() != desired:
            torch.set_num_threads(desired)
    except Exception:
        pass


def _select_device() -> str | None:
    """Honor VISIONAI_DEVICE; otherwise let Ultralytics auto-select (GPU if present)."""
    forced = os.environ.get("VISIONAI_DEVICE", "").strip()
    return forced or None


DETECTION_DEVICE = _select_device()
DETECTION_IMAGE_SIZE = 512
TILE_IMAGE_SIZE = 512
MAX_DETECTIONS = 60

MIN_CONFIDENCE_BY_LABEL = {
    "train": 0.85,
    "car": 0.60,
    "motorcycle": 0.55,
    "bus": 0.65,
    "truck": 0.65,
    "baseball bat": 0.88,
    "cell phone": 0.72,
    "bottle": 0.45,
    "hair drier": 0.80,
    "toilet": 0.80,
    "umbrella": 0.80,
    "laptop": 0.55,
    "cup": 0.50,
    "dog": 0.85,
    "cat": 0.85,
    "bird": 0.80,
    "horse": 0.80,
    "cow": 0.80,
    "sheep": 0.80,
    "elephant": 0.80,
    "bear": 0.80,
    "zebra": 0.80,
    "giraffe": 0.80,
    "airplane": 0.90,
    "boat": 0.85,
}
DEFAULT_MIN_CONFIDENCE = 0.55
OPEN_VOCABULARY_MIN_CONFIDENCE = 0.60
OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE = {
    "computer monitor": 0.55,
    "camera": 0.75,
    "keys": 0.75,
}
REFERENCE_MIN_SIMILARITY = 0.60
REFERENCE_MIN_MARGIN = 0.04
SECONDARY_REQUIRED_LABELS = {"airplane", "hair drier", "toilet", "cell phone", "remote", "cup", "vase"}
REFERENCE_REQUIRED_LABELS = {
    "bag",
    "notebook",
    "shoes",
    "adaptor",
    "switch board",
    "towel",
    "utensils",
    "umbrella",
    "fan",
    "good night machine",
    "pen pencil",
}
AMBIGUOUS_PRIMARY_LABELS = {"tv"}


def _read_result(result: Any, model: Any, offset_x: int = 0, offset_y: int = 0,
                 width: int | None = None, height: int | None = None) -> list[dict[str, Any]]:
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = str(model.names.get(class_id, "")).lower().strip()
        coords = [int(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0].item())
        if not label or len(coords) != 4:
            continue

        x1, y1, x2, y2 = coords
        x1, x2 = x1 + offset_x, x2 + offset_x
        y1, y2 = y1 + offset_y, y2 + offset_y
        if width is not None and height is not None:
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "class_id": class_id,
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "conf": confidence,
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })
    return detections


def _overlap(first: list[int], second: list[int]) -> tuple[float, float]:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    smaller_area_overlap = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    return iou, smaller_area_overlap


def _same_object(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["label"] != second["label"] or first["class_id"] != second["class_id"]:
        return False
    iou, smaller_area_overlap = _overlap(first["bbox"], second["bbox"])
    return iou >= 0.5 or smaller_area_overlap >= 0.65


def _deduplicate(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: item["conf"], reverse=True):
        if not any(
            _same_object(item, detection)
            for item in kept
        ):
            kept.append(detection)
    return kept


def _passes_quality_gate(detection: dict[str, Any], width: int, height: int) -> bool:
    if detection["label"] in AMBIGUOUS_PRIMARY_LABELS:
        return False
    if detection["conf"] < MIN_CONFIDENCE_BY_LABEL.get(
        detection["label"], DEFAULT_MIN_CONFIDENCE
    ):
        return False
    if detection["label"] == "train":
        x1, y1, x2, y2 = detection["bbox"]
        frame_area = width * height
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if frame_area and box_area / frame_area >= 0.85:
            return False
    return True


ANIMAL_LABELS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def _box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _filter_person_overlaps(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A person's torso is sometimes misread as a pet; a wrong name is worse than no name."""
    persons = [d for d in detections if d["label"] == "person"]
    if not persons:
        return detections
    return [
        d for d in detections
        if d["label"] not in ANIMAL_LABELS
        or not any(_box_iou(d["bbox"], p["bbox"]) >= 0.45 for p in persons)
    ]


class DetectionStabilizer:
    """Only publish detections that remain spatially consistent across recent frames."""

    def __init__(
        self,
        history_size: int = 3,
        minimum_hits: int = 2,
        max_age_sec: float = 3.0,
    ) -> None:
        self._history: deque[tuple[float, list[dict[str, Any]]]] = deque(maxlen=history_size)
        self._minimum_hits = minimum_hits
        self._max_age_sec = max_age_sec

    def reset(self) -> None:
        self._history.clear()

    def update(self, detections: list[dict[str, Any]], observed_at: float) -> list[dict[str, Any]]:
        confirmed: list[dict[str, Any]] = []
        for detection in detections:
            hit_count = 1
            for previous_at, previous_frame in self._history:
                if observed_at - previous_at > self._max_age_sec:
                    continue
                if any(_same_object(detection, previous) for previous in previous_frame):
                    hit_count += 1
            if hit_count >= self._minimum_hits:
                confirmed.append(detection)
        self._history.append((observed_at, [dict(detection) for detection in detections]))
        return confirmed


def _tile_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]
    return [0, length - tile_size]


def detect_with_detail_tiles(
    model: Any,
    frame: np.ndarray,
    confidence: float,
    detail_scan: bool,
    previous_tile_detections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run full-frame detection and periodically confirm overlapping tile detections."""
    height, width = frame.shape[:2]
    effective_confidence = max(DEFAULT_MIN_CONFIDENCE, confidence)
    full_result = model.predict(
        source=frame,
        imgsz=DETECTION_IMAGE_SIZE,
        conf=effective_confidence,
        device=DETECTION_DEVICE,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )[0]
    full_detections = [
        detection for detection in _read_result(full_result, model, width=width, height=height)
        if _passes_quality_gate(detection, width, height)
    ]
    full_detections = _filter_person_overlaps(full_detections)
    if not detail_scan:
        return full_detections, previous_tile_detections

    tile_width = min(width, max(320, int(width * 0.65)))
    tile_height = min(height, max(320, int(height * 0.65)))
    regions = [
        (x, y, frame[y:y + tile_height, x:x + tile_width])
        for y in _tile_starts(height, tile_height)
        for x in _tile_starts(width, tile_width)
    ]
    tile_confidence = max(DEFAULT_MIN_CONFIDENCE, min(confidence, 0.55))
    tile_results = model.predict(
        source=[region[2] for region in regions],
        imgsz=TILE_IMAGE_SIZE,
        conf=tile_confidence,
        device=DETECTION_DEVICE,
        max_det=MAX_DETECTIONS,
        verbose=False,
    )

    current_tiles: list[dict[str, Any]] = []
    for (offset_x, offset_y, _), result in zip(regions, tile_results):
        current_tiles.extend(
            detection
            for detection in _read_result(result, model, offset_x, offset_y, width, height)
            if _passes_quality_gate(detection, width, height)
        )
    current_tiles = _filter_person_overlaps(_deduplicate(current_tiles))

    confirmed_tiles = [
        detection for detection in current_tiles
        if any(_same_object(detection, item) for item in full_detections)
        or any(_same_object(detection, item) for item in previous_tile_detections)
    ]
    return _deduplicate(_filter_person_overlaps(full_detections + confirmed_tiles)), current_tiles
