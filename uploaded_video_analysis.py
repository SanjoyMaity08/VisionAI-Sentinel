"""YOLO11 vehicle, plate, helmet, red-light, speed, and person-absence analysis for uploaded video."""
from __future__ import annotations

import base64
import math
import re
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_detection import DETECTION_DEVICE, configure_torch_threads


ROOT = Path(__file__).resolve().parent
INCIDENTS_DIR = ROOT / "incidents"

VEHICLE_MODEL_PATH = ROOT / "yolo11n.pt"
VEHICLE_MODEL_NAME = str(VEHICLE_MODEL_PATH) if VEHICLE_MODEL_PATH.is_file() else "yolo11n.pt"
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle"}
PERSON_CLASS_ID = 0
TRAFFIC_LIGHT_CLASS_ID = 9
VEHICLE_CONFIDENCE = 0.3
TRAFFIC_LIGHT_CONFIDENCE = 0.25
PLATE_CONFIDENCE = 0.25
PERSON_ABSENCE_SECONDS = 10.0
SAMPLE_RATE_FPS = 5.0

PLATE_MODEL_PATH = ROOT / "weights" / "license_plate_yolo11n.pt"
HELMET_MODEL_PATH = ROOT / "yolov8s-worldv2.pt"
HELMET_LABELS = ("helmet", "head")
HELMET_CONFIDENCE = 0.30
HELMET_SCAN_INTERVAL_SEC = 1.0
HELMET_CONFIRMATIONS = 2

PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_OCR_INTERVAL_SEC = 0.8
PLATE_READ_CONFIDENCE = 0.80
PLATE_STABLE_VOTES = 2
PLATE_UNREADABLE_SIGHTINGS = 6

RED_LIGHT_MIN_FRACTION = 0.12
RED_LIGHT_MIN_STEADY_SEC = 0.4
RED_LIGHT_EVENT_GAP_SEC = 2.0
RED_STATE_GAP_TOLERANCE_SEC = 1.2

REFERENCE_VEHICLE_LENGTH_M = {2: 4.5, 3: 2.2}
SPEED_MIN_SAMPLES = 3
SPEED_SAMPLE_WINDOW_SEC = 3.5
SPEED_MAX_KMH = 220.0

STOP_LINE_BELOW_FRACTION = 0.06
STOP_LINE_SPAN_FRACTION = 0.30
STOP_LINE_KEEP_SEC = 2.0

CLIP_PRE_SECONDS = 5.0
CLIP_POST_SECONDS = 5.0
CLIP_MAX_WIDTH = 1280
CLIP_FPS = SAMPLE_RATE_FPS

VEHICLE_COLORS = {2: (255, 200, 0), 3: (0, 170, 255)}
PLATE_COLOR = (0, 220, 0)
HELMET_OK_COLOR = (0, 220, 0)
NO_HELMET_COLOR = (0, 0, 255)
SIGNAL_COLORS = {
    "red": (0, 0, 255),
    "green": (0, 210, 0),
    "yellow": (0, 215, 255),
    "unknown": (190, 190, 190),
}

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_analysis_models() -> tuple[Any, Any, Any]:
    """Load the vehicle, plate, and OCR models once and reuse them across jobs."""
    with _MODEL_CACHE_LOCK:
        configure_torch_threads()
        if "vehicle" not in _MODEL_CACHE:
            from ultralytics import YOLO
            _MODEL_CACHE["vehicle"] = YOLO(VEHICLE_MODEL_NAME)
        if "plate" not in _MODEL_CACHE:
            from ultralytics import YOLO
            _MODEL_CACHE["plate"] = YOLO(str(PLATE_MODEL_PATH))
        if "reader" not in _MODEL_CACHE:
            try:
                import easyocr
                _MODEL_CACHE["reader"] = easyocr.Reader(["en"], gpu=False, verbose=False)
            except Exception as exc:
                # Plate reading is an enhancement; speed and absence analysis still run.
                print(f"[Video Analysis] OCR unavailable, plate text will be skipped: {exc}", flush=True)
                _MODEL_CACHE["reader"] = None
        return _MODEL_CACHE["vehicle"], _MODEL_CACHE["plate"], _MODEL_CACHE["reader"]


def _load_helmet_model() -> Any | None:
    """Load the open-vocabulary helmet detector the first time a motorcycle appears."""
    with _MODEL_CACHE_LOCK:
        if "helmet" in _MODEL_CACHE:
            return _MODEL_CACHE["helmet"]
        _MODEL_CACHE["helmet"] = None
        if not HELMET_MODEL_PATH.is_file():
            print("[Video Analysis] Helmet weights missing; helmet screening is disabled.")
            return None
        try:
            configure_torch_threads()
            from ultralytics import YOLOWorld
            model = YOLOWorld(str(HELMET_MODEL_PATH))
            model.set_classes(list(HELMET_LABELS))
            _MODEL_CACHE["helmet"] = model
            print("[Video Analysis] Helmet screening ready.", flush=True)
        except Exception as exc:
            print(f"[Video Analysis] Helmet screening unavailable: {exc}", flush=True)
        return _MODEL_CACHE["helmet"]


def _format_video_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _plate_crop_candidates(frame: np.ndarray, bbox: list[int]) -> list[np.ndarray]:
    """Build upscaled, sharpened plate-crop variants so small night plates stay legible."""
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame.shape[:2]
    x1, x2 = max(0, x1), min(frame_width, x2)
    y1, y2 = max(0, y1), min(frame_height, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 12:
        return []
    height, width = crop.shape[:2]
    if width < 40:
        # Plate crops much smaller than this rarely survive OCR.
        return []
    scale = max(1.0, min(6.0, 320.0 / float(width)))
    size = (max(24, int(width * scale)), max(10, int(height * scale)))
    color = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.bilateralFilter(enhanced, 5, 60, 60)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    candidates = [sharpened, color]
    if float(enhanced.mean()) < 95.0:
        candidates.append(cv2.bitwise_not(enhanced))
    return candidates


def _plausible_plate(text: str) -> bool:
    if not 4 <= len(text) <= 12:
        return False
    if sum(character.isdigit() for character in text) < 2:
        return False
    return any(character.isalpha() for character in text) or len(text) >= 6


def _pick_plate_text(results: list[Any]) -> tuple[str, float]:
    """Choose the most plausible string, joining split reads such as 'MH12' + 'AB1234'."""
    items: list[tuple[float, str, float]] = []
    for item in results or []:
        if len(item) < 3:
            continue
        coordinates, text, confidence = item[0], item[1], float(item[2])
        normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        if not normalized:
            continue
        try:
            xs = [float(point[0]) for point in coordinates]
            center_x = sum(xs) / len(xs)
        except (TypeError, ValueError, IndexError):
            center_x = 0.0
        items.append((center_x, normalized, confidence))
    if not items:
        return "", 0.0
    items.sort(key=lambda entry: entry[0])
    candidates = [(text, confidence) for _, text, confidence in items]
    if len(items) > 1:
        joined = "".join(text for _, text, _ in items)
        joined_confidence = min(confidence for _, _, confidence in items)
        candidates.append((joined, joined_confidence))
    best_text, best_confidence = "", 0.0
    for text, confidence in candidates:
        if confidence < 0.20 or not _plausible_plate(text):
            continue
        if confidence > best_confidence:
            best_text, best_confidence = text, confidence
    return best_text, best_confidence


def _read_plate_text(reader: Any, frame: np.ndarray, bbox: list[int]) -> tuple[str, float]:
    """Run OCR over a few enhanced crop variants and keep the strongest plausible read."""
    if reader is None:
        return "", 0.0
    candidates = _plate_crop_candidates(frame, bbox)
    if not candidates:
        return "", 0.0
    attempts: list[tuple[np.ndarray, str | None]] = [(candidates[0], PLATE_ALLOWLIST)]
    if len(candidates) > 1:
        attempts.append((candidates[1], PLATE_ALLOWLIST))
    attempts.append((candidates[0], None))
    best_text, best_confidence = "", 0.0
    for prepared, allowlist in attempts:
        height, width = prepared.shape[:2]
        try:
            # The detector already localized the plate, so recognize-only skips
            # EasyOCR's own detection pass: ~4x faster and more accurate here.
            results = reader.recognize(
                prepared,
                horizontal_list=[[0, width, 0, height]],
                free_list=[],
                decoder="greedy",
                detail=1,
                allowlist=allowlist,
            )
        except TypeError:
            try:
                results = reader.readtext(prepared, detail=1, paragraph=False)
            except Exception:
                continue
        except Exception:
            continue
        text, confidence = _pick_plate_text(results)
        if confidence > best_confidence:
            best_text, best_confidence = text, confidence
        if best_confidence >= 0.55:
            break
    return best_text, best_confidence


def _classify_signals(
    frame: np.ndarray,
    boxes: list[tuple[list[int], float]],
) -> list[dict[str, Any]]:
    """Classify each detected traffic-light head as red, yellow, green, or unknown."""
    signals: list[dict[str, Any]] = []
    frame_height, frame_width = frame.shape[:2]
    for bbox, detection_confidence in boxes:
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 - x1 < 6 or y2 - y1 < 12:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        bright = value >= 110
        bright_count = int(bright.sum())
        if bright_count < 10:
            continue
        red_mask = ((hue <= 12) | (hue >= 168)) & (saturation >= 120) & bright
        yellow_mask = (hue >= 15) & (hue <= 35) & (saturation >= 120) & bright
        green_mask = (hue >= 40) & (hue <= 95) & (saturation >= 90) & bright
        fractions = {
            "red": int(red_mask.sum()) / bright_count,
            "yellow": int(yellow_mask.sum()) / bright_count,
            "green": int(green_mask.sum()) / bright_count,
        }
        state = max(fractions, key=fractions.get)
        if fractions[state] < RED_LIGHT_MIN_FRACTION:
            state = "unknown"
        confidence = 0.0
        if state != "unknown":
            confidence = min(0.99, fractions[state] * 3.0)
        signals.append({
            "bbox": [x1, y1, x2, y2],
            "state": state,
            "confidence": round(confidence, 2),
            "detection_confidence": round(detection_confidence, 2),
        })
    return signals


def _scan_helmet(
    model: Any,
    frame: np.ndarray,
    motorcycle_bbox: list[int],
) -> tuple[str, float, list[int] | None, list[int] | None]:
    """Look for a helmet/head around one motorcycle and return the rider verdict."""
    x1, y1, x2, y2 = motorcycle_bbox
    width, height = x2 - x1, y2 - y1
    if width < 40 or height < 40:
        return "unknown", 0.0, None, None
    frame_height, frame_width = frame.shape[:2]
    rx1 = max(0, int(x1 - 0.15 * width))
    rx2 = min(frame_width, int(x2 + 0.15 * width))
    ry1 = max(0, int(y1 - 0.55 * height))
    ry2 = min(frame_height, int(y2 + 0.05 * height))
    if rx2 - rx1 < 32 or ry2 - ry1 < 32:
        return "unknown", 0.0, None, None
    crop = frame[ry1:ry2, rx1:rx2]
    try:
        results = model.predict(
            crop,
            imgsz=512,
            conf=HELMET_CONFIDENCE,
            device=DETECTION_DEVICE,
            max_det=10,
            verbose=False,
        )
    except Exception:
        return "unknown", 0.0, None, None

    heads: list[dict[str, Any]] = []
    helmets: list[dict[str, Any]] = []
    names = model.names if isinstance(model.names, dict) else {}
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = str(names.get(class_id, "")).lower()
            confidence = float(box.conf[0].item())
            bx1, by1, bx2, by2 = (int(value) for value in box.xyxy[0].tolist())
            mapped = [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1]
            if name == "helmet":
                helmets.append({"bbox": mapped, "conf": confidence})
            elif name == "head":
                heads.append({"bbox": mapped, "conf": confidence})

    rider_top = y1 - 0.45 * height
    rider_bottom = y1 + 0.60 * height

    def in_rider_zone(item: dict[str, Any]) -> bool:
        bx1, by1, bx2, by2 = item["bbox"]
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2
        return (
            x1 - 0.2 * width <= center_x <= x2 + 0.2 * width
            and rider_top <= center_y <= rider_bottom
        )

    helmets = [item for item in helmets if in_rider_zone(item)]
    heads = [item for item in heads if in_rider_zone(item)]
    if helmets:
        best = max(helmets, key=lambda item: item["conf"])
        head_box = max(heads, key=lambda item: item["conf"])["bbox"] if heads else None
        return "helmet", best["conf"], best["bbox"], head_box
    if heads:
        best = max(heads, key=lambda item: item["conf"])
        return "no_helmet", best["conf"], None, best["bbox"]
    return "unknown", 0.0, None, None


def _draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - 7)
    left = max(0, min(x, image.shape[1] - text_width - 10))
    cv2.rectangle(
        image,
        (left, top),
        (left + text_width + 8, top + text_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (left + 4, top + text_height + 2),
        font,
        scale,
        (12, 16, 26),
        thickness,
        cv2.LINE_AA,
    )


def _annotate_frame(
    frame: np.ndarray,
    vehicles: list[dict[str, Any]],
    plates: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    helmets: list[dict[str, Any]],
    wall_clock: str = "",
    stop_line: dict[str, float] | None = None,
) -> np.ndarray:
    """Draw every detection as a labeled rectangle before clips and snapshots are encoded."""
    image = frame.copy()
    for vehicle in vehicles:
        x1, y1, x2, y2 = vehicle["bbox"]
        color = VEHICLE_COLORS.get(int(vehicle["class_id"]), (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{VEHICLE_CLASSES.get(int(vehicle['class_id']), 'VEHICLE').upper()} {int(vehicle['conf'] * 100)}%"
        _draw_label(image, label, x1, y1, color)
    for signal in signals:
        x1, y1, x2, y2 = signal["bbox"]
        state = str(signal["state"])
        color = SIGNAL_COLORS.get(state, SIGNAL_COLORS["unknown"])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        confidence = int(float(signal.get("confidence", 0.0)) * 100)
        _draw_label(image, f"SIGNAL {state.upper()} {confidence}%", x1, y1, color)
    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), PLATE_COLOR, 2)
        text = str(plate.get("text") or "")
        confidence = int(float(plate.get("conf", 0.0)) * 100)
        if text:
            label = f"{text} {confidence}%" if confidence else text
        else:
            label = f"PLATE {int(float(plate.get('det_conf', 0.0)) * 100)}%"
        _draw_label(image, label, x1, y1, PLATE_COLOR)
    for helmet in helmets:
        bbox = helmet.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        is_safe = helmet.get("state") == "helmet"
        color = HELMET_OK_COLOR if is_safe else NO_HELMET_COLOR
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        confidence = int(float(helmet.get("conf", 0.0)) * 100)
        text = "HELMET OK" if is_safe else "NO HELMET"
        _draw_label(image, f"{text} {confidence}%", x1, y1, color)
    if stop_line is not None:
        line_y = int(round(stop_line["y"]))
        cv2.line(
            image,
            (int(stop_line["x1"]), line_y),
            (int(stop_line["x2"]), line_y),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        _draw_label(image, "AUTO STOP LINE", int(stop_line["x1"]) + 4, max(2, line_y - 24), (0, 0, 255))
    if wall_clock:
        cv2.putText(
            image,
            wall_clock,
            (12, image.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def _encode_snapshot(image: np.ndarray, banner: str = "") -> str:
    if banner:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 44), (8, 13, 26), -1)
        image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
        cv2.putText(
            image,
            banner[:110],
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _open_clip_writer(path: Path, width: int, height: int) -> cv2.VideoWriter | None:
    attempts: list[tuple[int, str]] = []
    if sys.platform.startswith("win"):
        attempts.append((cv2.CAP_MSMF, "H264"))
    attempts.extend([(cv2.CAP_FFMPEG, "avc1"), (0, "avc1"), (0, "mp4v")])
    for api, codec in attempts:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            if api:
                writer = cv2.VideoWriter(str(path), api, fourcc, CLIP_FPS, (width, height))
            else:
                writer = cv2.VideoWriter(str(path), fourcc, CLIP_FPS, (width, height))
            if writer.isOpened():
                return writer
            writer.release()
        except Exception as exc:
            print(f"[Video Analysis] {codec} clip writer failed: {exc}", flush=True)
    return None


class _IouTracker:
    """Assign stable vehicle IDs across sampled frames without extra dependencies."""

    def __init__(self, iou_threshold: float = 0.3, max_age_seconds: float = 2.0) -> None:
        self._tracks: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._iou_threshold = iou_threshold
        self._max_age_seconds = max_age_seconds

    @staticmethod
    def _iou(first: list[int], second: list[int]) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
        if not intersection:
            return 0.0
        union = (
            max(0, ax2 - ax1) * max(0, ay2 - ay1)
            + max(0, bx2 - bx1) * max(0, by2 - by1)
            - intersection
        )
        return intersection / union if union else 0.0

    def update(self, timestamp: float, detections: list[tuple[int, list[int]]]) -> list[int]:
        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state["seen"] > self._max_age_seconds
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

        assigned: list[int] = []
        claimed: set[int] = set()
        for class_id, bbox in detections:
            best_id: int | None = None
            best_iou = self._iou_threshold
            for track_id, state in self._tracks.items():
                if track_id in claimed or state["class_id"] != class_id:
                    continue
                iou = self._iou(bbox, state["bbox"])
                if iou >= best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is None:
                # Fast pans at night break IoU; fall back to nearest centre of the
                # same class so one vehicle keeps one ID across the whole scene.
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                best_distance = float("inf")
                for track_id, state in self._tracks.items():
                    if track_id in claimed or state["class_id"] != class_id:
                        continue
                    previous = state["bbox"]
                    px, py = (previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2
                    reach_x = max(36.0, 0.9 * (previous[2] - previous[0]))
                    reach_y = max(24.0, 0.9 * (previous[3] - previous[1]))
                    if abs(cx - px) > reach_x or abs(cy - py) > reach_y:
                        continue
                    distance = math.hypot(cx - px, cy - py)
                    if distance < best_distance:
                        best_id, best_distance = track_id, distance
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            self._tracks[best_id] = {"bbox": bbox, "class_id": class_id, "seen": timestamp}
            claimed.add(best_id)
            assigned.append(best_id)
        return assigned


def validate_analysis_settings(settings: dict[str, Any]) -> dict[str, Any]:
    try:
        speed_limit_kmh = float(settings.get("speed_limit_kmh", 60))
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter a valid numeric speed limit value.") from exc
    if not math.isfinite(speed_limit_kmh) or not 1 <= speed_limit_kmh <= 300:
        raise ValueError("The speed limit must be between 1 and 300 km/h.")
    return {"speed_limit_kmh": speed_limit_kmh}


class VideoEventTracker:
    """Track vehicles and emit plate, helmet, red-light, and speed findings."""

    def __init__(self, speed_limit_kmh: float) -> None:
        self.speed_limit_kmh = speed_limit_kmh
        self.tracks: dict[int, dict[str, Any]] = {}
        self.person_hits = 0
        self.person_present = False
        self.person_absence_started: float | None = None
        self.person_absence_reported = False
        self.events: list[dict[str, Any]] = []
        self.pending_events: list[dict[str, Any]] = []
        self._last_red_light_event = -1e9
        self._stop_line: dict[str, float] | None = None
        self._stop_approach_side: int | None = None
        self._red_since: float | None = None
        self._red_last_seen: float = -1e9
        self._red_confidence = 0.0
        self.plate_tracks: list[dict[str, Any]] = []
        self._plate_text_events: dict[str, dict[str, Any]] = {}

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.pending_events = self.pending_events, []
        return events

    def _emit(self, event: dict[str, Any]) -> dict[str, Any]:
        self.events.append(event)
        self.pending_events.append(event)
        return event

    @staticmethod
    def _best_plate_text(state: dict[str, Any]) -> str:
        votes = state.get("votes") or {}
        if not votes:
            return ""
        return max(votes, key=lambda text: (votes[text]["count"], votes[text]["score"]))

    @property
    def stop_line(self) -> dict[str, float] | None:
        return self._stop_line

    @property
    def red_steady_seconds(self) -> float:
        if self._red_since is None:
            return 0.0
        return max(0.0, self._red_last_seen - self._red_since)

    def update_traffic_signals(
        self,
        timestamp: float,
        signals: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Maintain the automatic stop line and a flicker-tolerant red phase."""
        red_lights = [signal for signal in signals if signal.get("state") == "red"]
        if red_lights:
            if self._red_since is None:
                self._red_since = timestamp
            self._red_last_seen = timestamp
            self._red_confidence = max(
                float(signal.get("confidence", 0.0)) for signal in red_lights
            )
        elif signals and self._red_since is not None:
            self._red_since = None
            self._red_last_seen = -1e9
        elif self._red_since is not None and timestamp - self._red_last_seen > RED_STATE_GAP_TOLERANCE_SEC:
            self._red_since = None
            self._red_last_seen = -1e9

        if not red_lights or timestamp - self._red_since < RED_LIGHT_MIN_STEADY_SEC:
            if self._stop_line is not None and timestamp - self._red_last_seen > STOP_LINE_KEEP_SEC:
                self._stop_line = None
                self._stop_approach_side = None
            return

        best = max(red_lights, key=lambda signal: float(signal.get("confidence", 0.0)))
        x1, y1, x2, y2 = best["bbox"]
        line_y = min(float(frame_height - 1), y2 + STOP_LINE_BELOW_FRACTION * frame_height)
        line_x1 = max(0.0, x1 - STOP_LINE_SPAN_FRACTION * frame_width)
        line_x2 = min(float(frame_width), x2 + STOP_LINE_SPAN_FRACTION * frame_width)
        if self._stop_line is None:
            self._stop_line = {"y": line_y, "x1": line_x1, "x2": line_x2}
            for state in self.tracks.values():
                state["stop_side"] = 1 if state["vehicle_bbox"][3] < line_y else -1
            sides = [
                state.get("stop_side")
                for state in self.tracks.values()
                if state.get("stop_side") is not None
            ]
            self._stop_approach_side = (
                1 if sides.count(1) >= sides.count(-1) else -1
            ) if sides else None
        else:
            self._stop_line["y"] = 0.7 * self._stop_line["y"] + 0.3 * line_y
            self._stop_line["x1"] = 0.7 * self._stop_line["x1"] + 0.3 * line_x1
            self._stop_line["x2"] = 0.7 * self._stop_line["x2"] + 0.3 * line_x2

    def nearest_plate(
        self,
        timestamp: float,
        center: tuple[float, float],
        frame_width: int,
        max_age: float = 2.0,
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_distance = float("inf")
        for track in self.plate_tracks:
            if not track.get("text"):
                continue
            if timestamp - track["last_seen"] > max_age:
                continue
            bx1, _, bx2, _ = track["bbox"]
            distance = abs(center[0] - (bx1 + bx2) / 2)
            if distance > 0.35 * frame_width:
                continue
            if distance < best_distance:
                best_distance = distance
                best = {
                    "text": track["text"],
                    "confidence": float(track.get("conf", 0.0)),
                    "bbox": track["bbox"],
                }
        return best

    def _match_plate_track(self, bbox: list[int], used: set[int]) -> int | None:
        best_index: int | None = None
        best_iou = 0.3
        for index, track in enumerate(self.plate_tracks):
            if index in used:
                continue
            iou = _IouTracker._iou(bbox, track["bbox"])
            if iou >= best_iou:
                best_index, best_iou = index, iou
        if best_index is not None:
            return best_index
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        best_distance = float("inf")
        for index, track in enumerate(self.plate_tracks):
            if index in used:
                continue
            px, py = (track["bbox"][0] + track["bbox"][2]) / 2, (track["bbox"][1] + track["bbox"][3]) / 2
            if abs(cx - px) > max(30.0, bbox[2] - bbox[0]) or abs(cy - py) > max(16.0, bbox[3] - bbox[1]):
                continue
            distance = math.hypot(cx - px, cy - py)
            if distance < best_distance:
                best_index, best_distance = index, distance
        return best_index

    @staticmethod
    def _vehicle_type_for_plate(
        plate_bbox: list[int],
        vehicles: list[dict[str, Any]],
    ) -> str:
        px = (plate_bbox[0] + plate_bbox[2]) / 2
        py = (plate_bbox[1] + plate_bbox[3]) / 2
        best_type, best_distance = "", float("inf")
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            margin_x, margin_y = 0.25 * (x2 - x1), 0.35 * (y2 - y1)
            if not (
                x1 - margin_x <= px <= x2 + margin_x
                and y1 - margin_y <= py <= y2 + margin_y
            ):
                continue
            distance = abs(px - (x1 + x2) / 2)
            if distance < best_distance:
                best_distance = distance
                best_type = VEHICLE_CLASSES.get(int(vehicle["class_id"]), "Vehicle")
        return best_type or "Vehicle"

    def update_plates(
        self,
        timestamp: float,
        detections: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        frame: np.ndarray,
        reader: Any,
        now_str: str = "",
    ) -> None:
        """Track every plate separately from vehicles so IDs survive fast pans and night noise."""
        used: set[int] = set()
        for detection in detections:
            bbox = detection["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width < 14 or height < 6:
                continue
            aspect = width / float(height)
            # Real plates are wide strips (>=1.8:1) mounted on a detected vehicle;
            # this drops dash screens, lamps, and other plate-like clutter.
            if not 1.8 <= aspect <= 8.0:
                continue
            vehicle_type = self._vehicle_type_for_plate(bbox, vehicles)
            if vehicle_type == "Vehicle":
                continue
            index = self._match_plate_track(bbox, used)
            if index is None:
                self.plate_tracks.append({
                    "bbox": bbox,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "sightings": 0,
                    "votes": {},
                    "last_ocr": -1e9,
                    "reported": False,
                    "text": "",
                    "conf": 0.0,
                    "det_conf": 0.0,
                    "vehicle_type": "",
                })
                index = len(self.plate_tracks) - 1
            track = self.plate_tracks[index]
            used.add(index)
            track["bbox"] = bbox
            track["last_seen"] = timestamp
            track["sightings"] = int(track["sightings"]) + 1
            track["det_conf"] = max(float(track["det_conf"]), float(detection.get("conf", 0.0)))
            track["vehicle_type"] = vehicle_type

        self.plate_tracks = [
            track for track in self.plate_tracks if timestamp - track["last_seen"] <= 4.0
        ]
        if reader is not None:
            # OCR is the costliest step: read at most the two largest readable
            # plates per frame, and stop once a plate has been reported.
            candidates = [
                track
                for track in self.plate_tracks
                if track["last_seen"] == timestamp
                and not track["reported"]
                and track["bbox"][2] - track["bbox"][0] >= 44
                and timestamp - float(track["last_ocr"]) >= PLATE_OCR_INTERVAL_SEC
            ]
            candidates.sort(
                key=lambda track: (track["bbox"][2] - track["bbox"][0]) * (track["bbox"][3] - track["bbox"][1]),
                reverse=True,
            )
            for track in candidates[:2]:
                track["last_ocr"] = timestamp
                text, confidence = _read_plate_text(reader, frame, track["bbox"])
                if not text:
                    continue
                vote = track["votes"].setdefault(text, {"score": 0.0, "count": 0, "conf": 0.0})
                vote["score"] += confidence
                vote["count"] += 1
                vote["conf"] = max(vote["conf"], confidence)

        self._emit_plate_events(timestamp, now_str)

    def _emit_plate_events(self, timestamp: float, now_str: str) -> None:
        for track in self.plate_tracks:
            text = track.get("text") or self._best_plate_text(track)
            vote = (track.get("votes") or {}).get(text, {}) if text else {}
            best_confidence = float(vote.get("conf", 0.0))
            width = track["bbox"][2] - track["bbox"][0]
            if not track.get("reported"):
                stable = bool(vote) and (
                    vote["count"] >= PLATE_STABLE_VOTES or best_confidence >= PLATE_READ_CONFIDENCE
                )
                long_enough = track["sightings"] >= PLATE_UNREADABLE_SIGHTINGS
                if not stable and not long_enough:
                    continue
                if not text and (width < 90 or width / float(track["bbox"][3] - track["bbox"][1]) < 1.8):
                    continue
                track["reported"] = True
                track["text"] = text or "Unreadable"
                track["conf"] = round(best_confidence, 2) if text else 0.0
                if text and text in self._plate_text_events:
                    continue
                vehicle_type = track.get("vehicle_type") or "Vehicle"
                plate_text = track["text"]
                event = self._emit({
                    "type": "plate_read",
                    "label": (
                        f"{vehicle_type} number plate: {plate_text} at "
                        f"{_format_video_time(timestamp)}"
                    ),
                    "timestamp_seconds": round(timestamp, 2),
                    "datetime": now_str,
                    "plate_text": plate_text,
                    "confidence": track["conf"],
                    "detection_confidence": round(float(track.get("det_conf", 0.0)), 2),
                    "plate_bbox": list(track["bbox"]),
                    "vehicle_type": vehicle_type,
                    "details": (
                        f"Plate read at video time {_format_video_time(timestamp)} "
                        f"(wall clock {now_str}). OCR confidence {track['conf'] * 100:.0f}%. "
                        "AI-generated text; verify against the plate image."
                    ),
                })
                track["event"] = event
                track["reported_text"] = plate_text
                if text:
                    self._plate_text_events[text] = event
            elif text and best_confidence > float(track.get("conf", 0.0)) + 0.05:
                # A clearer read supersedes the first one for this plate.
                track["text"] = text
                track["conf"] = round(best_confidence, 2)
                event = track.get("event")
                if event is not None:
                    event["plate_text"] = text
                    event["confidence"] = track["conf"]
                    event["label"] = (
                        f"{event.get('vehicle_type', 'Vehicle')} number plate: {text} at "
                        f"{_format_video_time(float(event['timestamp_seconds']))}"
                    )
                    event["details"] = (
                        f"Plate read at video time "
                        f"{_format_video_time(float(event['timestamp_seconds']))} "
                        f"(wall clock {event.get('datetime', now_str)}). "
                        f"OCR confidence {track['conf'] * 100:.0f}%. "
                        "AI-generated text; verify against the plate image."
                    )
                    track["reported_text"] = text
                    self._plate_text_events.setdefault(text, event)

    def plate_labels(self, timestamp: float, max_age: float = 0.8) -> list[dict[str, Any]]:
        """Boxes for every plate currently on screen, including plates still being read."""
        labels: list[dict[str, Any]] = []
        for track in self.plate_tracks:
            if timestamp - track["last_seen"] > max_age:
                continue
            text = track.get("text") or self._best_plate_text(track)
            confidence = float(track.get("conf", 0.0))
            if not confidence and text:
                confidence = float((track.get("votes") or {}).get(text, {}).get("conf", 0.0))
            labels.append({
                "bbox": list(track["bbox"]),
                "text": text,
                "conf": confidence,
                "det_conf": float(track.get("det_conf", 0.0)),
            })
        return labels

    def update_persons(self, timestamp: float, person_detected: bool) -> None:
        if person_detected:
            self.person_hits += 1
            self.person_absence_started = None
            self.person_absence_reported = False
            if self.person_hits >= 2:
                self.person_present = True
            return
        self.person_hits = 0
        if not self.person_present:
            return
        if self.person_absence_started is None:
            self.person_absence_started = timestamp
        elif (
            not self.person_absence_reported
            and timestamp - self.person_absence_started >= PERSON_ABSENCE_SECONDS
        ):
            self.person_absence_reported = True
            self._emit({
                "type": "person_absence",
                "label": "Person absent from the monitored view",
                "timestamp_seconds": round(timestamp, 2),
                "details": "No person was detected for at least 10 seconds after a person had been detected.",
            })

    def update_helmet(
        self,
        timestamp: float,
        track_id: int,
        state_name: str,
        confidence: float,
        helmet_bbox: list[int] | None,
        head_bbox: list[int] | None,
        now_str: str = "",
    ) -> dict[str, Any] | None:
        state = self.tracks.get(track_id)
        if state is None:
            return None
        state["helmet_state"] = state_name
        state["helmet_conf"] = confidence
        state["helmet_bbox"] = helmet_bbox
        state["head_bbox"] = head_bbox
        if state_name != "no_helmet":
            state["helmet_checks"] = 0
            return None
        state["helmet_checks"] = int(state.get("helmet_checks", 0)) + 1
        if state["helmet_checks"] < HELMET_CONFIRMATIONS or state.get("helmetless_reported"):
            return None
        state["helmetless_reported"] = True
        plate_text = str(state.get("plate_text") or "Unreadable")
        return self._emit({
            "type": "helmetless",
            "label": f"Motorcycle rider without a helmet [{plate_text}]",
            "violation": True,
            "request_clip": True,
            "clip_title": "Helmet Violation",
            "timestamp_seconds": round(timestamp, 2),
            "datetime": now_str,
            "plate_text": plate_text,
            "confidence": round(confidence, 2),
            "vehicle_type": "Motorcycle",
            "details": (
                f"Video time {_format_video_time(timestamp)}. A rider's head was visible without a helmet "
                f"on the detected motorcycle (head confidence {confidence * 100:.0f}%). Plate: {plate_text}. "
                "AI screening only; verify the footage."
            ),
        })

    def update_vehicle(
        self,
        timestamp: float,
        track_id: int,
        class_id: int,
        bbox: list[int],
        frame_width: int,
        now_str: str = "",
    ) -> dict[str, Any] | None:
        state = self.tracks.setdefault(track_id, {
            "previous_center": None,
            "previous_bottom": None,
            "last_timestamp": timestamp,
            "plate_text": "",
            "plate_confidence": 0.0,
            "vehicle_bbox": bbox,
            "class_id": class_id,
            "speed_samples": [],
            "speed_event_created": False,
            "red_light_reported": False,
            "stop_side": None,
            "stop_exempt": False,
            "helmet_state": "unknown",
            "helmet_conf": 0.0,
            "helmet_bbox": None,
            "head_bbox": None,
            "helmet_checks": 0,
            "helmetless_reported": False,
        })
        state["vehicle_bbox"] = bbox
        state["class_id"] = class_id
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        bottom = float(y2)
        vehicle_type = VEHICLE_CLASSES.get(class_id, "Vehicle")

        nearest_plate = self.nearest_plate(timestamp, center, frame_width)
        if nearest_plate:
            state["plate_text"] = nearest_plate["text"]
            state["plate_confidence"] = nearest_plate["confidence"]

        previous = state["previous_center"]
        previous_bottom = state.get("previous_bottom")
        event: dict[str, Any] | None = None

        if previous is not None and previous_bottom is not None and timestamp > state["last_timestamp"]:
            delta_time = timestamp - state["last_timestamp"]

            stop_line = self._stop_line
            if (
                stop_line is not None
                and self.red_steady_seconds >= RED_LIGHT_MIN_STEADY_SEC
                and not state["red_light_reported"]
                and not state.get("stop_exempt")
                and timestamp - self._last_red_light_event >= RED_LIGHT_EVENT_GAP_SEC
            ):
                side_now = 1 if bottom < stop_line["y"] else -1
                if state.get("stop_side") is None:
                    state["stop_side"] = side_now
                    if self._stop_approach_side is not None and side_now != self._stop_approach_side:
                        state["stop_exempt"] = True
                side_previous = state.get("stop_side")
                in_vicinity = (
                    stop_line["x1"] - 0.10 * frame_width
                    <= center[0]
                    <= stop_line["x2"] + 0.10 * frame_width
                )
                if (
                    side_previous is not None
                    and side_previous != side_now
                    and in_vicinity
                    and abs(bottom - previous_bottom) >= 3.0
                ):
                    if self._stop_approach_side is None:
                        self._stop_approach_side = side_previous
                    if side_previous == self._stop_approach_side:
                        state["red_light_reported"] = True
                        self._last_red_light_event = timestamp
                        plate_text = state.get("plate_text") or "Unreadable"
                        event = self._emit({
                            "type": "red_light",
                            "label": f"Red-light crossing [{plate_text}]",
                            "violation": True,
                            "request_clip": True,
                            "clip_title": "Red Light Violation",
                            "timestamp_seconds": round(timestamp, 2),
                            "datetime": now_str,
                            "plate_text": plate_text,
                            "confidence": state.get("plate_confidence", 0.0),
                            "signal_confidence": round(float(self._red_confidence), 2),
                            "vehicle_type": vehicle_type,
                            "details": (
                                f"{vehicle_type} crossed the auto-detected stop line while the traffic "
                                f"signal was red at video time {_format_video_time(timestamp)}. "
                                f"Signal confidence {float(self._red_confidence) * 100:.0f}%. "
                                f"Plate: {plate_text}. AI screening only."
                            ),
                        })
                state["stop_side"] = side_now

            if delta_time > 0 and not state["speed_event_created"]:
                dx = abs(center[0] - previous[0])
                dy = abs(center[1] - previous[1])
                length_px = float(x2 - x1) if dx >= dy else float(y2 - y1)
                displacement = math.hypot(dx, dy)
                if length_px >= 18.0 and displacement >= max(4.0, 0.10 * length_px):
                    meters_per_pixel = REFERENCE_VEHICLE_LENGTH_M.get(class_id, 4.5) / length_px
                    sample_kmh = displacement * meters_per_pixel / delta_time * 3.6
                    if sample_kmh <= SPEED_MAX_KMH:
                        state["speed_samples"].append((timestamp, sample_kmh))
                state["speed_samples"] = [
                    sample for sample in state["speed_samples"]
                    if timestamp - sample[0] <= SPEED_SAMPLE_WINDOW_SEC
                ]
                samples = state["speed_samples"]
                if len(samples) >= SPEED_MIN_SAMPLES:
                    ordered = sorted(sample[1] for sample in samples)
                    speed_kmh = ordered[len(ordered) // 2]
                    plate_text = state.get("plate_text") or "Unreadable"
                    overspeed = speed_kmh > self.speed_limit_kmh
                    event = self._emit({
                        "type": "vehicle_speed",
                        "vehicle_type": vehicle_type,
                        "label": (
                            f"{vehicle_type} detected [{plate_text}] at {now_str} "
                            f"(speed: {speed_kmh:.1f} km/h)"
                        ),
                        "timestamp_seconds": round(timestamp, 2),
                        "datetime": now_str,
                        "speed_kmh": round(speed_kmh, 1),
                        "speed_limit_kmh": self.speed_limit_kmh,
                        "overspeed": overspeed,
                        "violation": overspeed,
                        "plate_text": plate_text,
                        "confidence": state.get("plate_confidence", 0.0),
                        "request_clip": overspeed,
                        "clip_title": "Overspeed",
                        "details": (
                            f"Plate: {plate_text} captured at video time {_format_video_time(timestamp)}. "
                            f"Speed estimated from the {vehicle_type.lower()} on-screen size "
                            f"(median of {len(samples)} samples: {speed_kmh:.1f} km/h). "
                            "OCR text is AI-generated and must be visually verified."
                        ),
                    })
                    state["speed_event_created"] = True

        state["previous_center"] = center
        state["previous_bottom"] = bottom
        state["last_timestamp"] = timestamp
        return event


class UploadedVideoAnalyzer:
    """Runs one uploaded-video analysis at a time and removes its temporary file."""

    def __init__(
        self,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_overspeed: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None
        self._on_event = on_event or on_overspeed
        self._pending_clips: list[dict[str, Any]] = []
        self._preroll: deque[tuple[float, bytes]] = deque(
            maxlen=int((CLIP_PRE_SECONDS + 3.0) * SAMPLE_RATE_FPS)
        )

    @property
    def is_active(self) -> bool:
        with self._lock:
            return bool(self._job and self._job["status"] in {"queued", "running"})

    def start(self, video_path: str, filename: str, settings: dict[str, Any]) -> str:
        validated_settings = validate_analysis_settings(settings)
        with self._lock:
            if self._job and self._job["status"] in {"queued", "running"}:
                raise RuntimeError("An uploaded video is already being analyzed.")
            job_id = uuid.uuid4().hex
            self._job = {
                "id": job_id,
                "filename": filename,
                "status": "queued",
                "progress": 0,
                "processed_seconds": 0.0,
                "duration_seconds": 0.0,
                "events": [],
                "error": "",
                "speed_limit_kmh": validated_settings["speed_limit_kmh"],
                "helmet_screening": HELMET_MODEL_PATH.is_file(),
            }
        worker = threading.Thread(
            target=self._analyze,
            args=(job_id, video_path, filename, validated_settings),
            name="uploaded-video-analysis",
            daemon=True,
        )
        worker.start()
        return job_id

    def status(self, job_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if not self._job or (job_id and self._job["id"] != job_id):
                return {"status": "idle", "progress": 0, "events": []}
            return dict(self._job)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job and self._job["id"] == job_id:
                self._job.update(values)

    def _dispatch(self, filename: str, event: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(filename, event)
        except Exception as exc:
            print(f"[Video Analysis] Event dispatch failed: {exc}", flush=True)

    def _start_clip(
        self,
        filename: str,
        event: dict[str, Any],
        timestamp: float,
        width: int,
        height: int,
    ) -> None:
        INCIDENTS_DIR.mkdir(exist_ok=True)
        title = str(event.get("clip_title") or "Violation")
        slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "Violation"
        base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_uploaded_{slug}"
        path = INCIDENTS_DIR / f"{base}.mp4"
        suffix = 2
        while path.exists():
            path = INCIDENTS_DIR / f"{base}_{suffix}.mp4"
            suffix += 1
        writer = _open_clip_writer(path, width, height)
        if writer is None:
            return
        capture = {
            "event": event,
            "writer": writer,
            "path": path,
            "end": timestamp + CLIP_POST_SECONDS,
            "frames": 0,
            "filename": filename,
        }
        for frame_timestamp, encoded in list(self._preroll):
            if frame_timestamp < timestamp - CLIP_PRE_SECONDS:
                continue
            frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and frame.shape[1] == width and frame.shape[0] == height:
                writer.write(frame)
                capture["frames"] += 1
        self._pending_clips.append(capture)

    def _finalize_clip(self, capture: dict[str, Any]) -> None:
        writer = capture["writer"]
        path: Path = capture["path"]
        try:
            writer.release()
        except Exception:
            pass
        valid = int(capture.get("frames", 0)) >= 4 and path.is_file() and path.stat().st_size > 2000
        if valid:
            check = cv2.VideoCapture(str(path))
            valid = check.isOpened()
            check.release()
        event = capture["event"]
        if valid:
            event["clip_file"] = path.name
            event["clip_sent"] = False
        else:
            path.unlink(missing_ok=True)
        self._dispatch(capture["filename"], event)

    def _buffer_frame(self, timestamp: float, frame: np.ndarray) -> None:
        for capture in self._pending_clips:
            try:
                capture["writer"].write(frame)
                capture["frames"] += 1
            except Exception:
                pass
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            self._preroll.append((timestamp, encoded.tobytes()))

    @staticmethod
    def _prepare_output_frame(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width > CLIP_MAX_WIDTH:
            scale = CLIP_MAX_WIDTH / float(width)
            frame = cv2.resize(
                frame,
                (CLIP_MAX_WIDTH, max(2, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        if frame.shape[1] % 2:
            frame = frame[:, : frame.shape[1] - 1]
        if frame.shape[0] % 2:
            frame = frame[: frame.shape[0] - 1]
        return frame

    def _analyze(
        self,
        job_id: str,
        video_path: str,
        filename: str,
        settings: dict[str, Any],
    ) -> None:
        capture: cv2.VideoCapture | None = None
        self._preroll.clear()
        self._pending_clips.clear()
        try:
            self._update(job_id, status="running")
            if not PLATE_MODEL_PATH.is_file():
                raise FileNotFoundError(f"License-plate detector weights are missing: {PLATE_MODEL_PATH}")
            capture = cv2.VideoCapture(video_path)
            if not capture.isOpened():
                raise ValueError("The uploaded file could not be opened as a video.")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                fps = 25.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if total_frames > 0 else 0.0
            stride = max(1, round(fps / SAMPLE_RATE_FPS))
            vehicle_model, plate_model, reader = _load_analysis_models()
            tracker = VideoEventTracker(settings["speed_limit_kmh"])
            iou_tracker = _IouTracker()
            helmet_model: Any | None = None
            helmet_scans: dict[int, float] = {}
            helmet_cache: dict[int, dict[str, Any]] = {}
            frame_index = 0
            processed_seconds = 0.0

            while True:
                if frame_index % stride != 0:
                    # Decode only sampled frames; grabbing the rest saves most of
                    # the CPU cost of scanning a long upload.
                    if not capture.grab():
                        break
                    frame_index += 1
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                timestamp = frame_index / fps
                wall_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                vehicle_results = vehicle_model.predict(
                    frame,
                    classes=[PERSON_CLASS_ID, *sorted(VEHICLE_CLASSES), TRAFFIC_LIGHT_CLASS_ID],
                    conf=VEHICLE_CONFIDENCE,
                    imgsz=640,
                    device=DETECTION_DEVICE,
                    verbose=False,
                )
                plate_results = plate_model.predict(
                    frame,
                    conf=PLATE_CONFIDENCE,
                    imgsz=640,
                    device=DETECTION_DEVICE,
                    verbose=False,
                )
                plate_detections: list[dict[str, Any]] = []
                for prediction in plate_results:
                    if prediction.boxes is None:
                        continue
                    for box in prediction.boxes:
                        px1, py1, px2, py2 = (int(value) for value in box.xyxy[0].tolist())
                        plate_detections.append({
                            "bbox": [px1, py1, px2, py2],
                            "conf": float(box.conf[0].item()),
                        })

                person_detected = False
                vehicle_detections: list[dict[str, Any]] = []
                light_boxes: list[tuple[list[int], float]] = []
                for prediction in vehicle_results:
                    boxes = prediction.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        class_id = int(box.cls[0].item())
                        confidence = float(box.conf[0].item())
                        coordinates = [int(value) for value in box.xyxy[0].tolist()]
                        if class_id == PERSON_CLASS_ID:
                            person_detected = True
                        elif class_id == TRAFFIC_LIGHT_CLASS_ID:
                            if confidence >= TRAFFIC_LIGHT_CONFIDENCE:
                                light_boxes.append((coordinates, confidence))
                        elif class_id in VEHICLE_CLASSES:
                            vehicle_detections.append({
                                "class_id": class_id,
                                "bbox": coordinates,
                                "conf": confidence,
                            })

                signals = _classify_signals(frame, light_boxes)
                tracker.update_traffic_signals(
                    timestamp,
                    signals,
                    frame.shape[1],
                    frame.shape[0],
                )

                track_ids = iou_tracker.update(
                    timestamp,
                    [(item["class_id"], item["bbox"]) for item in vehicle_detections],
                )

                for item, track_id in zip(vehicle_detections, track_ids):
                    item["track_id"] = track_id
                    if item["class_id"] != 3:
                        continue
                    if timestamp - helmet_scans.get(track_id, -1e9) < HELMET_SCAN_INTERVAL_SEC:
                        continue
                    helmet_scans[track_id] = timestamp
                    if helmet_model is None:
                        helmet_model = _load_helmet_model()
                    if helmet_model is None:
                        continue
                    state_name, confidence, helmet_bbox, head_bbox = _scan_helmet(
                        helmet_model, frame, item["bbox"]
                    )
                    helmet_cache[track_id] = {
                        "state": state_name,
                        "conf": confidence,
                        "bbox": helmet_bbox or head_bbox,
                        "helmet_bbox": helmet_bbox,
                        "head_bbox": head_bbox,
                    }
                    tracker.update_helmet(
                        timestamp,
                        track_id,
                        state_name,
                        confidence,
                        helmet_bbox,
                        head_bbox,
                        wall_clock,
                    )

                tracker.update_plates(
                    timestamp,
                    plate_detections,
                    vehicle_detections,
                    frame,
                    reader,
                    wall_clock,
                )

                for item in vehicle_detections:
                    tracker.update_vehicle(
                        timestamp,
                        item["track_id"],
                        item["class_id"],
                        item["bbox"],
                        frame.shape[1],
                        now_str=wall_clock,
                    )
                tracker.update_persons(timestamp, person_detected)

                plate_labels = tracker.plate_labels(timestamp)
                helmet_labels = [
                    {
                        "state": info["state"],
                        "conf": info["conf"],
                        "bbox": info["bbox"],
                    }
                    for info in helmet_cache.values()
                    if info.get("bbox") and info["state"] in {"helmet", "no_helmet"}
                ]

                annotated = _annotate_frame(
                    frame,
                    vehicle_detections,
                    plate_labels,
                    signals,
                    helmet_labels,
                    wall_clock,
                    stop_line=tracker.stop_line,
                )
                output_frame = self._prepare_output_frame(annotated)
                output_height, output_width = output_frame.shape[:2]

                for event in tracker.drain_events():
                    event["snapshot"] = _encode_snapshot(output_frame, str(event.get("label", "")))
                    if event.get("request_clip"):
                        self._start_clip(
                            filename,
                            event,
                            timestamp,
                            output_width,
                            output_height,
                        )
                    else:
                        self._dispatch(filename, event)

                self._buffer_frame(timestamp, output_frame)

                for pending in list(self._pending_clips):
                    if timestamp >= float(pending["end"]):
                        self._pending_clips.remove(pending)
                        self._finalize_clip(pending)

                processed_seconds = timestamp
                progress = min(99, int((frame_index + 1) * 100 / total_frames)) if total_frames > 0 else 0
                self._update(
                    job_id,
                    progress=progress,
                    processed_seconds=round(processed_seconds, 1),
                    duration_seconds=round(duration, 1),
                    events=list(tracker.events),
                )
                frame_index += 1

            if frame_index == 0:
                raise ValueError("The uploaded video contains no readable frames.")
            if total_frames <= 0:
                duration = frame_index / fps
            for pending in list(self._pending_clips):
                self._pending_clips.remove(pending)
                self._finalize_clip(pending)
            self._update(
                job_id,
                status="completed",
                progress=100,
                processed_seconds=round(duration, 1),
                duration_seconds=round(duration, 1),
                events=list(tracker.events),
            )
        except Exception as exc:
            self._update(job_id, status="error", error=str(exc))
        finally:
            for pending in list(self._pending_clips):
                try:
                    pending["writer"].release()
                except Exception:
                    pass
                try:
                    pending["path"].unlink(missing_ok=True)
                except OSError:
                    pass
            self._pending_clips.clear()
            self._preroll.clear()
            if capture is not None:
                capture.release()
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError as exc:
                print(f"[Uploaded Video] Could not remove temporary upload: {exc}")
