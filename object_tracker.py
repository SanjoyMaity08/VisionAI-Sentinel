"""Gyanovate VisionAI - Object Retriever & Last Detected Location Tracker.

Tracks every object that appears within the camera's field of view
(across all 80 COCO categories + common search aliases like book bag, pen, camera, etc.)
and indexes its:
- Last seen timestamp
- Precise spatial coordinates and room quadrant
- High-resolution cropped snapshot
- 10-second MP4 video clip showing the item in the scene
"""
from __future__ import annotations

import difflib
import json
from datetime import datetime
from pathlib import Path
import re
import threading
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from vision_detection import (
    DEFAULT_MIN_CONFIDENCE,
    MIN_CONFIDENCE_BY_LABEL,
    OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE,
    OPEN_VOCABULARY_MIN_CONFIDENCE,
    REFERENCE_REQUIRED_LABELS,
    REFERENCE_MIN_SIMILARITY,
    REFERENCE_MIN_MARGIN,
    SECONDARY_REQUIRED_LABELS,
)

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "object_index.json"
CLIP_ASSOC_VERSION_FILE = ROOT / ".clip_association_v3"
SNAPSHOT_DIR = ROOT / "incidents" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ACTIVE_OBJECT_CAPTURES = 6

# Common human search term aliases and common typos mapped to YOLO COCO + Micro-AI classes
ALIASES: dict[str, list[str]] = {
    # Bags & luggage
    "bag": ["bag", "backpack", "handbag", "suitcase"],
    "bagg": ["backpack", "handbag"],
    "book bag": ["backpack", "handbag"],
    "bookbag": ["backpack"],
    "backpack": ["backpack"],
    "handbag": ["handbag"],
    "purse": ["handbag"],
    "suitcase": ["suitcase"],
    "luggage": ["suitcase"],
    # Phones & mobile
    "phone": ["cell phone", "smartphone", "mobile phone"],
    "cell phone": ["cell phone"],
    "cellphone": ["cell phone"],
    "mobile": ["cell phone", "mobile phone"],
    "smartphone": ["cell phone", "smartphone"],
    "fon": ["cell phone"],
    "phn": ["cell phone"],
    # Cameras & optics
    "camera": ["camera"],
    "camra": ["camera"],
    "camer": ["camera"],
    "cemera": ["camera"],
    # Writing & stationary
    "pen": ["pen"],
    "pencil": ["pencil"],
    "pencile": ["pencil"],
    "pensil": ["pencil"],
    "stationary": ["pen", "pencil"],
    "stylus": ["pen"],
    "pen pencil": ["pen pencil", "pen", "pencil"],
    # Keys & badges
    "keys": ["keys"],
    "key": ["keys"],
    "keis": ["keys"],
    "badge": ["keys", "book"],
    "id card": ["keys", "book"],
    "wallet": ["handbag", "keys"],
    "scissors": ["scissors"],
    "book": ["book"],
    "notebook": ["notebook", "book"],
    # Computers
    "laptop": ["laptop"],
    "leptop": ["laptop"],
    "laptp": ["laptop"],
    "computer": ["computer monitor", "desktop computer", "computer tower", "computer case", "laptop"],
    "pc": ["desktop computer", "computer tower", "computer case", "laptop"],
    "desktop": ["computer monitor", "desktop computer", "computer tower", "computer case"],
    "cpu": ["desktop computer", "computer tower", "computer case"],
    "screen": ["computer monitor"],
    "monitor": ["computer monitor"],
    "tablet": ["tablet"],
    # Bottles & cups (with common typos)
    "bottle": ["bottle"],
    "bottel": ["bottle"],
    "botle": ["bottle"],
    "bootle": ["bottle"],
    "botol": ["bottle"],
    "water bottle": ["bottle"],
    "water bottel": ["bottle"],
    "water botle": ["bottle"],
    "cup": ["cup", "wine glass", "bowl"],
    "mug": ["cup"],
    "glass": ["wine glass", "cup"],
    # Furniture & peripherals
    "chair": ["chair", "couch"],
    "keyboard": ["keyboard"],
    "mouse": ["mouse", "computer mouse"],
    "computer mouse": ["computer mouse", "mouse"],
    "remote": ["remote", "remote control"],
    "mug": ["cup", "mug"],
    "water bottle": ["bottle", "water bottle"],
    "glasses": ["eyeglasses"],
    "headphones": ["headphones"],
    "usb": ["usb flash drive"],
    "clock": ["clock"],
    "umbrella": ["umbrella"],
    "adaptor": ["adaptor"],
    "adapter": ["adaptor"],
    "charger": ["adaptor"],
    "switch board": ["switch board"],
    "switchboard": ["switch board"],
    "extension board": ["switch board"],
    "towel": ["towel"],
    "utensils": ["utensils"],
    "shoe": ["shoes"],
    "shoes": ["shoes"],
    "fan": ["fan"],
    "ceiling fan": ["fan"],
    "good night": ["good night machine"],
    "good night machine": ["good night machine"],
    "mosquito vaporizer": ["good night machine"],
}

STOP_WORDS = {
    "where", "is", "the", "a", "an", "find", "search", "for", "locate", "show", "me",
    "what", "are", "my", "our", "in", "room", "view", "camera", "fov", "please",
    "can", "you", "get", "clip", "video", "footage", "last", "known", "seen", "location", "of", "about"
}


def get_quadrant_description(cx: int, cy: int, w: int, h: int) -> str:
    """Describe spatial location in human terms (e.g. Center Desk, Upper Right)."""
    horiz = "Left" if cx < w * 0.35 else ("Right" if cx > w * 0.65 else "Center")
    vert = "Upper" if cy < h * 0.35 else ("Lower" if cy > h * 0.65 else "Middle")
    return f"{vert} {horiz} Area (X: {cx}, Y: {cy})"


class ObjectTracker:
    def __init__(self) -> None:
        self._index_lock = threading.RLock()
        self.index: dict[str, dict[str, Any]] = self._load_index()
        self.last_clip_time: dict[str, float] = {}
        self._live_frames_lock = threading.Lock()
        self._clip_writer_lock = threading.Lock()
        self._pre_detection_frames: deque[tuple[float, bytes, list[dict[str, Any]]]] = deque(maxlen=180)
        self._active_object_captures: dict[str, dict[str, Any]] = {}
        self._encoding_object_labels: set[str] = set()
        self._clip_valid_after: dict[str, float] = {}
        self._index_changed_during_load = False
        self._snapshot_last_write: dict[str, float] = {}
        self._last_index_save = 0.0
        self._last_prebuffer_append = 0.0
        self._clip_listing_cache: tuple[float, list[str]] = (0.0, [])
        self._clip_existence_cache: dict[str, tuple[float, bool]] = {}
        self._quarantine_known_false_detections()
        self.clip_writer: Any = None
        # Older builds assigned the newest incident clip to every object,
        # even when the object was absent. Drop those unverified links once.
        if not CLIP_ASSOC_VERSION_FILE.exists():
            for key, item in self.index.items():
                if isinstance(item, dict):
                    item["video_clip"] = ""
            self._save_index()
            CLIP_ASSOC_VERSION_FILE.write_text("verified object clip links\n", encoding="utf-8")
        for label, item in self.index.items():
            try:
                self._clip_valid_after[label] = float(item.get("clip_valid_after", 0.0))
            except (TypeError, ValueError):
                self._clip_valid_after[label] = 0.0
            clip = item.get("video_clip") if isinstance(item, dict) else ""
            if self._is_object_detection_clip(clip, label):
                path = ROOT / "incidents" / Path(clip).name
                self.last_clip_time[label] = path.stat().st_mtime
        # Quarantined labels must keep their old clips hidden even when the
        # files' mtimes were touched after the original cutoff.
        for label, item in self.index.items():
            if isinstance(item, dict) and item.get("detection_verified") is False:
                self._clip_valid_after[label] = max(self._clip_valid_after.get(label, 0.0), time.time())
                item["clip_valid_after"] = self._clip_valid_after[label]
                self._index_changed_during_load = True
        if self._index_changed_during_load:
            self._save_index()

    def _seed_default_index(self) -> None:
        """Retained for compatibility; unobserved objects are never pre-populated."""

    def _quarantine_known_false_detections(self) -> None:
        """Hide known stale false detections and preserve their clips on disk."""
        for key, item in self.index.items():
            if not isinstance(item, dict) or item.get("detection_verified") is False:
                continue
            label = str(item.get("label") or key).lower().strip()
            source = str(item.get("detection_source", "")).lower()
            if source == "manual review using supplied reference":
                item["confidence"] = "Reviewed"
                self._index_changed_during_load = True
                continue
            reason = ""
            if label == "pen":
                bbox = item.get("bbox")
                try:
                    touches_edge = (
                        isinstance(bbox, (list, tuple))
                        and len(bbox) == 4
                        and (int(bbox[0]) <= 0 or int(bbox[1]) <= 0)
                    )
                except (TypeError, ValueError):
                    touches_edge = False
                if touches_edge or "geometry" in source or "contour" in source:
                    reason = "Unverified edge-contour match"
            if label == "tv":
                reason = "Ambiguous display class; use verified monitor or computer labels"
            try:
                confidence = float(str(item.get("confidence", "0")).rstrip("%")) / 100.0
            except (TypeError, ValueError):
                confidence = 0.0
            threshold = (
                OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE.get(label, OPEN_VOCABULARY_MIN_CONFIDENCE)
                if source == "open vocabulary ai"
                else REFERENCE_MIN_SIMILARITY if source == "reference photo ai"
                else MIN_CONFIDENCE_BY_LABEL.get(label, DEFAULT_MIN_CONFIDENCE)
            )
            if confidence < threshold:
                reason = f"Quarantined low-confidence {label} match"
            if reason:
                self._quarantine_detection(label, item, reason)

    def _quarantine_detection(self, label: str, item: dict[str, Any], reason: str) -> None:
        try:
            cutoff = float(item.get("clip_valid_after", 0.0)) or time.time()
        except (TypeError, ValueError):
            cutoff = time.time()
        item["detection_verified"] = False
        item["detection_source"] = reason
        item["clip_valid_after"] = cutoff
        item["video_clip"] = ""
        item["snapshot_file"] = ""
        item["bbox"] = []
        item["center"] = []
        item["confidence"] = "Unverified"
        item["location_desc"] = f"No verified {label} detection"
        self._clip_valid_after[label] = cutoff
        self._index_changed_during_load = True

    def _load_index(self) -> dict[str, dict[str, Any]]:
        if INDEX_FILE.exists():
            try:
                data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
                if data:
                    return data
            except Exception:
                pass
        return {}

    def _save_index(self, force: bool = False) -> None:
        """Persist the catalog, throttled so per-frame detections stay fast."""
        now = time.monotonic()
        with self._index_lock:
            if not force and now - self._last_index_save < 3.0:
                return
            try:
                INDEX_FILE.write_text(json.dumps(self.index, indent=2), encoding="utf-8")
                self._last_index_save = now
            except Exception as exc:
                print(f"[Object Tracker] Index save error: {exc}")

    def invalidate_clip_cache(self) -> None:
        """Force the next clip lookup to rescan the incidents folder."""
        self._clip_listing_cache = (0.0, [])
        self._clip_existence_cache.clear()

    def _incident_clip_names(self) -> list[str]:
        now = time.monotonic()
        if now - self._clip_listing_cache[0] < 3.0:
            return self._clip_listing_cache[1]
        try:
            names = sorted(
                path.name
                for path in (ROOT / "incidents").glob("*.mp4")
                if path.is_file()
            )
        except OSError:
            names = []
        self._clip_listing_cache = (now, names)
        return names

    def _clip_file_exists(self, clip_name: str) -> bool:
        if not clip_name or Path(clip_name).name != clip_name:
            return False
        now = time.monotonic()
        cached = self._clip_existence_cache.get(clip_name)
        if cached is not None and now - cached[0] < 2.0:
            return cached[1]
        exists = (ROOT / "incidents" / clip_name).is_file()
        self._clip_existence_cache[clip_name] = (now, exists)
        return exists

    @staticmethod
    def _is_verified_detection(box: dict[str, Any]) -> bool:
        label = str(box.get("label", "")).lower().strip()
        source = str(box.get("source", "YOLO"))
        if not label or label in {"person", "tv"} or len(box.get("bbox", [])) != 4:
            return False
        if label in REFERENCE_REQUIRED_LABELS and source != "Reference Photo AI":
            return False
        if label in SECONDARY_REQUIRED_LABELS and source not in {"Reference Photo AI", "Open Vocabulary AI"}:
            return False
        if source == "Reference Photo AI":
            return (float(box.get("conf", 0.0)) >= REFERENCE_MIN_SIMILARITY
                    and float(box.get("reference_margin", 0.0)) >= REFERENCE_MIN_MARGIN)
        threshold = (OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE.get(label, OPEN_VOCABULARY_MIN_CONFIDENCE)
                     if source == "Open Vocabulary AI"
                     else MIN_CONFIDENCE_BY_LABEL.get(label, DEFAULT_MIN_CONFIDENCE))
        return float(box.get("conf", 0.0)) >= threshold and (
            label != "pen" or source in {"Deep Neural Secondary AI", "Open Vocabulary AI"})

    def update_detections(
        self,
        frame: np.ndarray,
        boxes_data: list[dict[str, Any]],
        now: float,
        encoded_frame: bytes | None = None,
    ) -> None:
        """Record latest observed location for all detected non-person objects."""
        h, w = frame.shape[:2]
        boxes_data = [box for box in boxes_data if self._is_verified_detection(box)]
        updated = False

        # Capture by elapsed time rather than frame count; CPU inference can
        # run below 15 FPS, so a 150-frame clip could otherwise take minutes.
        frame_sample: tuple[float, bytes, list[dict[str, Any]]] | None = None
        if now - self._last_prebuffer_append >= 0.06:
            try:
                jpeg_bytes = encoded_frame
                if not jpeg_bytes:
                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    jpeg_bytes = encoded.tobytes() if ok else None
                if jpeg_bytes:
                    live_boxes = [
                        {
                            "label": str(b.get("label", "")).lower().strip(),
                            "display_name": str(b.get("display_name") or b.get("label", "")).strip(),
                            "bbox": list(b.get("bbox", [])),
                            "conf": float(b.get("conf", 0.0)),
                            "source": str(b.get("source", "YOLO")),
                            "reference_margin": float(b.get("reference_margin", 0.0)),
                        }
                        for b in boxes_data
                        if b.get("label") and b.get("label", "").lower() != "person" and len(b.get("bbox", [])) == 4
                    ]
                    frame_sample = (now, jpeg_bytes, live_boxes)
                    self._last_prebuffer_append = now
                    with self._live_frames_lock:
                        self._pre_detection_frames.append(frame_sample)
                        while self._pre_detection_frames and now - self._pre_detection_frames[0][0] > 3.0:
                            self._pre_detection_frames.popleft()
            except Exception:
                pass

        completed_captures: list[tuple[str, list[tuple[float, bytes, list[dict[str, Any]]]]]] = []
        if frame_sample is not None:
            detected_labels = {box["label"] for box in frame_sample[2]}
            with self._live_frames_lock:
                for active_label, capture in list(self._active_object_captures.items()):
                    capture["frames"].append(frame_sample)
                    if now - capture["started"] >= 7.0:
                        completed_captures.append((active_label, capture["frames"]))
                        self._encoding_object_labels.add(active_label)
                        del self._active_object_captures[active_label]

                active_labels = set(self._active_object_captures)
                # A clip is named after its object, so only start one when the
                # label has actually appeared across several recent frames.
                recent_samples = list(self._pre_detection_frames)[-15:]
                eligible = [label for label in detected_labels
                            if label not in active_labels
                            and time.time() - self.last_clip_time.get(label, 0) >= 180
                            and sum(
                                1 for sample in recent_samples
                                if any(box["label"] == label for box in sample[2])
                            ) >= 2]
                for label in eligible:
                    if len(self._active_object_captures) >= MAX_ACTIVE_OBJECT_CAPTURES:
                        break
                    self._active_object_captures[label] = {
                        "started": now,
                        "frames": list(self._pre_detection_frames),
                    }
                    self.last_clip_time[label] = time.time()

        for b in boxes_data:
            label = b.get("label", "").lower().strip()
            if not label or label == "person":
                continue
            source = str(b.get("source", "YOLO"))
            detection_confidence = float(b.get("conf", 0.0))

            cx, cy = b["center"]
            conf = detection_confidence
            bbox = b.get("bbox", [0, 0, 0, 0])
            loc_desc = get_quadrant_description(cx, cy, w, h)
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save cropped snapshot with margin
            x1, y1, x2, y2 = bbox
            pad = 0
            sx1, sy1 = max(0, x1 - pad), max(0, y1 - pad)
            sx2, sy2 = min(w, x2 + pad), min(h, y2 + pad)
            snap_name = f"snap_obj_{label.replace(' ', '_')}.jpg"
            snap_path = SNAPSHOT_DIR / snap_name
            last_snapshot_write = self._snapshot_last_write.get(label)
            if last_snapshot_write is None or now - last_snapshot_write >= 2.0:
                try:
                    crop = frame[sy1:sy2, sx1:sx2].copy()
                    if crop.size > 0:
                        cv2.rectangle(crop, (x1 - sx1, y1 - sy1), (x2 - sx1, y2 - sy1), (0, 229, 255), 2)
                        if cv2.imwrite(str(snap_path), crop):
                            self._snapshot_last_write[label] = now
                except Exception:
                    pass
            snapshot_ready = label in self._snapshot_last_write

            # A fresh detection does not prove that the object appears in the
            # latest incident recording. Keep its prior verified clip link.
            with self._index_lock:
                clip_file = self.index.get(label, {}).get("video_clip", "")
                if (
                    not self._is_object_detection_clip(clip_file, label)
                    or self.is_object_clip_quarantined(clip_file)
                ):
                    clip_file = self._find_latest_object_detection_clip(label)
                updated_item = {
                    "label": label,
                    "display_name": str(b.get("display_name") or label.title()),
                    "last_seen_time": timestamp_str,
                    "confidence": f"{int(conf * 100)}%",
                    "detection_verified": True,
                    "detection_source": source,
                    "bbox": bbox,
                    "center": [cx, cy],
                    "location_desc": loc_desc,
                    "video_clip": clip_file,
                    "snapshot_file": snap_name if snapshot_ready else "",
                }
                if self._clip_valid_after.get(label, 0.0):
                    updated_item["clip_valid_after"] = self._clip_valid_after[label]
                self.index[label] = updated_item
            updated = True

        if updated:
            self._save_index()

        for label, samples in completed_captures:
            threading.Thread(
                target=self._save_object_detection_clip,
                args=(label, samples),
                daemon=True,
            ).start()

    def _save_object_detection_clip(self, label: str, samples: list[tuple[float, bytes, list[dict[str, Any]]]]) -> None:
        saved = False
        try:
            with self._clip_writer_lock:
                saved = self._encode_object_detection_clip(label, samples)
        except Exception as exc:
            print(f"[Object Clip Error] {label}: {exc}")
        finally:
            with self._live_frames_lock:
                self._encoding_object_labels.discard(label)
                if not saved:
                    self.last_clip_time.pop(label, None)

    def _encode_object_detection_clip(self, label: str, samples: list[tuple[float, bytes, list[dict[str, Any]]]]) -> bool:
        if not self.clip_writer:
            return False
        frames: list[np.ndarray] = []
        matching_frame_count = 0
        matching_times = []
        for observed_at, jpg, boxes in samples:
            matching_boxes = [box for box in boxes if box["label"] == label and self._is_verified_detection(box)]
            if not matching_boxes:
                continue
            image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            h, w = image.shape[:2]
            if w > 960 or h > 540:
                scale = min(960 / w, 540 / h)
                image = cv2.resize(image, (max(2, int(w * scale)), max(2, int(h * scale))), interpolation=cv2.INTER_AREA)
                scale_y = image.shape[0] / h
                scale_x = image.shape[1] / w
            else:
                scale_x = scale_y = 1.0
            matching_frame_count += 1
            matching_times.append(observed_at)
            for box in matching_boxes:
                bx1, by1, bx2, by2 = map(int, box["bbox"])
                x1, y1, x2, y2 = (int(bx1 * scale_x), int(by1 * scale_y), int(bx2 * scale_x), int(by2 * scale_y))
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 3)
                confidence = float(box.get("conf", 0.0))
                display_name = str(box.get("display_name") or self.index.get(label, {}).get("display_name") or label.title())
                cv2.putText(image, f"{display_name.upper()} | MODEL SCORE {confidence:.0%}", (max(4, x1), max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2)
            frames.append(image)
        if matching_frame_count < 3:
            return False
        filename = f"object_{re.sub(r'[^a-zA-Z0-9]+', '_', label).strip('_')}_{int(time.time() * 1000)}.mp4"
        path = ROOT / "incidents" / filename
        elapsed = max(0.1, matching_times[-1] - matching_times[0])
        actual_fps = max(1.0, min(30.0, (len(frames) - 1) / elapsed))
        saved = self.clip_writer(path, frames, fps=actual_fps)
        if saved:
            with self._index_lock:
                if (label in self.index
                        and self.index[label].get("detection_verified") is not False
                        and matching_times[-1] >= self._clip_valid_after.get(label, 0.0)):
                    self.index[label]["video_clip"] = filename
            self.invalidate_clip_cache()
            self._save_index(force=True)
        return bool(saved)

    def _ensure_media_links(self, item: dict[str, Any]) -> None:
        """Only expose this object's own detection clip, never a generic incident clip."""
        if item.get("detection_verified") is False:
            item["video_clip"] = ""
            item["snapshot_file"] = ""
            return
        clip_name = item.get("video_clip", "")
        label = item.get("label", "")
        if not self._is_object_detection_clip(clip_name, label) or self.is_object_clip_quarantined(clip_name):
            item["video_clip"] = self._find_latest_object_detection_clip(label)

        snap_name = item.get("snapshot_file", "")
        if not snap_name or not (SNAPSHOT_DIR / snap_name).is_file():
            item["snapshot_file"] = ""

    @staticmethod
    def _object_clip_prefix(label: str) -> str:
        normalized_label = re.sub(r"[^a-zA-Z0-9]+", "_", label).strip("_")
        return f"object_{normalized_label}_"

    def _find_latest_object_detection_clip(self, label: str) -> str:
        if not label:
            return ""
        prefix = self._object_clip_prefix(label)
        candidates = [
            name
            for name in self._incident_clip_names()
            if name.startswith(prefix) and not self.is_object_clip_quarantined(name)
        ]
        if not candidates:
            return ""
        incidents_dir = ROOT / "incidents"
        latest = max(
            candidates,
            key=lambda name: (incidents_dir / name).stat().st_mtime
            if (incidents_dir / name).is_file() else 0.0,
            default="",
        )
        return latest

    def is_object_clip_quarantined(self, clip_name: str) -> bool:
        if not clip_name or Path(clip_name).name != clip_name:
            return False
        if not self._clip_file_exists(clip_name):
            return False
        for label, cutoff in self._clip_valid_after.items():
            if clip_name.startswith(self._object_clip_prefix(label)):
                try:
                    return (ROOT / "incidents" / clip_name).stat().st_mtime < cutoff
                except OSError:
                    return False
        return False

    def _is_object_detection_clip(self, clip_name: str, label: str) -> bool:
        if not clip_name or Path(clip_name).name != clip_name or not clip_name.startswith("object_"):
            return False
        return clip_name.startswith(self._object_clip_prefix(label)) and self._clip_file_exists(clip_name)

    def is_object_clip_recording(self, label: str) -> bool:
        with self._live_frames_lock:
            return label in self._active_object_captures or label in self._encoding_object_labels

    def reset_live_capture(self, finalize: bool = False) -> None:
        with self._live_frames_lock:
            captures = list(self._active_object_captures.items()) if finalize else []
            for label in self._active_object_captures:
                self.last_clip_time.pop(label, None)
            self._encoding_object_labels.update(label for label, _ in captures)
            self._active_object_captures.clear()
            self._pre_detection_frames.clear()
            self._last_prebuffer_append = 0.0
        for label, capture in captures:
            threading.Thread(target=self._save_object_detection_clip, args=(label, capture["frames"]), daemon=True).start()

    def request_clip_for_search(self, label: str) -> None:
        """Prioritize a searched object if it is visible in the recent live frames."""
        clip_name = self.index.get(label, {}).get("video_clip", "")
        if self._is_object_detection_clip(clip_name, label) and not self.is_object_clip_quarantined(clip_name):
            return
        with self._live_frames_lock:
            if (label in self._active_object_captures or label in self._encoding_object_labels
                    or len(self._active_object_captures) >= MAX_ACTIVE_OBJECT_CAPTURES):
                return
            recent = list(self._pre_detection_frames)
            if (not recent or time.time() - recent[-1][0] > 1.0
                    or sum(any(box["label"] == label for box in sample[2]) for sample in recent) < 2):
                return
            self._active_object_captures[label] = {
                "started": time.time(),
                "frames": recent,
            }
            self.last_clip_time[label] = time.time()

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search for objects matching query string or alias keywords with typo & NLP tolerance."""
        raw_q = query.lower().strip()
        if not raw_q:
            return self.get_all_tracked()

        # Clean punctuation and filter out conversational stop words
        cleaned = re.sub(r"[^\w\s]", " ", raw_q)
        tokens = [w for w in cleaned.split() if w and w not in STOP_WORDS]
        clean_q = " ".join(tokens) if tokens else raw_q

        matched_keys: set[str] = set()

        # 1. Exact label and token matches
        for key in self.index.keys():
            if clean_q == key or key in tokens:
                matched_keys.add(key)

        # 2. Alias lookup (including common misspellings like 'bottel')
        for alias_key, target_classes in ALIASES.items():
            if clean_q == alias_key or alias_key in tokens:
                for target in target_classes:
                    if target in self.index:
                        matched_keys.add(target)

        # 3. Fuzzy matching via difflib for unrecognized typos (e.g. 'bottel', 'camra')
        if not matched_keys:
            candidate_pool = list(self.index.keys()) + list(ALIASES.keys())
            close_matches = difflib.get_close_matches(clean_q, candidate_pool, n=3, cutoff=0.55)
            for cm in close_matches:
                if cm in self.index:
                    matched_keys.add(cm)
                elif cm in ALIASES:
                    for target in ALIASES[cm]:
                        if target in self.index:
                            matched_keys.add(target)

            for token in tokens:
                close_tok = difflib.get_close_matches(token, candidate_pool, n=2, cutoff=0.55)
                for cm in close_tok:
                    if cm in self.index:
                        matched_keys.add(cm)
                    elif cm in ALIASES:
                        for target in ALIASES[cm]:
                            if target in self.index:
                                matched_keys.add(target)

        results = []
        for k in matched_keys:
            if self.index[k].get("detection_verified") is False:
                continue
            self.request_clip_for_search(k)
            item = dict(self.index[k])
            self._ensure_media_links(item)
            item["clip_recording"] = self.is_object_clip_recording(k)
            results.append(item)

        # Sort by latest seen
        results.sort(key=lambda x: x.get("last_seen_time", ""), reverse=True)
        return results

    def get_all_tracked(self) -> list[dict[str, Any]]:
        items = []
        for v in self.index.values():
            if v.get("detection_verified") is False:
                continue
            item = dict(v)
            self._ensure_media_links(item)
            items.append(item)
        items.sort(key=lambda x: x.get("last_seen_time", ""), reverse=True)
        return items
