"""Asynchronous open-vocabulary detector for common objects beyond the primary model."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
from vision_detection import (
    DETECTION_DEVICE,
    OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE,
    OPEN_VOCABULARY_MIN_CONFIDENCE,
    REFERENCE_MIN_SIMILARITY,
    REFERENCE_MIN_MARGIN,
    REFERENCE_REQUIRED_LABELS,
    configure_torch_threads,
)

ROOT = Path(__file__).resolve().parent
OPEN_VOCABULARY_LABELS = (
    "pen", "pencil", "keyboard", "computer mouse", "keys", "wallet",
    "eyeglasses", "headphones", "computer monitor", "tablet", "smartphone",
    "mobile phone", "remote control", "mug", "water bottle", "marker",
    "highlighter", "stapler", "scissors", "usb flash drive", "adaptor",
    "cable", "earbuds", "watch", "shoes", "hat", "box", "notebook",
    "screwdriver", "hammer", "calculator", "power bank", "mouse pad",
    "keychain", "document", "paper", "lunch box", "tissue box",
    "desktop computer", "computer tower", "computer case", "switch board",
    "laptop", "bag", "towel", "utensils", "umbrella", "fan",
    "good night machine",
)
REFERENCE_DIR = ROOT / "reference_objects"
REFERENCE_MANIFEST = REFERENCE_DIR / "references.json"
REFERENCE_MODEL_PATH = ROOT / "weights" / "clip" / "ViT-B-32.pt"


class PrecisionMicroAI:
    """Runs YOLO-World detail scans off the camera thread and confirms repeated hits."""

    SCAN_INTERVAL_SEC = 1.5
    RESULT_MAX_AGE_SEC = 8.0
    MIN_CONFIDENCE = 0.65
    REFERENCE_PROPOSAL_MIN_CONFIDENCE = 0.15
    REFERENCE_MIN_SIMILARITY = REFERENCE_MIN_SIMILARITY
    REFERENCE_MIN_MARGIN = REFERENCE_MIN_MARGIN
    REFERENCE_MIN_EDGE_DENSITY = 0.006
    REFERENCE_MAX_CROPS = 32
    DEFAULT_LABEL_MIN_CONFIDENCE = OPEN_VOCABULARY_MIN_CONFIDENCE
    LABEL_MIN_CONFIDENCE = OPEN_VOCABULARY_LABEL_MIN_CONFIDENCE
    REQUIRED_CONSECUTIVE_SCANS = 2
    GENERIC_LABELS = {"box"}
    COMPUTER_TOWER_LABELS = {"desktop computer", "computer tower", "computer case"}

    def __init__(self, primary_model_names: dict[int, str] | None = None) -> None:
        self.secondary_yolo = None
        self.primary_names = primary_model_names or {}
        self.reference_model = None
        self.reference_preprocess = None
        self.reference_features = None
        self.reference_entries: list[dict[str, str]] = []
        self._reference_state = "not started"
        self._reference_error = ""
        self._previous_micro_boxes: dict[str, list[dict[str, Any]]] = {}
        self._generation = 0
        self._scan_queue: queue.Queue[tuple[np.ndarray, float, list[dict[str, Any]], int]] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_results: list[dict[str, Any]] = []
        self._last_scan_submit = 0.0
        self._worker_started = False
        self._model_state = "not started"
        self._model_error = ""
        self._init_lock = threading.Lock()

    def reset(self) -> None:
        with self._result_lock:
            self._generation += 1
            self._latest_results = []
            self._previous_micro_boxes = {}
            self._last_scan_submit = 0.0

    @staticmethod
    def _appearance(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return cv2.resize(frame[y1:y2, x1:x2], (48, 48), interpolation=cv2.INTER_AREA)

    @classmethod
    def _appearance_matches(cls, frame: np.ndarray, item: dict[str, Any]) -> bool:
        x1, y1, x2, y2 = item["bbox"]
        height, width = frame.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return False
        current = cls._appearance(frame, item["bbox"])
        previous = item["_appearance"]
        difference = float(np.abs(current.astype(np.float32) - previous).mean())
        correlation = float(cv2.matchTemplate(current, previous, cv2.TM_CCOEFF_NORMED)[0, 0])
        return difference <= 20.0 and correlation >= 0.80

    @classmethod
    def _locate_appearance(cls, frame: np.ndarray, item: dict[str, Any]) -> list[int] | None:
        if cls._appearance_matches(frame, item):
            return list(item["bbox"])
        x1, y1, x2, y2 = item["bbox"]
        height, width = frame.shape[:2]
        box_width, box_height = x2 - x1, y2 - y1
        if box_width <= 0 or box_height <= 0:
            return None
        margin_x = max(box_width // 2, width // 6)
        margin_y = max(box_height // 2, height // 6)
        left, top = max(0, x1 - margin_x), max(0, y1 - margin_y)
        right, bottom = min(width, x2 + margin_x), min(height, y2 + margin_y)
        if right - left < box_width or bottom - top < box_height:
            return None
        search_width = max(48, round((right - left) * 48 / box_width))
        search_height = max(48, round((bottom - top) * 48 / box_height))
        search = cv2.resize(frame[top:bottom, left:right], (search_width, search_height), interpolation=cv2.INTER_AREA)
        scores = cv2.matchTemplate(search, item["_appearance"], cv2.TM_CCOEFF_NORMED)
        _, score, _, position = cv2.minMaxLoc(scores)
        if score < 0.75:
            return None
        moved_x = left + round(position[0] * (right - left) / search_width)
        moved_y = top + round(position[1] * (bottom - top) / search_height)
        candidate = dict(item, bbox=[moved_x, moved_y, moved_x + box_width, moved_y + box_height])
        return candidate["bbox"] if cls._appearance_matches(frame, candidate) else None

    def ensure_started(self) -> None:
        """Load the open-vocabulary model lazily so dashboard startup stays fast."""
        with self._init_lock:
            if self._model_state != "not started" or self._worker_started:
                return
            self._model_state = "loading"
            threading.Thread(target=self._init_secondary_model, daemon=True).start()

    def _init_secondary_model(self) -> None:
        """Load optional open-vocabulary weights, then start a single background inference worker."""
        weights = ROOT / "yolov8s-worldv2.pt"
        if not weights.is_file():
            self._model_state = "missing"
            print("[Open Vocabulary AI] Weights missing; run python download_models.py to install yolov8s-worldv2.pt and enable custom object labels.")
            return

        try:
            configure_torch_threads()
            from ultralytics import YOLOWorld

            model = YOLOWorld(str(weights))
            model.add_callback("on_predict_start", configure_torch_threads)
            model.set_classes(list(OPEN_VOCABULARY_LABELS))
            model.predict(
                source=np.zeros((320, 320, 3), dtype=np.uint8),
                imgsz=448,
                device=DETECTION_DEVICE,
                verbose=False,
            )
            self.secondary_yolo = model
            self._init_reference_matcher()
            threading.Thread(target=self._inference_worker, daemon=True).start()
            self._worker_started = True
            self._model_state = "ready"
            print(f"[Open Vocabulary AI] Ready with {len(OPEN_VOCABULARY_LABELS)} searchable object labels.")
        except Exception as exc:
            self.secondary_yolo = None
            self._model_state = "error"
            self._model_error = str(exc)
            print(f"[Open Vocabulary AI] Could not initialize YOLO-World: {exc}")

    def _init_reference_matcher(self) -> None:
        if not REFERENCE_MODEL_PATH.is_file() or not REFERENCE_MANIFEST.is_file():
            self._reference_state = "missing"
            print("[Reference Photo AI] Skipped; run python download_models.py for ViT-B-32.pt and add reference_objects/references.json.")
            return
        self._reference_state = "loading"
        try:
            import clip
            import torch
            from PIL import Image

            entries = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
            images = []
            valid_entries = []
            for entry in entries:
                path = REFERENCE_DIR / Path(entry["file"]).name
                if not path.is_file():
                    continue
                images.append(self._reference_image(path, Image))
                valid_entries.append({
                    "label": str(entry["label"]).lower().strip(),
                    "display_name": str(entry["display_name"]).strip(),
                })
            if not images:
                self._reference_state = "missing"
                return
            model, preprocess = clip.load(
                "ViT-B/32", device="cpu", download_root=str(REFERENCE_MODEL_PATH.parent)
            )
            model.eval()
            images = [preprocess(image) for image in images]
            with torch.inference_mode():
                features = model.encode_image(torch.stack(images))
                features /= features.norm(dim=-1, keepdim=True)
            self.reference_model = model
            self.reference_preprocess = preprocess
            self.reference_features = features.float()
            self.reference_entries = valid_entries
            self._reference_state = "ready"
            print(
                f"[Reference Photo AI] Ready with {len(valid_entries)} photos and "
                f"{len({entry['label'] for entry in valid_entries})} verified object names."
            )
        except Exception as exc:
            self.reference_model = None
            self.reference_features = None
            self.reference_entries = []
            self._reference_state = "error"
            self._reference_error = str(exc)
            print(f"[Reference Photo AI] Could not initialize: {exc}")

    @staticmethod
    def _reference_image(path: Path, image_module: Any) -> Any:
        with image_module.open(path) as image:
            return image.convert("RGB")

    def status(self) -> dict[str, Any]:
        return {
            "state": self._model_state,
            "label_count": len(OPEN_VOCABULARY_LABELS) if self._model_state == "ready" else 0,
            "error": self._model_error,
            "reference_photos": self._reference_state,
            "reference_photo_count": len(self.reference_entries),
            "reference_error": self._reference_error,
        }

    @staticmethod
    def _tile_regions(frame: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        height, width = frame.shape[:2]
        tile_width = min(width, max(320, int(width * 0.65)))
        tile_height = min(height, max(320, int(height * 0.65)))
        xs = [0] if width <= tile_width else [0, width - tile_width]
        ys = [0] if height <= tile_height else [0, height - tile_height]
        return [
            (x, y, frame[y:y + tile_height, x:x + tile_width])
            for y in ys
            for x in xs
        ]

    @staticmethod
    def _overlaps(first: list[int], second: list[int]) -> bool:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = area_a + area_b - intersection
        return bool(union and intersection / union >= 0.25)

    @classmethod
    def _same_semantic_item(cls, first: dict[str, Any], second: dict[str, Any]) -> bool:
        if first["label"] == second["label"]:
            return True
        return (
            first["label"] in cls.COMPUTER_TOWER_LABELS
            and second["label"] in cls.COMPUTER_TOWER_LABELS
        )

    def _detect_tiles(self, frame: np.ndarray) -> list[dict[str, Any]]:
        regions = self._tile_regions(frame)
        results = self.secondary_yolo.predict(
            source=[region[2] for region in regions],
            imgsz=448,
            conf=self.REFERENCE_PROPOSAL_MIN_CONFIDENCE,
            device=DETECTION_DEVICE,
            max_det=40,
            verbose=False,
        )
        height, width = frame.shape[:2]
        candidates: list[dict[str, Any]] = []
        for (offset_x, offset_y, _), result in zip(regions, results):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(self.secondary_yolo.names.get(class_id, "")).lower().strip()
                coords = [int(value) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0].item())
                if not label or len(coords) != 4:
                    continue
                x1, y1, x2, y2 = coords
                x1, x2 = max(0, min(width, x1 + offset_x)), max(0, min(width, x2 + offset_x))
                y1, y2 = max(0, min(height, y1 + offset_y)), max(0, min(height, y2 + offset_y))
                if x2 <= x1 or y2 <= y1:
                    continue
                candidate = {
                    "label": label,
                    "display_name": label.title(),
                    "bbox": [x1, y1, x2, y2],
                    "conf": confidence,
                    "source": "Open Vocabulary AI",
                }
                overlaps_existing = any(
                    self._same_semantic_item(item, candidate)
                    and self._overlaps(item["bbox"], candidate["bbox"])
                    for item in candidates
                )
                if label in self.GENERIC_LABELS:
                    overlaps_specific = any(
                        item["label"] not in self.GENERIC_LABELS
                        and self._overlaps(item["bbox"], candidate["bbox"])
                        for item in candidates
                    )
                    if overlaps_specific:
                        continue
                elif any(
                    item["label"] in self.GENERIC_LABELS
                    and self._overlaps(item["bbox"], candidate["bbox"])
                    for item in candidates
                ):
                    candidates = [
                        item for item in candidates
                        if item["label"] not in self.GENERIC_LABELS
                        or not self._overlaps(item["bbox"], candidate["bbox"])
                    ]
                if not overlaps_existing:
                    candidates.append(candidate)
        return candidates

    def _match_reference_objects(
        self,
        frame: np.ndarray,
        proposals: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.reference_model is None or self.reference_features is None:
            return []
        import torch
        from PIL import Image

        candidates = []
        images = []
        for item in sorted(
            proposals,
            key=lambda value: max(0, value["bbox"][2] - value["bbox"][0])
            * max(0, value["bbox"][3] - value["bbox"][1]),
            reverse=True,
        ):
            if item.get("label") == "person" or len(item.get("bbox", [])) != 4:
                continue
            x1, y1, x2, y2 = item["bbox"]
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            border_y, border_x = max(1, gray.shape[0] // 10), max(1, gray.shape[1] // 10)
            interior = gray[border_y:-border_y, border_x:-border_x]
            edge_density = float((cv2.Canny(interior, 60, 140) > 0).mean())
            if edge_density < self.REFERENCE_MIN_EDGE_DENSITY:
                continue
            candidates.append(item)
            images.append(self.reference_preprocess(Image.fromarray(crop[:, :, ::-1])))
            if len(candidates) >= self.REFERENCE_MAX_CROPS:
                break
        if not candidates:
            return []

        matches = []
        with torch.inference_mode():
            features = self.reference_model.encode_image(torch.stack(images))
            features /= features.norm(dim=-1, keepdim=True)
            similarities = (features.float() @ self.reference_features.T).cpu()
        labels = list(dict.fromkeys(entry["label"] for entry in self.reference_entries))
        for item, scores in zip(candidates, similarities):
            class_scores = []
            for label in labels:
                indices = [
                    index for index, entry in enumerate(self.reference_entries)
                    if entry["label"] == label
                ]
                class_scores.append((max(float(scores[index]) for index in indices), label))
            class_scores.sort(reverse=True)
            best_score, best_label = class_scores[0]
            margin = best_score - class_scores[1][0] if len(class_scores) > 1 else 1.0
            if best_score < self.REFERENCE_MIN_SIMILARITY or margin < self.REFERENCE_MIN_MARGIN:
                continue
            entry = next(entry for entry in self.reference_entries if entry["label"] == best_label)
            matches.append({
                "label": best_label,
                "display_name": entry["display_name"],
                "bbox": list(item["bbox"]),
                "conf": best_score,
                "reference_margin": margin,
                "source": "Reference Photo AI",
            })
        unique_matches = []
        for item in sorted(matches, key=lambda value: value["conf"], reverse=True):
            if any(self._overlaps(item["bbox"], old["bbox"]) for old in unique_matches):
                continue
            unique_matches.append(item)
        return unique_matches

    def _inference_worker(self) -> None:
        while True:
            frame, submitted_at, existing_boxes, generation = self._scan_queue.get()
            try:
                with self._result_lock:
                    if generation != self._generation:
                        continue
                    previous_boxes = {
                        label: [dict(item) for item in items]
                        for label, items in self._previous_micro_boxes.items()
                    }
                proposals = self._detect_tiles(frame)
                reference_matches = self._match_reference_objects(
                    frame, proposals + existing_boxes
                )
                candidates = []
                for item in proposals:
                    if item["label"] in REFERENCE_REQUIRED_LABELS:
                        continue
                    min_confidence = self.LABEL_MIN_CONFIDENCE.get(
                        item["label"], self.DEFAULT_LABEL_MIN_CONFIDENCE
                    )
                    if item["conf"] < min_confidence:
                        continue
                    if any(
                        self._overlaps(item["bbox"], match["bbox"])
                        for match in reference_matches
                    ):
                        continue
                    candidates.append(item)
                candidates.extend(reference_matches)
                current: dict[str, list[dict[str, Any]]] = {}
                verified: list[dict[str, Any]] = []
                for item in candidates:
                    label = item["label"]
                    matching_hit_count = next(
                        (
                            old["hit_count"] for old in previous_boxes.get(label, [])
                            if (located := self._locate_appearance(frame, old)) is not None
                            and self._overlaps(item["bbox"], located)
                        ),
                        0,
                    )
                    item["hit_count"] = matching_hit_count + 1
                    item["_appearance"] = self._appearance(frame, item["bbox"])
                    item["observed_at"] = submitted_at
                    current.setdefault(label, []).append(item)
                    if item["hit_count"] >= self.REQUIRED_CONSECUTIVE_SCANS:
                        verified.append(item)
                with self._result_lock:
                    if generation == self._generation:
                        self._previous_micro_boxes = current
                        self._latest_results = verified
            except Exception as exc:
                print(f"[Open Vocabulary AI] Detail scan failed: {exc}")
                with self._result_lock:
                    if generation == self._generation:
                        self._previous_micro_boxes = {}
                        self._latest_results = []
            finally:
                self._scan_queue.task_done()

    def scan_frame_for_micro_objects(
        self,
        frame: np.ndarray,
        existing_boxes: list[dict[str, Any]],
        now: float,
        active_video_clip: str | None = None,
    ) -> list[dict[str, Any]]:
        """Schedule a non-blocking tiled scan and return only recent confirmed detections."""
        self.ensure_started()
        if self.secondary_yolo is None or not self._worker_started:
            return []

        if now - self._last_scan_submit >= self.SCAN_INTERVAL_SEC:
            try:
                existing_snapshot = [
                    {
                        "label": str(item.get("label", "")),
                        "bbox": list(item.get("bbox", [])),
                        "conf": float(item.get("conf", 0.0)),
                    }
                    for item in existing_boxes
                    if len(item.get("bbox", [])) == 4
                ]
                sample = (frame.copy(), now, existing_snapshot, self._generation)
                try:
                    self._scan_queue.put_nowait(sample)
                except queue.Full:
                    try:
                        self._scan_queue.get_nowait()
                        self._scan_queue.task_done()
                    except queue.Empty:
                        pass
                    self._scan_queue.put_nowait(sample)
                self._last_scan_submit = now
            except queue.Full:
                pass

        with self._result_lock:
            tracked: dict[int, dict[str, Any]] = {}
            for item in self._latest_results + [
                item for items in self._previous_micro_boxes.values() for item in items
            ]:
                if id(item) in tracked or now - item["observed_at"] > self.RESULT_MAX_AGE_SEC:
                    continue
                located = self._locate_appearance(frame, item)
                if located is not None:
                    item["bbox"] = located
                    tracked[id(item)] = item
            self._latest_results = [item for item in self._latest_results if id(item) in tracked]
            self._previous_micro_boxes = {
                label: [item for item in items if id(item) in tracked]
                for label, items in self._previous_micro_boxes.items()
            }
            results = [
                {key: value for key, value in item.items() if key not in {"_appearance", "hit_count"}}
                for item in self._latest_results
            ]

        return [
            item for item in results
            if item.get("source") == "Reference Photo AI" or not any(
                existing.get("label") == item["label"]
                and len(existing.get("bbox", [])) == 4
                and self._overlaps(existing.get("bbox", []), item["bbox"])
                for existing in existing_boxes
            )
        ]
