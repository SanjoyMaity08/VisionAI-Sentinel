"""Low-rate pose checks for possible falls and sustained raised-arm interactions."""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np

from vision_detection import DETECTION_DEVICE, configure_torch_threads

ROOT = Path(__file__).resolve().parent


class EmergencyActionDetector:
    """Run pose inference off the camera thread and emit conservative event candidates."""

    SCAN_INTERVAL_SEC = 1.0
    POSE_CONFIDENCE = 0.45
    EVENT_HISTORY_SIZE = 8
    EVIDENCE_SECONDS = 12.0
    MAX_SCAN_GAP_SEC = 2.5

    def __init__(self) -> None:
        self._frames: queue.Queue[tuple[np.ndarray, float, int]] = queue.Queue(maxsize=1)
        self._events: deque[dict[str, Any]] = deque(maxlen=self.EVENT_HISTORY_SIZE)
        self._lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._model: Any = None
        self._state = "not started"
        self._last_submit = 0.0
        self._generation = 0
        self._previous_poses: list[dict[str, Any]] = []
        self._next_person_id = 0
        self._last_scan_at: float | None = None
        self._fall_candidates: dict[int, dict[str, Any]] = {}
        self._interaction_candidates: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._captured_at: float | None = None
        self._status_candidates: list[dict[str, Any]] = []

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def status(self) -> dict[str, Any]:
        """Return a detached dashboard snapshot; captured_at identifies its freshness."""
        with self._lock:
            candidates = [dict(candidate) for candidate in self._status_candidates]
            state = self._state
            if state == "ready" and candidates:
                state = "confirmed" if any(item["confirmed"] for item in candidates) else "checking"
            return {"state": state, "captured_at": self._captured_at, "candidates": candidates}

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._state = "loading"
            self._worker = threading.Thread(target=self._run, name="emergency-pose-detector", daemon=True)
            self._worker.start()

    def reset(self) -> None:
        with self._analysis_lock:
            self._previous_poses = []
            self._next_person_id = 0
            self._last_scan_at = None
            self._fall_candidates.clear()
            self._interaction_candidates.clear()
            self._last_submit = 0.0
            with self._lock:
                self._generation += 1
                self._events.clear()
                self._captured_at = None
                self._status_candidates = []
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def submit(self, frame: np.ndarray, captured_at: float) -> None:
        if self.state != "ready" or captured_at - self._last_submit < self.SCAN_INTERVAL_SEC:
            return
        self._last_submit = captured_at
        with self._lock:
            generation = self._generation
        sample = (frame.copy(), captured_at, generation)
        try:
            self._frames.put_nowait(sample)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(sample)
            except queue.Full:
                pass

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _run(self) -> None:
        try:
            configure_torch_threads()
            from ultralytics import YOLO

            pose_path = ROOT / "yolo26n-pose.pt"
            model = YOLO(str(pose_path) if pose_path.is_file() else "yolo26n-pose.pt")
            model.add_callback("on_predict_start", configure_torch_threads)
        except Exception as exc:
            with self._lock:
                self._state = f"unavailable: {exc}"
            print(f"[Emergency Pose] Detector unavailable: {exc}")
            return

        with self._lock:
            self._model = model
            self._state = "ready"
        print("[Emergency Pose] Pose model ready; checking possible falls and raised-arm interactions for 12 seconds.")

        while True:
            frame, captured_at, generation = self._frames.get()
            try:
                result = model.predict(
                    source=frame,
                    imgsz=480,
                    conf=0.35,
                    device=DETECTION_DEVICE,
                    max_det=8,
                    verbose=False,
                )[0]
                poses = self._extract_poses(result)
                with self._analysis_lock:
                    with self._lock:
                        is_current_session = generation == self._generation
                    if is_current_session:
                        self._analyze_poses(poses, frame.shape[1], frame.shape[0], captured_at)
            except Exception as exc:
                with self._lock:
                    self._state = f"error: {exc}"
                print(f"[Emergency Pose] Inference failed: {exc}")
                return

    @classmethod
    def _extract_poses(cls, result: Any) -> list[dict[str, Any]]:
        if result.boxes is None or result.keypoints is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        points = result.keypoints.xy.cpu().numpy()
        confidences = result.keypoints.conf
        if confidences is None:
            return []
        keypoint_scores = confidences.cpu().numpy()
        poses = []
        for box, keypoints, scores in zip(boxes, points, keypoint_scores):
            if len(keypoints) < 17 or len(scores) < 17:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                continue
            poses.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "keypoints": keypoints,
                "scores": scores,
            })
        return poses

    @staticmethod
    def _visible(pose: dict[str, Any], index: int, threshold: float) -> bool:
        return float(pose["scores"][index]) >= threshold

    @staticmethod
    def _point(pose: dict[str, Any], index: int) -> tuple[float, float]:
        point = pose["keypoints"][index]
        return float(point[0]), float(point[1])

    @classmethod
    def _shoulder_hip_axis(cls, pose: dict[str, Any]) -> tuple[float, float, float, float] | None:
        if not all(cls._visible(pose, index, cls.POSE_CONFIDENCE) for index in (5, 6, 11, 12)):
            return None
        left_shoulder, right_shoulder = cls._point(pose, 5), cls._point(pose, 6)
        left_hip, right_hip = cls._point(pose, 11), cls._point(pose, 12)
        return (
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0,
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0,
        )

    @classmethod
    def _is_horizontal(cls, pose: dict[str, Any]) -> bool:
        axis = cls._shoulder_hip_axis(pose)
        if axis is None:
            return False
        sx, sy, hx, hy = axis
        x1, y1, x2, y2 = pose["bbox"]
        width, height = x2 - x1, y2 - y1
        torso_dx, torso_dy = abs(hx - sx), abs(hy - sy)
        return width / max(height, 1.0) >= 1.05 and torso_dx >= max(18.0, torso_dy * 1.2)

    @classmethod
    def _is_upright(cls, pose: dict[str, Any]) -> bool:
        """Require visible, vertically oriented torso evidence, not merely non-horizontal."""
        axis = cls._shoulder_hip_axis(pose)
        if axis is None:
            return False
        sx, sy, hx, hy = axis
        x1, y1, x2, y2 = pose["bbox"]
        return (
            (y2 - y1) / max(x2 - x1, 1.0) >= 1.05
            and hy - sy >= max(18.0, abs(hx - sx) * 1.2)
        )

    @classmethod
    def _matches_previous(
        cls,
        pose: dict[str, Any],
        previous: list[dict[str, Any]],
        now: float,
        max_distance: float,
    ) -> dict[str, Any] | None:
        candidates = [
            item for item in previous
            if 0.0 <= now - item["time"] <= cls.MAX_SCAN_GAP_SEC
            and ("person_id" not in pose or item.get("person_id") == pose["person_id"])
            and math.dist(pose["center"], item["center"]) <= max_distance
        ]
        return min(candidates, key=lambda item: math.dist(pose["center"], item["center"]), default=None)

    @classmethod
    def _arm_contact_candidate(
        cls,
        poses: list[dict[str, Any]],
        previous: list[dict[str, Any]],
        width: int,
        height: int,
        now: float,
        *,
        only_actor_index: int | None = None,
    ) -> bool:
        frame_diagonal = math.hypot(width, height)
        for actor_index, actor in enumerate(poses):
            if only_actor_index is not None and actor_index != only_actor_index:
                continue
            old_actor = cls._matches_previous(actor, previous, now, frame_diagonal * 0.12)
            if old_actor is None:
                continue
            for target_index, target in enumerate(poses):
                if actor_index == target_index:
                    continue
                if math.dist(actor["center"], target["center"]) > frame_diagonal * 0.35:
                    continue
                x1, y1, x2, y2 = target["bbox"]
                target_width, target_height = x2 - x1, y2 - y1
                upper_body = (
                    x1 - target_width * 0.08,
                    y1 - target_height * 0.12,
                    x2 + target_width * 0.08,
                    y1 + target_height * 0.68,
                )
                target_center = ((x1 + x2) / 2.0, y1 + target_height * 0.32)
                for elbow_index, wrist_index, shoulder_index in ((7, 9, 5), (8, 10, 6)):
                    if not all(
                        cls._visible(actor, index, cls.POSE_CONFIDENCE)
                        for index in (elbow_index, wrist_index, shoulder_index)
                    ):
                        continue
                    wrist = cls._point(actor, wrist_index)
                    old_wrist = old_actor["wrists"].get(wrist_index)
                    if old_wrist is None:
                        continue
                    inside_target = (
                        upper_body[0] <= wrist[0] <= upper_body[2]
                        and upper_body[1] <= wrist[1] <= upper_body[3]
                    )
                    arm_extended = math.dist(cls._point(actor, shoulder_index), wrist) >= frame_diagonal * 0.08
                    fast_motion = math.dist(wrist, old_wrist) >= frame_diagonal * 0.055
                    moved_toward_target = (
                        math.dist(wrist, target_center)
                        <= math.dist(old_wrist, target_center) - frame_diagonal * 0.035
                    )
                    if inside_target and arm_extended and fast_motion and moved_toward_target:
                        return True
        return False

    @classmethod
    def _hand_raised(
        cls, pose: dict[str, Any]
    ) -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
        """Return wrists held clearly above the shoulder with the arm extended."""
        x1, y1, x2, y2 = pose["bbox"]
        height = max(1.0, y2 - y1)
        raised = []
        for wrist_index, shoulder_index in ((9, 5), (10, 6)):
            if not all(
                cls._visible(pose, index, cls.POSE_CONFIDENCE)
                for index in (wrist_index, shoulder_index)
            ):
                continue
            wrist = cls._point(pose, wrist_index)
            shoulder = cls._point(pose, shoulder_index)
            above_shoulder = wrist[1] <= shoulder[1] - max(10.0, height * 0.16)
            arm_extended = math.dist(wrist, shoulder) >= max(28.0, height * 0.25)
            if above_shoulder and arm_extended:
                raised.append((wrist_index, wrist, shoulder))
        return raised

    @classmethod
    def _raised_hand_threat(
        cls,
        poses: list[dict[str, Any]],
        previous: list[dict[str, Any]],
        width: int,
        height: int,
        now: float,
        *,
        only_actor_index: int | None = None,
    ) -> bool:
        """Find raised-arm geometry near another person; this does not determine intent."""
        diagonal = math.hypot(width, height)
        for actor_index, actor in enumerate(poses):
            if only_actor_index is not None and actor_index != only_actor_index:
                continue
            raised_hands = cls._hand_raised(actor)
            if not raised_hands:
                continue
            old_actor = cls._matches_previous(actor, previous, now, diagonal * 0.12)
            actor_x, actor_y = actor["center"]
            for target_index, target in enumerate(poses):
                if actor_index == target_index:
                    continue
                if math.dist(actor["center"], target["center"]) > diagonal * 0.45:
                    continue
                target_x, target_y = target["center"]
                to_target = (target_x - actor_x, target_y - actor_y)
                target_norm = math.hypot(*to_target)
                if target_norm < 1.0:
                    continue
                x1, y1, x2, y2 = target["bbox"]
                box_w, box_h = x2 - x1, y2 - y1
                target_zone = (
                    x1 - box_w * 0.10,
                    y1 - box_h * 0.15,
                    x2 + box_w * 0.10,
                    y1 + box_h * 0.80,
                )
                for wrist_index, wrist, shoulder in raised_hands:
                    arm = (wrist[0] - shoulder[0], wrist[1] - shoulder[1])
                    arm_norm = math.hypot(*arm)
                    if arm_norm < 1.0:
                        continue
                    alignment = (
                        arm[0] * to_target[0] + arm[1] * to_target[1]
                    ) / (arm_norm * target_norm)
                    inside_target = (
                        target_zone[0] <= wrist[0] <= target_zone[2]
                        and target_zone[1] <= wrist[1] <= target_zone[3]
                    )
                    near_target = (
                        inside_target
                        or math.dist(wrist, target["center"]) <= diagonal * 0.30
                    )
                    sustained = bool(
                        old_actor and wrist_index in old_actor.get("raised", {})
                    )
                    if alignment >= 0.20 and near_target and (sustained or inside_target):
                        return True
        return False

    def _analyze_poses(self, poses: list[dict[str, Any]], width: int, height: int, now: float) -> None:
        # Missing scans cannot count as evidence, nor can out-of-order timestamps.
        if self._last_scan_at is not None and not (
            0.0 < now - self._last_scan_at <= self.MAX_SCAN_GAP_SEC
        ):
            self._previous_poses = []
            self._fall_candidates.clear()
            self._interaction_candidates.clear()
        self._last_scan_at = now
        diagonal = math.hypot(width, height)

        # Local geometric tracks do not recognize a person's identity.
        matches: dict[int, dict[str, Any]] = {}
        used_previous: set[int] = set()
        distances = sorted(
            (math.dist(pose["center"], old["center"]), index, old_index)
            for index, pose in enumerate(poses)
            for old_index, old in enumerate(self._previous_poses)
            if 0.0 < now - old["time"] <= self.MAX_SCAN_GAP_SEC
            and math.dist(pose["center"], old["center"]) <= diagonal * 0.12
        )
        for _, index, old_index in distances:
            if index not in matches and old_index not in used_previous:
                matches[index] = self._previous_poses[old_index]
                used_previous.add(old_index)

        current: list[dict[str, Any]] = []
        tracked_poses: list[dict[str, Any]] = []
        falls: dict[int, dict[str, Any]] = {}
        for index, pose in enumerate(poses):
            old_pose = matches.get(index)
            if old_pose is None:
                person_id = self._next_person_id
                self._next_person_id += 1
            else:
                person_id = old_pose["person_id"]
            tracked_poses.append({**pose, "person_id": person_id})
            horizontal = self._is_horizontal(pose)
            current.append({
                "person_id": person_id,
                "center": pose["center"],
                "time": now,
                "upright": self._is_upright(pose),
                "wrists": {
                    wrist_index: self._point(pose, wrist_index)
                    for wrist_index in (9, 10)
                    if self._visible(pose, wrist_index, self.POSE_CONFIDENCE)
                },
                "raised": {
                    wrist_index: wrist
                    for wrist_index, wrist, _ in self._hand_raised(pose)
                },
            })
            if not horizontal:
                continue
            candidate = self._fall_candidates.get(person_id)
            if candidate is None:
                # An initially horizontal or unobservable torso is not a fall transition.
                if old_pose is None or not old_pose["upright"]:
                    continue
                candidate = {
                    "type": "fall", "person_id": person_id,
                    "started_at": now, "confirmed": False,
                }
            candidate = {
                **candidate, "center": pose["center"], "captured_at": now,
                "evidence_seconds": now - candidate["started_at"],
            }
            if candidate["evidence_seconds"] >= self.EVIDENCE_SECONDS and not candidate["confirmed"]:
                self._append_event(
                    "fall", "possible fall after upright-to-horizontal transition",
                    candidate["evidence_seconds"], now,
                    person_id=person_id, center=pose["center"],
                )
                candidate["confirmed"] = True
            falls[person_id] = candidate
        # Replacing the map resets absent/negative evidence and per-person latches.
        self._fall_candidates = falls
        # A horizontal pose does not diagnose sleep; no second event for this episode.

        interactions: dict[tuple[str, int, int], dict[str, Any]] = {}
        for actor_index, actor in enumerate(tracked_poses):
            for target_index, target in enumerate(tracked_poses):
                if actor_index == target_index:
                    continue
                pair = [actor, target]
                old_actor = matches.get(actor_index)
                previous = [old_actor] if old_actor is not None else []
                for kind, check in (
                    ("contact", self._arm_contact_candidate),
                    ("raised_hand_threat", self._raised_hand_threat),
                ):
                    # Restrict the existing geometry checks to this directed pair.
                    if not check(pair, previous, width, height, now, only_actor_index=0):
                        continue
                    key = (kind, actor["person_id"], target["person_id"])
                    candidate = self._interaction_candidates.get(key)
                    if candidate is None:
                        candidate = {
                            "type": "aggression", "evidence_kind": kind,
                            "actor_id": actor["person_id"], "target_id": target["person_id"],
                            "started_at": now, "confirmed": False,
                        }
                    candidate = {
                        **candidate, "actor_center": actor["center"],
                        "target_center": target["center"], "captured_at": now,
                        "evidence_seconds": now - candidate["started_at"],
                    }
                    # Brief contact must not be stretched into sustained evidence.
                    if candidate["evidence_seconds"] >= self.EVIDENCE_SECONDS and not candidate["confirmed"]:
                        self._append_event(
                            "aggression", "raised-arm interaction; review context; intent not determined",
                            candidate["evidence_seconds"], now,
                            evidence_kind=kind, actor_id=actor["person_id"],
                            target_id=target["person_id"], actor_center=actor["center"],
                            target_center=target["center"],
                        )
                        candidate["confirmed"] = True
                    interactions[key] = candidate
        self._interaction_candidates = interactions
        self._previous_poses = current
        with self._lock:
            self._captured_at = now
            self._status_candidates = [
                {key: value for key, value in candidate.items() if key != "started_at"}
                for candidate in (*falls.values(), *interactions.values())
            ]

    def _append_event(
        self, event_type: str, details: str, evidence_seconds: float,
        captured_at: float, **identity: Any,
    ) -> None:
        with self._lock:
            self._events.append({
                "type": event_type, "details": details,
                "evidence_seconds": evidence_seconds, "captured_at": captured_at,
                **identity,
            })
