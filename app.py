"""VisionAI Sentinel - CCMS Web Command Center.

Ultra-luxurious, futuristic cyber-command interface (100% Emoji-Free):
- Prominently highlighted START CAMERA button right in the center of the video screen
- High-tech top command ribbon with glowing controls and animated status pips
- Live low-latency video streaming with AI targeting reticles
- Circular SVG countdown ring for the 3-minute batch cycle
- Embedded 10-second incident video player for recorded MP4 clips
- Real-time Compliance Radar checklist
- Searchable audit trail table
"""
from __future__ import annotations

import collections
import csv
import errno
import http.server
import json
import math
import os
import re
import socketserver
import struct
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

print("=" * 65, flush=True)
print("   VISIONAI SENTINEL - CCMS COMMAND CENTER", flush=True)
print("=" * 65, flush=True)
print("Loading server modules...", flush=True)

import cv2
import numpy as np
import requests

print("Server modules loaded. Starting dashboard...", flush=True)
import speaker_alarm
from emergency_action_detector import EmergencyActionDetector
from telegram_listener import TelegramTwoWayListener, get_local_ip
import whatsapp_manager
from object_tracker import ObjectTracker, SNAPSHOT_DIR
from micro_object_ai import PrecisionMicroAI
from vision_detection import (
    DETECTION_DEVICE,
    REFERENCE_REQUIRED_LABELS,
    SECONDARY_REQUIRED_LABELS,
    DetectionStabilizer,
    configure_torch_threads,
    detect_with_detail_tiles,
)
from uploaded_video_analysis import UploadedVideoAnalyzer, validate_analysis_settings

ROOT = Path(__file__).resolve().parent
INCIDENTS_DIR = ROOT / "incidents"
SNAPSHOTS_DIR = SNAPSHOT_DIR  # object tracker saves object crops to incidents/snapshots
LOG_FILE = ROOT / "incident_log.csv"
CONFIG_FILE = ROOT / "telegram_config.json"
INCIDENTS_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)

GLOBAL_OBJECT_TRACKER = ObjectTracker()
GLOBAL_MICRO_AI = PrecisionMicroAI()
GLOBAL_ACTIVE_SEARCH_TARGET: str | None = None
APP_START_TIME = time.time()
TELEGRAM_EMERGENCY_EVENTS: collections.deque[dict[str, Any]] = collections.deque(maxlen=50)
TELEGRAM_EMERGENCY_EVENTS_LOCK = threading.Lock()
INCIDENTS_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
INCIDENTS_CACHE_LOCK = threading.Lock()
RADAR_MONITORS = (
    "PPE Safety Vest",
    "Danger Perimeter",
    "Shift Vigor",
    "Duty Desk",
    "Phone Distraction",
    "Queue Monitor",
    "Fall Watch",
    "Conduct Watch",
    "Inactivity Watch",
    "Loitering Watch",
    "Truck Turnaround",
    "VIP Entrance",
    "Cleanliness Check",
)


def is_search_target_match(label: str, target: str | None) -> bool:
    """Check if detected object label matches the user's active search target."""
    if not target or not label:
        return False
    t = target.lower().strip()
    lbl = label.lower().strip()
    if t == lbl or t in lbl or lbl in t:
        return True
    from object_tracker import ALIASES
    aliases = ALIASES.get(t, [])
    if lbl in aliases:
        return True
    for k, v in ALIASES.items():
        if (t == k or t in k or k in t) and lbl in v:
            return True
    return False

MODEL_NAME = "yolo26n.pt"
PORT = int(os.environ.get("VISIONAI_PORT", "8080"))

CLASS_PERSON = 0
CLASS_CAR = 2
CLASS_BUS = 5
CLASS_TRUCK = 7
CLASS_BACKPACK = 24
CLASS_HANDBAG = 26
CLASS_BOTTLE = 39
CLASS_PHONE = 67
MONITORED_CLASSES = [CLASS_PERSON, CLASS_CAR, CLASS_BUS, CLASS_TRUCK, CLASS_BACKPACK, CLASS_HANDBAG, CLASS_BOTTLE, CLASS_PHONE]

# Palette (BGR)
COLOR_BG = (16, 20, 26)
COLOR_BORDER = (45, 55, 72)
COLOR_TEXT_PRIMARY = (235, 240, 245)
COLOR_TEXT_MUTED = (140, 155, 175)
COLOR_CYAN = (255, 210, 0)
COLOR_GOLD = (35, 195, 255)
COLOR_SUCCESS = (80, 225, 60)
COLOR_WARNING = (0, 175, 255)
COLOR_DANGER = (40, 40, 245)

# Live detection overlay palette (BGR) - one color per object category
DETECTION_CATEGORY_COLORS = {
    "person": (94, 197, 34),       # #22c55e green
    "vehicle": (11, 158, 245),     # #f59e0b amber
    "electronics": (212, 182, 6),  # #06b6d4 cyan
    "furniture": (246, 92, 139),   # #8b5cf6 violet
    "animal": (153, 72, 236),      # #ec4899 pink
    "kitchen": (22, 204, 132),     # #84cc16 lime
    "hazard": (68, 68, 239),       # #ef4444 red
    "accessory": (233, 165, 14),   # #0ea5e9 sky
    "default": (184, 163, 148),    # #94a3b8 slate
}

DETECTION_CATEGORY_LABELS = {
    "person": {"person", "people", "human"},
    "vehicle": {
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    },
    "electronics": {
        "cell phone", "laptop", "tv", "keyboard", "mouse", "remote", "computer monitor",
        "tablet", "camera", "microwave", "oven", "toaster", "refrigerator", "clock",
    },
    "furniture": {
        "chair", "couch", "bed", "dining table", "bench", "potted plant", "toilet",
        "sink", "book", "vase",
    },
    "animal": {
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    },
    "kitchen": {
        "bottle", "wine glass", "cup", "fork", "spoon", "bowl", "banana", "apple",
        "orange", "broccoli", "carrot", "sandwich", "hot dog", "pizza", "donut", "cake",
    },
    "hazard": {
        "knife", "scissors", "baseball bat", "baseball glove", "sports ball",
        "fire hydrant", "stop sign", "parking meter", "skateboard", "surfboard",
    },
    "accessory": {
        "backpack", "umbrella", "handbag", "tie", "suitcase", "sunglasses", "wallet",
        "keys", "pen", "pencil",
    },
}


def load_telegram_config() -> dict | None:
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if data.get("bot_token") and data.get("chat_id"):
            return data
    except Exception:
        return None
    return None


def send_telegram_video_clip(config: dict, video_path: str, caption: str) -> tuple[bool, str]:
    if not config:
        return False, "Telegram not configured"

    token = config["bot_token"]
    chat_id = config["chat_id"]
    url_video = f"https://api.telegram.org/bot{token}/sendVideo"

    try:
        with open(video_path, "rb") as vf:
            files = {"video": ("incident_clip.mp4", vf, "video/mp4")}
            data = {"chat_id": chat_id, "caption": caption, "supports_streaming": True}
            resp = requests.post(url_video, data=data, files=files, timeout=35)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("ok"):
                return True, "Telegram 3-minute video clip delivered successfully"

        # Fallback to sendDocument
        url_doc = f"https://api.telegram.org/bot{token}/sendDocument"
        with open(video_path, "rb") as vf:
            files_doc = {"document": ("incident_clip.mp4", vf, "video/mp4")}
            resp2 = requests.post(url_doc, data={"chat_id": chat_id, "caption": caption}, files=files_doc, timeout=35)
            if resp2.json().get("ok"):
                return True, "Telegram video clip delivered (document format)"
            return False, payload.get("description", "Error sending video")
    except Exception as exc:
        return False, f"Telegram delivery failed ({type(exc).__name__})"


EMERGENCY_VOICE_MESSAGES = {
    "fall": "Emergency. A person has fallen to the floor. Medical assistance is required immediately.",
    "aggression": "Emergency. Aggressive behaviour detected. Security assistance is required immediately.",
    "sleeping": "Health alert. A person is lying motionless on the floor. Please check on them immediately.",
}


def dispatch_camera_emergency(event_type: str, details: str) -> None:
    handle_two_way_event("CAMERA EMERGENCY", f"{event_type}: {details}")
    speaker_alarm.trigger_two_way_alert(
        f"Safety alert. {details} Please verify the scene immediately.",
        siren_duration=6.0,
    )


def dispatch_camera_violation(title: str, details: str) -> None:
    """Warn locally once; the recorded clip's caption is the Telegram message."""
    handle_two_way_event("CAMERA VIOLATION", f"{title}: {details}")
    speaker_alarm.trigger_two_way_alert(
        f"Warning. {title}. {details} Please verify the scene.",
        siren_duration=6.0,
    )


def append_incident_log(timestamp: str, event: str, details: str, video_file: str) -> None:
    exists = LOG_FILE.exists()
    try:
        with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "event", "details", "video_clip"])
            writer.writerow([timestamp, event, details, video_file])
    except OSError as exc:
        print(f"Could not write incident log: {exc}")


def latest_incident_clip() -> dict[str, str] | None:
    """Return the newest non-quarantined incident clip, if any."""
    if not INCIDENTS_DIR.exists():
        return None
    for path in sorted(INCIDENTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
        if GLOBAL_OBJECT_TRACKER.is_object_clip_quarantined(path.name):
            continue
        parts = path.stem.split("_", 3)
        return {
            "filename": path.name,
            "title": parts[3].replace("_", " ").title() if len(parts) >= 4 else path.stem,
        }
    return None


def summarize_incident_log(hours: float = 1.0) -> str:
    """Summarize incident_log.csv entries within the last N hours."""
    if not LOG_FILE.exists():
        return f"No incidents logged in the last {hours:g} hour(s). The bot has stayed quiet."
    cutoff = datetime.now() - timedelta(hours=hours)
    counts: dict[str, int] = {}
    rows: list[tuple[str, str, str]] = []
    try:
        with LOG_FILE.open("r", newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                stamp = (row.get("timestamp") or "").strip()
                try:
                    when = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                if when < cutoff:
                    continue
                event = (row.get("event") or "EVENT").strip() or "EVENT"
                counts[event] = counts.get(event, 0) + 1
                rows.append((stamp[11:], event, (row.get("details") or "").strip()))
    except OSError as exc:
        return f"Could not read the incident log: {exc}"
    if not rows:
        return f"No incidents logged in the last {hours:g} hour(s). The bot has stayed quiet."
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:4]
    summary = ", ".join(f"{name} x{count}" for name, count in top)
    latest = rows[-3:][::-1]
    latest_text = "; ".join(
        f"{time_text} {event}" + (f" ({details[:70]})" if details else "")
        for time_text, event, details in latest
    )
    return (
        f"Last {hours:g} hour(s): {len(rows)} event(s). Top: {summary}. "
        f"Most recent: {latest_text}."
    )


def camera_status_summary() -> str:
    """Build a live status sentence for the chat assistant."""
    cam = GLOBAL_CAMERA
    parts = [
        f"Camera is {'ACTIVE' if cam.is_running else 'STANDBY'} on index {cam.camera_index} — "
        f"{cam.fps:.1f} FPS, inference {cam.inference_ms:.0f} ms."
    ]
    if cam.last_error:
        parts.append(f"Last error: {cam.last_error}.")
    violations = cam.last_eval.get("active_violations", [])
    if violations:
        parts.append("Active alerts: " + ", ".join(violations) + ".")
    else:
        parts.append("No active violations right now.")
    radar = cam.last_eval.get("radar_status", {}) if cam.is_running else {}
    flagged = []
    for name, value in radar.items():
        state = value[0] if isinstance(value, (list, tuple)) and value else str(value)
        level = value[1] if isinstance(value, (list, tuple)) and len(value) > 1 else ""
        if level in ("danger", "warning"):
            flagged.append(f"{name} {state}")
    if flagged:
        parts.append("Watch flags: " + ", ".join(flagged[:5]) + ".")
    engine = cam.engine
    if cam.is_running and engine.cycle_start_time is not None:
        remaining = max(0.0, engine.interval_sec - (time.time() - engine.cycle_start_time))
        parts.append(f"Next auto-check in {remaining:.0f} s.")
    if engine.pending_encode:
        parts.append("An incident clip is still encoding.")
    elif engine.pending_video_path is not None:
        parts.append("An incident clip is ready to send.")
    remaining_mute = speaker_alarm.mute_remaining()
    if remaining_mute > 0:
        parts.append(f"Siren muted for another {remaining_mute / 60:.1f} min.")
    return " ".join(parts)


def save_live_snapshot() -> tuple[str, str]:
    """Save the latest annotated frame; returns (relative_url, message)."""
    with GLOBAL_CAMERA.lock:
        jpeg = GLOBAL_CAMERA.latest_jpeg
    if not jpeg:
        return "", "There is no live frame yet, so a snapshot could not be saved. Start the camera and retry."
    name = f"snap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    try:
        (SNAPSHOTS_DIR / name).write_bytes(jpeg)
    except OSError as exc:
        return "", f"Snapshot could not be saved: {exc}"
    return f"/snapshot/{name}", f"Snapshot saved from the live camera ({len(jpeg) // 1024} KB)."


def draw_corner_reticle(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], thickness: int = 1, corner_len: int = 14) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    c_thick = thickness + 2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, c_thick)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, c_thick)
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, c_thick)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, c_thick)
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, c_thick)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, c_thick)
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, c_thick)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, c_thick)


def detection_box_color(label: str, class_id: int = -1) -> tuple[int, int, int]:
    clean_label = (label or "").strip().lower()
    for category, labels in DETECTION_CATEGORY_LABELS.items():
        if clean_label in labels:
            return DETECTION_CATEGORY_COLORS[category]
    if class_id == CLASS_PERSON:
        return DETECTION_CATEGORY_COLORS["person"]
    return DETECTION_CATEGORY_COLORS["default"]


def draw_detection_box(frame: np.ndarray, detection: dict[str, Any]) -> None:
    """Draw a category-colored rectangle with a readable label for one detection."""
    x1, y1, x2, y2 = (int(value) for value in detection.get("bbox", (0, 0, 0, 0)))
    if x2 <= x1 or y2 <= y1:
        return
    label = detection.get("display_name") or (detection.get("label") or "").replace("_", " ").title()
    confidence = float(detection.get("conf") or 0.0)
    color = detection_box_color(detection.get("label", ""), int(detection.get("class_id", -1)))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence * 100:.0f}%" if confidence > 0 else str(label)
    font_scale = 0.42
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    label_y = y1 - 6
    if label_y - text_h - baseline < 0:
        label_y = min(frame.shape[0] - baseline - 1, y2 + text_h + 8)
    bg_top = max(0, label_y - text_h - baseline - 3)
    bg_right = min(frame.shape[1], x1 + text_w + 10)
    cv2.rectangle(frame, (x1, bg_top), (bg_right, min(frame.shape[0], label_y + 3)), color, -1)
    cv2.putText(
        frame, text, (x1 + 5, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BG, 1, cv2.LINE_AA,
    )


def qt_faststart(in_path: Path) -> bool:
    """Reorder MP4 atoms so moov is placed before mdat for instant web browser playback."""
    try:
        with in_path.open("rb") as f:
            data = f.read()
        pos = 0
        atoms = []
        while pos + 8 <= len(data):
            size, btype = struct.unpack(">I4s", data[pos : pos + 8])
            if size == 1:
                if pos + 16 > len(data):
                    break
                size, = struct.unpack(">Q", data[pos + 8 : pos + 16])
            elif size == 0:
                size = len(data) - pos
            atoms.append((btype, pos, size))
            pos += size

        moov = [a for a in atoms if a[0] == b"moov"]
        mdat = [a for a in atoms if a[0] == b"mdat"]
        if not moov or not mdat:
            return False
        if moov[0][1] < mdat[0][1]:
            return True  # already faststart

        moov_data = bytearray(data[moov[0][1] : moov[0][1] + moov[0][2]])
        moov_len = len(moov_data)

        # Patch stco (32-bit chunk offsets) inside moov
        idx = 0
        while True:
            pos_stco = moov_data.find(b"stco", idx)
            if pos_stco == -1:
                break
            idx = pos_stco + 4
            count, = struct.unpack(">I", moov_data[pos_stco + 8 : pos_stco + 12])
            offset_pos = pos_stco + 12
            for _ in range(count):
                cur_offset, = struct.unpack(">I", moov_data[offset_pos : offset_pos + 4])
                struct.pack_into(">I", moov_data, offset_pos, cur_offset + moov_len)
                offset_pos += 4

        # Patch co64 (64-bit chunk offsets) inside moov if present
        idx = 0
        while True:
            pos_co64 = moov_data.find(b"co64", idx)
            if pos_co64 == -1:
                break
            idx = pos_co64 + 4
            count, = struct.unpack(">I", moov_data[pos_co64 + 8 : pos_co64 + 12])
            offset_pos = pos_co64 + 12
            for _ in range(count):
                cur_offset, = struct.unpack(">Q", moov_data[offset_pos : offset_pos + 8])
                struct.pack_into(">Q", moov_data, offset_pos, cur_offset + moov_len)
                offset_pos += 8

        out = bytearray()
        for a in atoms:
            if a[0] == b"moov":
                continue
            if a[0] == b"mdat":
                out.extend(moov_data)
                out.extend(data[a[1] : a[1] + a[2]])
            else:
                out.extend(data[a[1] : a[1] + a[2]])

        tmp_path = in_path.with_suffix(".fast.mp4")
        with tmp_path.open("wb") as f:
            f.write(out)
        tmp_path.replace(in_path)
        return True
    except Exception as exc:
        print(f"[qt_faststart warning] {exc}")
        return False


def save_browser_h264_clip(video_path: Path, frames: list[np.ndarray], fps: float = 15.0) -> bool:
    """Save frames to MP4 using true H.264 video codec with faststart for HTML5 web playback."""
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    written = False

    # 1. Windows Media Foundation H.264 (hardware/OS native, browser compatible)
    if sys.platform.startswith("win"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*"H264")
            out = cv2.VideoWriter(str(video_path), cv2.CAP_MSMF, fourcc, fps, (w, h))
            if out.isOpened():
                for f in frames:
                    out.write(f)
                out.release()
                if video_path.exists() and video_path.stat().st_size > 1000:
                    written = True
        except Exception as exc:
            print(f"[MSMF VideoWriter] {exc}")

    # 2. FFMPEG avc1
    if not written:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            if out.isOpened():
                for f in frames:
                    out.write(f)
                out.release()
                if video_path.exists() and video_path.stat().st_size > 1000:
                    written = True
        except Exception as exc:
            print(f"[avc1 VideoWriter] {exc}")

    # 3. Fallback mp4v
    if not written:
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            if out.isOpened():
                for f in frames:
                    out.write(f)
                out.release()
                check = cv2.VideoCapture(str(video_path))
                written = check.isOpened() and check.read()[0]
                check.release()
        except Exception as exc:
            print(f"[mp4v VideoWriter] {exc}")

    if not written and video_path.exists():
        try:
            video_path.unlink()
        except OSError:
            pass

    # Move moov atom to the front for instant streaming
    if written and video_path.exists():
        qt_faststart(video_path)

    return written


# Object searches use the same validated browser-compatible MP4 encoder as
# incident footage, but build the frames from the live target's own detections.
GLOBAL_OBJECT_TRACKER.clip_writer = save_browser_h264_clip


CRITICAL_AUTO_DISPATCH_TITLES = {
    "Sudden Fall and Emergency Collapse",
    "Raised Hand / Aggressive Arm Gesture",
    "Person Sleeping Condition",
}
INFORMATIONAL_CLIP_TITLES = {"VIP Welcoming Alert", "Truck Unloading Turnaround"}
EMERGENCY_ALERT_COOLDOWN_SEC = 120.0


class ViolationDeliveryQueue:
    def __init__(self, path: Path, interval_sec: float = 180.0) -> None:
        self.path = path
        self.interval_sec = interval_sec
        self.next_batch = time.monotonic() + interval_sec
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._sending = False
        self._pending: dict[str, dict[str, str]] = {}
        self._delivered: set[str] = set()
        self._event_refs: dict[str, dict[str, Any]] = {}
        self.last_result = "No batch sent"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._pending = data.get("pending", {})
            self._delivered = set(data.get("delivered", []))

    def _save_locked(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "pending": self._pending,
            "delivered": sorted(self._delivered),
        }), encoding="utf-8")
        temporary.replace(self.path)

    def enqueue(self, video_path: Path, caption: str, event: dict[str, Any] | None = None) -> bool:
        if video_path.parent.resolve() != INCIDENTS_DIR.resolve() or not video_path.is_file():
            return False
        key = video_path.name
        with self._lock:
            if event is not None:
                event["clip_sent"] = key in self._delivered
                event["delivery_status"] = "sent" if key in self._delivered else "queued"
                self._event_refs[key] = event
            if key in self._pending or key in self._delivered:
                return False
            self._pending[key] = {"filename": key, "caption": caption}
            self._save_locked()
        return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "pending_count": len(self._pending),
                "sending": self._sending,
                "remaining_seconds": max(0.0, self.next_batch - time.monotonic()),
                "interval_seconds": self.interval_sec,
                "last_result": self.last_result,
            }

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self.next_batch = time.monotonic() + self.interval_sec
        self._worker = threading.Thread(target=self._run, name="telegram-violation-batches", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(max(0.05, self.next_batch - time.monotonic())):
            self.dispatch(time.monotonic())

    def dispatch(self, now: float, force: bool = False) -> bool:
        with self._lock:
            if self._sending or (not force and now < self.next_batch):
                return False
            self.next_batch = now + self.interval_sec
            batch = {key: item.copy() for key, item in self._pending.items()}
            if not batch:
                return False
            self._sending = True
        threading.Thread(target=self._deliver_batch, args=(batch,), name="telegram-batch-delivery", daemon=True).start()
        return True

    def _deliver_batch(self, batch: dict[str, dict[str, str]]) -> None:
        sent_count = 0
        try:
            listener = _uploaded_video_listener()
            for key, item in batch.items():
                filename = item["filename"]
                path = INCIDENTS_DIR / filename
                sent = False
                try:
                    if listener is not None and Path(filename).name == filename and path.is_file():
                        sent = bool(listener.send_violation_clip(path, item["caption"]))
                except Exception as exc:
                    print(f"[Telegram Batch] Delivery failed ({type(exc).__name__})")
                with self._lock:
                    event = self._event_refs.get(key)
                    if event is not None:
                        event["clip_sent"] = sent
                        event["delivery_status"] = "sent" if sent else "retry_pending"
                    if sent:
                        sent_count += 1
                        self._pending.pop(key, None)
                        self._delivered.add(key)
                        self._event_refs.pop(key, None)
                        self._save_locked()
        except Exception as exc:
            print(f"[Telegram Batch] Batch failed ({type(exc).__name__})")
        finally:
            with self._lock:
                self.last_result = f"{sent_count}/{len(batch)} clips acknowledged; {len(self._pending)} pending"
                self._sending = False


class PeriodicClipEngine:

    def __init__(
        self,
        interval_sec: float = 180.0,
        phone_sec: float = 12.0,
        desk_empty_sec: float = 12.0,
        emergency_callback: Callable[[str, str], None] | None = None,
        violation_callback: Callable[[str, str], None] | None = None,
    ) -> None:
        self.interval_sec = interval_sec
        self.phone_sec = phone_sec
        self.desk_empty_sec = desk_empty_sec

        self.cycle_start_time: float | None = None
        self.cycle_start_stamp = datetime.now().strftime("%H:%M:%S")

        # Captures share immutable images while retaining the confirmation evidence.
        self.pre_buffer: collections.deque[np.ndarray] = collections.deque()
        self._pre_buffer_times: collections.deque[float] = collections.deque()
        self._active_captures: dict[str, dict[str, Any]] = {}
        self._capture_cooldowns: dict[str, float] = {}
        self._trigger_frame_source: np.ndarray | None = None
        self._trigger_frame_copy: np.ndarray | None = None
        self.is_recording_clip = False
        self.frames_needed = 0  # Legacy dashboard field; capture is no longer frame-count based.
        self.recorded_frames: list[np.ndarray] = []
        self.current_incident_meta: dict[str, str] = {}

        # The delivery outbox is separate from the latest local replay.
        self.pending_video_path: Path | None = None
        self.pending_video_meta: dict[str, str] | None = None
        self.pending_encode = False
        self._pending_encode_count = 0
        self._encode_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        existing_clips = (
            sorted(INCIDENTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True)
            if INCIDENTS_DIR.exists()
            else []
        )
        self.latest_saved_clip: str | None = existing_clips[0].name if existing_clips else None

        self.phone_dwell_start: float | None = None
        self.sleep_start: float | None = None
        self.fall_start: float | None = None
        self.empty_desk_start: float | None = None
        self.danger_zone_start: float | None = None

        # Rule 7: Queue Overflow
        self.queue_overflow_start: float | None = None
        # Rule 8: Caregiver Abuse / Conduct
        self.prev_people_boxes: list[tuple[int, int]] = []
        # Rule 9: Inactivity Watch
        self.prev_person_centers: list[tuple[int, int]] = []
        self.stationary_since: float | None = None
        # Rule 10: Night Loitering
        self.loiter_start: float | None = None
        # Rule 11: Truck Unloading Turnaround
        self.truck_bay_start: float | None = None
        self.truck_in_bay: bool = False
        # Rule 12: VIP Welcoming
        self.vip_cooldown: float = 0.0
        self.prev_vip_centers: list[tuple[int, int]] = []
        # Rule 13: Cleanliness Check
        self.cleanliness_start: float | None = None
        self.emergency_callback = emergency_callback
        self.violation_callback = violation_callback
        self.emergency_cooldowns: dict[str, float] = {}
        self.fall_alert_latched = False
        self.confirmation_sec = 12.0
        self._rule_evidence: dict[str, list[dict[str, Any]]] = {}
        self._rule_last_seen: dict[str, float] = {}
        self._episode_latched: set[str] = set()

    def reset_emergency_state(self) -> None:
        self.emergency_cooldowns.clear()
        self.fall_alert_latched = False
        self._rule_evidence.clear()
        self._rule_last_seen.clear()
        self._episode_latched.clear()

    def _observe_rule(self, key: str, subjects: list[dict], now: float) -> float:
        previous = self._rule_evidence.get(key, [])
        last = self._rule_last_seen.get(key)
        if last is None or not 0.0 < now - last <= 2.5:
            previous = []
            self._episode_latched.discard(key)
        used: set[int] = set()
        tracks = []
        for subject in subjects:
            bbox = subject["bbox"]
            center = subject["center"]
            choices = []
            for index, track in enumerate(previous):
                if index in used or track["class_id"] != subject["class_id"]:
                    continue
                old = track["bbox"]
                intersection = max(0, min(old[2], bbox[2]) - max(old[0], bbox[0])) * max(0, min(old[3], bbox[3]) - max(old[1], bbox[1]))
                smallest = min((old[2] - old[0]) * (old[3] - old[1]), (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                distance = math.dist(center, track["center"])
                if intersection / max(1, smallest) >= 0.35 and distance <= max(30, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.35):
                    choices.append((distance, index))
            match = min(choices, default=None)
            if match is not None:
                used.add(match[1])
            tracks.append({
                "bbox": list(bbox), "center": tuple(center), "class_id": subject["class_id"],
                "start": previous[match[1]]["start"] if match is not None else now,
            })
        if not tracks or not used:
            self._episode_latched.discard(key)
        self._rule_last_seen[key] = now
        self._rule_evidence[key] = tracks
        return max((now - track["start"] for track in tracks), default=0.0)

    def _raise_emergency_alert(self, key: str, title: str, details: str, now: float) -> None:
        last_alert = self.emergency_cooldowns.get(key)
        if last_alert is not None and now - last_alert < EMERGENCY_ALERT_COOLDOWN_SEC:
            return
        self.emergency_cooldowns[key] = now
        if self.emergency_callback:
            # dispatch_camera_emergency expects fall/aggression/sleeping keys.
            self.emergency_callback(key, f"{title}: {details}")

    def start_cycle(self, now: float) -> None:
        self.cycle_start_time = now
        self.cycle_start_stamp = datetime.now().strftime("%H:%M:%S")

    def pause_cycle(self) -> None:
        self.cycle_start_time = None

    def trigger_clip_capture(
        self, title: str, details: str, frame: np.ndarray, now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        last_capture = self._capture_cooldowns.get(title)
        if title in self._active_captures or title in self._episode_latched or (
            last_capture is not None and now - last_capture < EMERGENCY_ALERT_COOLDOWN_SEC
        ):
            return
        self._capture_cooldowns[title] = now
        if title in self._rule_evidence:
            self._episode_latched.add(title)

        while self._pre_buffer_times and self._pre_buffer_times[0] < now - self.confirmation_sec:
            self._pre_buffer_times.popleft()
            self.pre_buffer.popleft()
        if self._trigger_frame_source is not frame:
            self._trigger_frame_source = frame
            self._trigger_frame_copy = frame.copy()
        frames = list(self.pre_buffer)
        frame_times = list(self._pre_buffer_times)
        frames.append(self._trigger_frame_copy)
        frame_times.append(now)
        meta = {
            "title": title,
            "details": details,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._active_captures[title] = {
            "frames": frames,
            "times": frame_times,
            "deadline": now + 3.0,
            "meta": meta,
        }
        self.is_recording_clip = True
        self.recorded_frames = frames
        self.current_incident_meta = meta
        print(f"\n[EVENT CAPTURED] {title.upper()} - Recording evidence window and 3s follow-up...")
        if title not in CRITICAL_AUTO_DISPATCH_TITLES and title not in INFORMATIONAL_CLIP_TITLES:
            if self.violation_callback:
                self.violation_callback(title, details)

    def update_recording(self, frame: np.ndarray, now: float | None = None, finalize: bool = False) -> None:
        now = time.time() if now is None else now
        sample = (
            self._trigger_frame_copy
            if self._trigger_frame_source is frame
            else frame.copy()
        )
        self._trigger_frame_source = None
        self._trigger_frame_copy = None
        self.pre_buffer.append(sample)
        self._pre_buffer_times.append(now)
        while self._pre_buffer_times and self._pre_buffer_times[0] < now - self.confirmation_sec:
            self._pre_buffer_times.popleft()
            self.pre_buffer.popleft()

        completed = []
        for title, capture in list(self._active_captures.items()):
            # A camera stall must not extend the recording past its deadline.
            if capture["frames"][-1] is not sample and capture["times"][-1] < now <= capture["deadline"]:
                capture["frames"].append(sample)
                capture["times"].append(now)
            if finalize or now >= capture["deadline"]:
                completed.append(capture)
                del self._active_captures[title]

        self.is_recording_clip = bool(self._active_captures)
        current = next(iter(self._active_captures.values()), None)
        self.recorded_frames = current["frames"] if current else []
        self.current_incident_meta = current["meta"] if current else {}
        for capture in completed:
            frames = capture["frames"]
            elapsed = capture["times"][-1] - capture["times"][0]
            # Sparse/startup clips can be shorter than ten seconds. Never compress
            # a long real recording into ten seconds by assuming a fixed 15 FPS.
            duration = elapsed if elapsed > 0.0 else 1.0
            meta = capture["meta"].copy()
            meta["duration_seconds"] = f"{duration:.2f}"
            fps = len(frames) / duration
            with self._dispatch_lock:
                self._pending_encode_count += 1
                self.pending_encode = True
            threading.Thread(
                target=self._encode_incident_clip,
                args=(frames, meta, fps),
                name="incident-clip-encoder",
                daemon=True,
            ).start()

    def finalize_recording(self) -> None:
        if self.pre_buffer:
            self.update_recording(self.pre_buffer[-1], now=self._pre_buffer_times[-1], finalize=True)
        self.pre_buffer.clear()
        self._pre_buffer_times.clear()

    def _encode_incident_clip(
        self, frames: list[np.ndarray], meta: dict[str, str], fps: float = 15.0,
    ) -> None:
        """Serialize encoding, not capture or clip delivery, off the camera thread."""
        video_path = None
        try:
            with self._encode_lock:
                # Use capture time, not encoding time, so delayed same-title jobs
                # keep distinct files and the existing dashboard filename format.
                try:
                    event_time = datetime.strptime(meta["timestamp"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, ValueError):
                    event_time = datetime.now()
                time_tag = event_time.strftime("%Y%m%d_%H%M%S")
                raw_title = meta.get("title", "safety_event").lower()
                safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_title).strip("_")[:24]
                video_filename = f"clip_{time_tag}_{safe_name}_{time.time_ns()}.mp4"
                path = INCIDENTS_DIR / video_filename

                if not save_browser_h264_clip(path, frames, fps=fps):
                    print(f"[CLIP ERROR] Could not encode playable video: {video_filename}")
                    return
                video_path = path
                with self._dispatch_lock:
                    self.latest_saved_clip = video_filename
                    if meta.get("title") not in INFORMATIONAL_CLIP_TITLES:
                        self.pending_video_path = path
                        self.pending_video_meta = meta.copy()
                append_incident_log(
                    meta.get("timestamp", ""),
                    meta.get("title", "Safety Event"),
                    meta.get("details", ""),
                    video_filename,
                )
                GLOBAL_OBJECT_TRACKER.invalidate_clip_cache()
                print(f"[CLIP READY] {meta.get('duration_seconds', 'available')}s video saved: {video_filename}\n")
        except Exception as exc:
            print(f"[CLIP ERROR] Incident clip encoding failed: {exc}")
        finally:
            frames.clear()
            with self._dispatch_lock:
                self._pending_encode_count = max(0, self._pending_encode_count - 1)
                self.pending_encode = self._pending_encode_count > 0

        if video_path is not None and meta.get("title") not in INFORMATIONAL_CLIP_TITLES:
            GLOBAL_DELIVERY.enqueue(video_path, self._violation_caption(meta))

    @staticmethod
    def _violation_caption(meta: dict[str, str]) -> str:
        title = meta.get("title", "Safety Violation")
        heading = "SAFETY REVIEW" if title in CRITICAL_AUTO_DISPATCH_TITLES else "VIOLATION REVIEW"
        return (
            f"VISIONAI - {heading}\n"
            f"Event: {title}\n"
            f"Time: {meta.get('timestamp', '')}\n"
            f"Details: {meta.get('details', '')}\n\n"
            f"Attached: {meta.get('duration_seconds', 'available')}-second incident clip.\n"
            "AI screening only; verify the scene and the recorded evidence."
        )

    def dispatch_3min_cycle(self, now: float, config: dict | None, is_manual: bool = False) -> bool:
        return GLOBAL_DELIVERY.dispatch(time.monotonic(), force=is_manual)

    @staticmethod
    def _match_person_centers(
        previous: list[tuple[int, int]],
        current: list[tuple[int, int]],
        max_distance: float,
    ) -> dict[int, tuple[int, int]]:
        candidates = []
        for current_index, (cx, cy) in enumerate(current):
            for px, py in previous:
                distance = math.hypot(cx - px, cy - py)
                if distance <= max_distance:
                    candidates.append((distance, current_index, (px, py)))

        matches: dict[int, tuple[int, int]] = {}
        used_previous: set[tuple[int, int]] = set()
        for _, current_index, previous_center in sorted(candidates):
            if current_index not in matches and previous_center not in used_previous:
                matches[current_index] = previous_center
                used_previous.add(previous_center)
        return matches

    def evaluate_core_rules(
        self,
        frame: np.ndarray,
        boxes_data: list[dict],
        now: float,
        emergency_events: list[dict[str, Any]] | None = None,
        pose_status: str = "not started",
        pose_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        h, w = frame.shape[:2]
        emergency_events = [
            event for event in emergency_events or []
            if float(event.get("evidence_seconds", 0)) >= self.confirmation_sec
            and 0.0 <= now - float(event.get("captured_at", 0)) <= 2.5
        ]
        for box in boxes_data:
            box.pop("is_hazard_breach", None)
            box.pop("is_fall", None)

        # --- Zone Definitions ---
        # Danger Perimeter Zone (right side)
        dz_x1, dz_y1 = int(w * 0.48), int(h * 0.20)
        dz_x2, dz_y2 = int(w * 0.95), int(h * 0.90)

        # Staff / Cashier Desk Zone (left side)
        desk_x1, desk_y1 = int(w * 0.08), int(h * 0.25)
        desk_x2, desk_y2 = int(w * 0.44), int(h * 0.85)

        # Queue Corridor Zone (center-bottom)
        q_x1, q_y1 = int(w * 0.30), int(h * 0.50)
        q_x2, q_y2 = int(w * 0.70), int(h * 0.95)

        # Entrance Threshold Zone (top-center strip for VIP welcoming)
        ent_x1, ent_y1 = int(w * 0.20), 0
        ent_x2, ent_y2 = int(w * 0.80), int(h * 0.15)

        # Truck Bay Zone (bottom-right)
        bay_x1, bay_y1 = int(w * 0.55), int(h * 0.60)
        bay_x2, bay_y2 = int(w * 0.98), int(h * 0.98)

        # Floor-Level Cleanliness Zone (bottom strip)
        fl_x1, fl_y1 = int(w * 0.02), int(h * 0.75)
        fl_x2, fl_y2 = int(w * 0.98), int(h * 0.98)

        # --- Classify detections ---
        people = [b for b in boxes_data if b["class_id"] == CLASS_PERSON]
        phones = [b for b in boxes_data if b["class_id"] == CLASS_PHONE]
        vehicles = [b for b in boxes_data if b["class_id"] in (CLASS_CAR, CLASS_BUS, CLASS_TRUCK)]
        floor_objects = [b for b in boxes_data if b["class_id"] in (CLASS_BACKPACK, CLASS_HANDBAG, CLASS_BOTTLE)]

        # No static zone boxes are drawn on the frame; reticles only appear on actual detected targets/incidents.

        active_violations: list[str] = []
        radar_status: dict[str, tuple[str, str]] = {}

        # ===================================================================
        # RULE 1: PPE & Vest Compliance
        # ===================================================================
        ppe_missing = 0
        for p in people:
            x1, y1, x2, y2 = p["bbox"]
            bw, bh = x2 - x1, y2 - y1
            tx1, tx2 = max(0, x1 + int(bw * 0.15)), min(w, x2 - int(bw * 0.15))
            ty1, ty2 = max(0, y1 + int(bh * 0.15)), min(h, y1 + int(bh * 0.60))

            has_vest = False
            if tx2 > tx1 and ty2 > ty1:
                torso_roi = frame[ty1:ty2, tx1:tx2]
                hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
                m_yellow = cv2.inRange(hsv, np.array([20, 60, 60]), np.array([45, 255, 255]))
                m_orange = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([19, 255, 255]))
                ratio = cv2.countNonZero(m_yellow | m_orange) / float((tx2 - tx1) * (ty2 - ty1))
                has_vest = ratio >= 0.07

            p["has_vest"] = has_vest
            if not has_vest:
                ppe_missing += 1

        if people and ppe_missing > 0:
            radar_status["PPE Safety Vest"] = ("VEST COLOUR CHECK · VERIFY", "warning")
        elif people:
            radar_status["PPE Safety Vest"] = ("HIGH-VIS COLOUR SEEN", "warning")
        else:
            radar_status["PPE Safety Vest"] = ("STANDBY", "muted")

        # ===================================================================
        # RULE 2: Machine Danger Perimeter
        # Only triggers if a person enters and dwells in the hazard zone
        # ===================================================================
        hazard_subjects = 0
        if people:
            for p in people:
                cx, cy = p["center"]
                if dz_x1 <= cx <= dz_x2 and dz_y1 <= cy <= dz_y2:
                    hazard_subjects += 1
                    p["is_hazard_breach"] = True

        dwell = self._observe_rule("Machine Danger Perimeter Breach", [p for p in people if p.get("is_hazard_breach")], now)
        if hazard_subjects > 0:
            if dwell >= self.confirmation_sec:
                active_violations.append("PERIMETER BREACH")
                radar_status["Danger Perimeter"] = (f"BREACH ({hazard_subjects})", "danger")
                self.trigger_clip_capture("Machine Danger Perimeter Breach", f"Person remained in the demo perimeter for {dwell:.1f}s. Verify the site-defined hazard; no machinery control is connected.", frame, now=now)
            else:
                radar_status["Danger Perimeter"] = (f"APPROACH ({dwell:.1f}s)", "warning")
        else:
            self.danger_zone_start = None
            radar_status["Danger Perimeter"] = ("SECURE", "ok")

        # ===================================================================
        # RULE 3: Shift Laziness Tracking (Sleeping on Duty)
        # ===================================================================
        slumped = False
        for p in people:
            cx, cy = p["center"]
            if desk_x1 <= cx <= desk_x2 and desk_y1 <= cy <= desk_y2:
                bw = p["bbox"][2] - p["bbox"][0]
                bh = p["bbox"][3] - p["bbox"][1]
                aspect_ratio = bw / max(1, bh)
                if aspect_ratio >= 0.95 or bh < int(h * 0.25):
                    slumped = True
                    break

        if slumped:
            if self.sleep_start is None:
                self.sleep_start = now
            slump_dur = now - self.sleep_start
            radar_status["Shift Vigor"] = ("POSTURE CHECK · NOT SLEEP DIAGNOSIS", "warning")
        else:
            self.sleep_start = None
            radar_status["Shift Vigor"] = ("ACTIVE", "ok")

        # ===================================================================
        # RULE 4: Unmanned Billing Desk
        # Only count people in the queue corridor as waiting customers.
        # ===================================================================
        waiting_people = [p for p in people if q_x1 <= p["center"][0] <= q_x2 and q_y1 <= p["center"][1] <= q_y2]
        desk_occupied = any(desk_x1 <= p["center"][0] <= desk_x2 and desk_y1 <= p["center"][1] <= desk_y2 for p in people)
        empty_dur = self._observe_rule("Unmanned Billing Desk Alert", waiting_people if not desk_occupied else [], now)
        if not people:
            self.empty_desk_start = None
            radar_status["Duty Desk"] = ("STANDBY", "muted")
        else:
            desk_staff = sum(1 for p in people if desk_x1 <= p["center"][0] <= desk_x2 and desk_y1 <= p["center"][1] <= desk_y2)
            customers = sum(1 for p in people if q_x1 <= p["center"][0] <= q_x2 and q_y1 <= p["center"][1] <= q_y2)
            if desk_staff == 0 and customers >= 1:
                if empty_dur >= max(self.confirmation_sec, self.desk_empty_sec):
                    active_violations.append("UNMANNED DESK")
                    radar_status["Duty Desk"] = (f"VACANT ({empty_dur:.1f}s)", "danger")
                    self.trigger_clip_capture("Unmanned Billing Desk Alert", f"No person observed in the demo desk zone while a person remained in the queue zone for {empty_dur:.1f}s.", frame, now=now)
                else:
                    radar_status["Duty Desk"] = (f"WAITING ({empty_dur:.1f}s)", "warning")
            elif desk_staff > 0:
                self.empty_desk_start = None
                radar_status["Duty Desk"] = (f"OCCUPIED ({desk_staff})", "ok")
            else:
                self.empty_desk_start = None
                radar_status["Duty Desk"] = ("CLEAR", "ok")

        # ===================================================================
        # RULE 5: Counter Phone Addiction
        # ===================================================================
        phone_held = False
        desk_phones = []
        for ph in phones:
            px1, py1, px2, py2 = ph["bbox"]
            pcx, pcy = ph["center"]
            draw_corner_reticle(frame, (px1, py1), (px2, py2), COLOR_DANGER, 1, 8)
            cv2.putText(frame, "DEVICE: PHONE", (px1, max(18, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DANGER, 1)
            for p in people:
                x1, y1, x2, y2 = p["bbox"]
                if (desk_x1 <= p["center"][0] <= desk_x2 and desk_y1 <= p["center"][1] <= desk_y2
                        and x1 <= pcx <= x2 and y1 + (y2 - y1) * 0.15 <= pcy <= y1 + (y2 - y1) * 0.80):
                    phone_held = True
                    desk_phones.append(ph)
                    break

        dwell = self._observe_rule("Sustained Phone Presence at Desk", desk_phones, now)
        if phone_held:
            if dwell >= max(self.confirmation_sec, self.phone_sec):
                active_violations.append("PHONE PRESENCE AT DESK")
                radar_status["Phone Distraction"] = (f"IN USE ({dwell:.1f}s)", "danger")
                self.trigger_clip_capture("Sustained Phone Presence at Desk", f"A phone remained within a person's torso region at the demo desk for {dwell:.1f}s. Verify use and workplace policy; browsing or addiction cannot be inferred.", frame, now=now)
            else:
                radar_status["Phone Distraction"] = (f"IN USE ({dwell:.1f}s)", "warning")
        else:
            self.phone_dwell_start = None
            radar_status["Phone Distraction"] = ("CLEAR", "ok")

        # ===================================================================
        # RULE 6: Queue Overflow Alert (2+ customers in queue corridor)
        # ===================================================================
        in_queue = len(waiting_people)
        q_dur = self._observe_rule("Queue Overflow Alert", waiting_people if in_queue >= 2 else [], now)
        if in_queue >= 2:
            if q_dur >= self.confirmation_sec:
                active_violations.append("QUEUE OVERFLOW")
                radar_status["Queue Monitor"] = (f"OVERFLOW ({in_queue} people)", "danger")
                self.trigger_clip_capture("Queue Overflow Alert", f"At least two people remained in the demo queue corridor for {q_dur:.1f}s. Verify congestion and service context.", frame, now=now)
            else:
                radar_status["Queue Monitor"] = (f"QUEUE ({in_queue})", "warning")
        else:
            self.queue_overflow_start = None
            if in_queue == 1:
                radar_status["Queue Monitor"] = ("1 IN QUEUE", "ok")
            else:
                radar_status["Queue Monitor"] = ("CLEAR", "ok")

        # ===================================================================
        # RULE 7: Sudden Fall Detection
        # A wide bbox alone false-fires on close-up faces, so the emergency is
        # raised only when the pose detector confirms a horizontal torso.
        # The bbox heuristic is kept as a "CHECKING" warning for visibility.
        # ===================================================================
        pose_fall_events = [
            event for event in emergency_events or [] if event.get("type") == "fall"
        ]
        fall_detected = bool(pose_fall_events)
        bbox_horizontal = False
        for p in people:
            bw = p["bbox"][2] - p["bbox"][0]
            bh = p["bbox"][3] - p["bbox"][1]
            aspect_ratio = bw / max(1, bh)
            cy = p["center"][1]
            box_bottom = p["bbox"][3]
            box_width_ratio = bw / max(1, w)
            box_area_ratio = (bw * bh) / max(1, w * h)
            if (aspect_ratio >= 1.65
                    and cy > int(h * 0.65)
                    and box_bottom > int(h * 0.85)
                    and box_width_ratio < 0.50
                    and box_area_ratio < 0.35):
                bbox_horizontal = True
                if fall_detected:
                    p["is_fall"] = True

        if fall_detected:
            if self.fall_start is None:
                self.fall_start = now
            active_violations.append("SUDDEN FALL")
            radar_status["Fall Watch"] = ("POSSIBLE FALL · REVIEW", "danger")
            self.trigger_clip_capture("Sudden Fall and Emergency Collapse", pose_fall_events[0]["details"] + "; 12s continuous pose evidence. Verify the scene.", frame, now=now)
            if not self.fall_alert_latched:
                details = pose_fall_events[0]["details"]
                self._raise_emergency_alert(
                    "fall",
                    "SAFETY REVIEW - POSSIBLE FALL",
                    details,
                    now,
                )
                self.fall_alert_latched = True
        elif bbox_horizontal:
            if self.fall_start is None:
                self.fall_start = now
            radar_status["Fall Watch"] = (f"CHECKING ({now - self.fall_start:.1f}s)", "warning")
            self.fall_alert_latched = False
        else:
            self.fall_start = None
            self.fall_alert_latched = False
            radar_status["Fall Watch"] = ("NORMAL", "ok")

        # Check for prolonged floor sleeping condition
        sleeping_events = [
            event for event in emergency_events or [] if event.get("type") == "sleeping"
        ]
        if sleeping_events:
            active_violations.append("PERSON SLEEPING")
            radar_status["Shift Vigor"] = ("SLEEPING DETECTED", "danger")
            details = sleeping_events[0]["details"]
            self.trigger_clip_capture("Person Sleeping Condition", details, frame)
            self._raise_emergency_alert(
                "sleeping",
                "HEALTH WATCH - PERSON SLEEPING CONDITION",
                details,
                now,
            )

        # ===================================================================
        # RULE 8: Caregiver Abuse / Aggressive Conduct / Raised Hand
        # ===================================================================
        current_centers = [(p["center"][0], p["center"][1]) for p in people]
        center_matches = self._match_person_centers(
            self.prev_people_boxes,
            current_centers,
            max_distance=max(80.0, math.hypot(w, h) * 0.15),
        )
        rapid_interaction = False
        for i, person in enumerate(people):
            previous_center = center_matches.get(i)
            if previous_center is None:
                continue
            cx, cy = current_centers[i]
            displacement = math.hypot(cx - previous_center[0], cy - previous_center[1])
            if displacement <= 45:
                continue
            if any(
                j != i and math.hypot(cx - other["center"][0], cy - other["center"][1]) < 150
                for j, other in enumerate(people)
            ):
                rapid_interaction = True
                break

        self.prev_people_boxes = current_centers

        aggression_events = [
            event for event in emergency_events or [] if event.get("type") == "aggression"
        ]
        if aggression_events:
            active_violations.append("RAISED HAND / AGGRESSION")
            radar_status["Conduct Watch"] = ("RAISED HAND DETECTED", "danger")
            details = aggression_events[0]["details"]
            self.trigger_clip_capture("Raised Hand / Aggressive Arm Gesture", details + "; 12s continuous pose evidence.", frame, now=now)
            self._raise_emergency_alert(
                "aggression",
                "SAFETY REVIEW - SUSTAINED RAISED-ARM INTERACTION",
                details,
                now,
            )
        elif rapid_interaction:
            radar_status["Conduct Watch"] = ("RAPID INTERACTION", "warning")
            radar_status["Conduct Watch"] = ("MOTION CHECK · INTENT UNKNOWN", "warning")
        elif pose_status != "ready":
            radar_status["Conduct Watch"] = (f"POSE {pose_status.upper()[:22]}", "warning")
        else:
            radar_status["Conduct Watch"] = ("NORMAL", "ok")

        # ===================================================================
        # RULE 9: Inactivity Watch (elder motionless > 8s)
        # ===================================================================
        if people:
            inactivity_matches = self._match_person_centers(
                self.prev_person_centers,
                current_centers,
                max_distance=max(80.0, math.hypot(w, h) * 0.15),
            )
            if len(inactivity_matches) == len(current_centers):
                movements = [
                    math.hypot(
                        current_centers[i][0] - previous[0],
                        current_centers[i][1] - previous[1],
                    )
                    for i, previous in inactivity_matches.items()
                ]
                if movements and sum(movements) / len(movements) < 15:
                    if self.stationary_since is None:
                        self.stationary_since = now
                    still_dur = now - self.stationary_since
                    if still_dur >= self.confirmation_sec:
                        radar_status["Inactivity Watch"] = (f"STATIONARY ({still_dur:.1f}s)", "warning")
                    else:
                        radar_status["Inactivity Watch"] = (f"STILL ({still_dur:.1f}s)", "warning")
                else:
                    self.stationary_since = None
                    radar_status["Inactivity Watch"] = ("ACTIVE", "ok")
            else:
                self.stationary_since = None
                radar_status["Inactivity Watch"] = ("TRACKING", "muted")
            self.prev_person_centers = current_centers
        else:
            self.prev_person_centers = []
            self.stationary_since = None
            radar_status["Inactivity Watch"] = ("STANDBY", "muted")

        # ===================================================================
        # RULE 10: Perimeter Loitering (person dwelling in the outer 8% frame edge)
        # ===================================================================
        edge_x, edge_y = int(w * 0.08), int(h * 0.08)
        loiterers = [
            p for p in people
            if p["center"][0] <= edge_x or p["center"][0] >= w - edge_x
            or p["center"][1] <= edge_y or p["center"][1] >= h - edge_y
        ]
        loiter_dur = self._observe_rule("Perimeter Dwell Alert", loiterers, now)
        if loiterers:
            if loiter_dur >= 15.0:
                active_violations.append("PERIMETER DWELL")
                radar_status["Loitering Watch"] = (f"DWELLING ({loiter_dur:.1f}s)", "danger")
                self.trigger_clip_capture("Perimeter Dwell Alert", f"A person remained near the frame edge for {loiter_dur:.1f}s. Time of day and suspicious intent are not inferred.", frame, now=now)
            else:
                radar_status["Loitering Watch"] = (f"PRESENT ({loiter_dur:.1f}s)", "warning")
        else:
            self.loiter_start = None
            radar_status["Loitering Watch"] = ("CLEAR", "ok")

        # ===================================================================
        # RULE 11: Truck Unloading Turnaround (vehicle bay occupancy timer)
        # ===================================================================
        vehicle_in_bay = any(
            bay_x1 <= v["center"][0] <= bay_x2 and bay_y1 <= v["center"][1] <= bay_y2
            for v in vehicles
        )

        if vehicle_in_bay:
            if not self.truck_in_bay:
                self.truck_bay_start = now
                self.truck_in_bay = True
            bay_dur = now - (self.truck_bay_start or now)
            # Draw vehicle bay reticle
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["bbox"]
                draw_corner_reticle(frame, (vx1, vy1), (vx2, vy2), COLOR_GOLD, 1, 10)
                cv2.putText(frame, f"VEHICLE: BAY {bay_dur:.0f}s", (vx1, max(18, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GOLD, 1)
            radar_status["Truck Turnaround"] = (f"IN BAY ({bay_dur:.0f}s)", "warning")
        else:
            if self.truck_in_bay and self.truck_bay_start is not None:
                total_bay_time = now - self.truck_bay_start
                if total_bay_time > 3.0:
                    self.trigger_clip_capture("Truck Unloading Turnaround", f"Vehicle departed bay after {total_bay_time:.0f}s turnaround.", frame)
            self.truck_in_bay = False
            self.truck_bay_start = None
            radar_status["Truck Turnaround"] = ("NO VEHICLE", "ok")

        # ===================================================================
        # RULE 12: Entrance Arrival (person crosses into the top-center threshold)
        # ===================================================================
        vip_centers = [(p["center"][0], p["center"][1]) for p in people]
        vip_matches = self._match_person_centers(
            self.prev_vip_centers,
            vip_centers,
            max_distance=max(80.0, math.hypot(w, h) * 0.15),
        )
        vip_entering = sum(
            1 for i, (cx, cy) in enumerate(vip_centers)
            if i in vip_matches
            and ent_x1 <= cx <= ent_x2
            and ent_y1 <= cy <= ent_y2
            and vip_matches[i][1] > ent_y2
        )
        if vip_entering > 0 and now > self.vip_cooldown:
            radar_status["VIP Entrance"] = (f"ARRIVAL ({vip_entering})", "warning")
            self.trigger_clip_capture("VIP Welcoming Alert", f"{vip_entering} person(s) crossing entrance threshold. Welcome protocol.", frame)
            self.vip_cooldown = now + 30.0  # 30-second cooldown to avoid spam
        elif vip_entering > 0:
            radar_status["VIP Entrance"] = ("COOLDOWN", "muted")
        else:
            radar_status["VIP Entrance"] = ("NO ENTRY", "ok")
        self.prev_vip_centers = vip_centers

        # ===================================================================
        # RULE 13: Cleanliness Check (unaccompanied objects at floor level > 2.5s)
        # ===================================================================
        floor_debris = [
            obj for obj in floor_objects
            if fl_x1 <= obj["center"][0] <= fl_x2
            and fl_y1 <= obj["center"][1] <= fl_y2
            and obj["bbox"][3] >= int(h * 0.88)
            and not any(
                (person["bbox"][0] - 20) <= obj["center"][0] <= (person["bbox"][2] + 20)
                and (person["bbox"][1] - 20) <= obj["center"][1] <= (person["bbox"][3] + 20)
                for person in people
            )
        ]
        clean_dur = self._observe_rule("Unattended Floor Object Alert", floor_debris, now)
        if floor_debris:
            # Draw debris reticles
            for obj in floor_debris:
                ox1, oy1, ox2, oy2 = obj["bbox"]
                draw_corner_reticle(frame, (ox1, oy1), (ox2, oy2), COLOR_WARNING, 1, 8)
                cv2.putText(frame, "FLOOR: DEBRIS", (ox1, max(18, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WARNING, 1)
            if clean_dur >= self.confirmation_sec:
                active_violations.append("UNATTENDED FLOOR OBJECT")
                radar_status["Cleanliness Check"] = (f"DEBRIS ({len(floor_debris)} items)", "danger")
                self.trigger_clip_capture("Unattended Floor Object Alert", f"An unaccompanied detected object remained near floor level for {clean_dur:.1f}s. Verify obstruction; spill or trash is not classified.", frame, now=now)
            else:
                radar_status["Cleanliness Check"] = (f"DETECTED ({clean_dur:.1f}s)", "warning")
        else:
            self.cleanliness_start = None
            radar_status["Cleanliness Check"] = ("CLEAN", "ok")

        # ===================================================================
        # Render Person Reticles with Status Tags
        # ===================================================================
        for p in people:
            x1, y1, x2, y2 = p["bbox"]
            is_fall = p.get("is_fall", False)
            has_vest = p.get("has_vest", True)
            is_hazard = p.get("is_hazard_breach", False)

            if is_fall and self.fall_start and (now - self.fall_start) >= 2.0:
                color = COLOR_DANGER
                tag = "COLLAPSE ALERT"
            elif is_hazard:
                color = COLOR_DANGER
                tag = "HAZARD BREACH"
            elif not has_vest:
                color = COLOR_WARNING
                tag = "PPE CHECK"
            else:
                color = COLOR_SUCCESS
                tag = "SUBJECT: ACTIVE"

            draw_corner_reticle(frame, (x1, y1), (x2, y2), color, 1, 14)
            lbl_w = 175
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lbl_w, y1), (20, 24, 32), -1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lbl_w, y1), color, 1)
            cv2.putText(frame, tag, (x1 + 6, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_PRIMARY, 1)

        for candidate in pose_candidates or []:
            key = "Fall Watch" if candidate["type"] == "fall" else "Conduct Watch"
            duration = float(candidate["evidence_seconds"])
            if candidate["confirmed"]:
                radar_status[key] = ("EVIDENCE CONFIRMED · REVIEW", "danger")
            elif radar_status.get(key, ("", ""))[1] != "danger":
                radar_status[key] = (f"CHECKING {duration:.1f}/12s", "warning")
        return {
            "active_violations": active_violations,
            "radar_status": radar_status,
        }


def handle_two_way_event(event_name: str, details: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_incident_log(stamp, event_name, details, "SPEAKER_AUDIO")
    if event_name in {"TELEGRAM EMERGENCY", "CAMERA EMERGENCY", "CAMERA VIOLATION"}:
        with TELEGRAM_EMERGENCY_EVENTS_LOCK:
            TELEGRAM_EMERGENCY_EVENTS.append({
                "id": time.time_ns(),
                "timestamp": stamp,
                "event": event_name,
                "details": details,
            })
    print(f"\n[TWO-WAY AUDIT LOGGED] {event_name}: {details}")


def set_search_target_from_telegram(target: str) -> None:
    """Invoked when user searches an object via Telegram bot to sync live HUD targeting reticles."""
    global GLOBAL_ACTIVE_SEARCH_TARGET
    clean = target.strip().lower() if target else None
    GLOBAL_ACTIVE_SEARCH_TARGET = clean
    print(f"\n[TWO-WAY AI TELEGRAM] Active HUD search target set to: {clean}")


class CameraManager:
    """Manages the background camera worker thread, YOLO model, and streaming frames."""

    def __init__(self) -> None:
        self.is_running = False
        self.camera_index = 0
        self.confidence = 0.25
        self._inference_frame_count = 0
        self._previous_tile_detections: list[dict[str, Any]] = []
        self._detection_stabilizer = DetectionStabilizer()
        self.model = None
        self._model_lock = threading.Lock()
        self._model_thread_guard = threading.Lock()
        self._model_thread: threading.Thread | None = None
        self.engine = PeriodicClipEngine(
            emergency_callback=dispatch_camera_emergency,
            violation_callback=dispatch_camera_violation,
        )
        self.action_detector = EmergencyActionDetector()
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None
        self.camera_ready_event = threading.Event()
        self.startup_error = ""
        self.last_error = ""
        self.inference_ms = 0.0
        self.latest_jpeg: bytes | None = None
        self.lock = threading.Lock()
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0.0
        self.last_eval: dict[str, Any] = {
            "active_violations": [],
            "radar_status": {name: ("CAMERA OFFLINE", "muted") for name in RADAR_MONITORS},
        }
        self.config = load_telegram_config()
        self.listener: TelegramTwoWayListener | None = None
        if self.config:
            self.listener = TelegramTwoWayListener(
                self.config,
                on_event_callback=handle_two_way_event,
                object_tracker=GLOBAL_OBJECT_TRACKER,
                status_callback=lambda: {
                    "running": self.is_running,
                    "has_frame": self.latest_jpeg is not None,
                    "camera": self.camera_index,
                    "fps": self.fps,
                    "violations": self.last_eval.get("active_violations", []),
                    "active_target": GLOBAL_ACTIVE_SEARCH_TARGET or "",
                },
                set_target_callback=set_search_target_from_telegram,
                camera_control_callback=self._control_camera_from_telegram,
                port=PORT,
            )
            self.listener.start()

    def _control_camera_from_telegram(self, action: str, camera_index: int | None) -> tuple[bool, str]:
        print(f"[Camera Control] Telegram requested camera '{action}' (index={camera_index}).", flush=True)
        if action == "start":
            selected_camera = self.camera_index if camera_index is None else camera_index
            return self.start(selected_camera, self.confidence)
        if action == "stop":
            return self.stop()
        return False, "Unsupported camera action."

    def start(self, camera_index: int = 0, confidence: float = 0.25) -> tuple[bool, str]:
        if self.is_running:
            return True, "Camera already running"

        try:
            camera_index = int(camera_index)
            confidence = float(confidence)
        except (TypeError, ValueError):
            return False, "Camera number and confidence must be numeric."
        if not 0 <= camera_index <= 9:
            return False, "Camera number must be between 0 and 9."
        if not 0.05 <= confidence <= 0.95:
            return False, "Confidence must be between 0.05 and 0.95."

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=6.0)
            if self.thread.is_alive():
                return False, "Previous camera startup is still releasing the device. Wait a moment and retry."

        self.camera_index = camera_index
        self.confidence = confidence
        self._inference_frame_count = 0
        self._previous_tile_detections = []
        self._detection_stabilizer.reset()
        GLOBAL_MICRO_AI.reset()
        GLOBAL_OBJECT_TRACKER.reset_live_capture()
        self.action_detector.reset()
        self.action_detector.start()
        self.engine.reset_emergency_state()
        self.startup_error = ""
        self.last_error = ""
        self.inference_ms = 0.0
        self.camera_ready_event.clear()
        with self.lock:
            self.latest_jpeg = None
        self.fps = 0.0
        self.last_eval = {
            "active_violations": [],
            "radar_status": {name: ("STARTING", "muted") for name in RADAR_MONITORS},
        }
        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        if not self.camera_ready_event.wait(timeout=90.0):
            self.stop()
            return False, "Camera startup timed out because the camera driver did not respond. Close Camera/Teams/Zoom, check the USB cable, then retry."
        if self.startup_error:
            return False, self.startup_error
        self.engine.start_cycle(time.time())
        return True, f"Camera {self.camera_index} detected and started successfully"

    def stop(self) -> tuple[bool, str]:
        self.engine.pause_cycle()
        self.is_running = False
        GLOBAL_MICRO_AI.reset()
        cap = self.cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        t = self.thread
        if t and t.is_alive():
            t.join(timeout=4.0)
        GLOBAL_OBJECT_TRACKER.reset_live_capture(finalize=True)
        self.engine.finalize_recording()
        with self.lock:
            self.latest_jpeg = None
        self.fps = 0.0
        self.inference_ms = 0.0
        self.last_eval = {
            "active_violations": [],
            "radar_status": {name: ("CAMERA OFFLINE", "muted") for name in RADAR_MONITORS},
        }
        return True, "Camera stopped successfully"

    @staticmethod
    def _frame_has_sensor_noise(frame: np.ndarray) -> bool:
        """Catch the severe colored static some webcam backends return on USB."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return True
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.int16)
        neighbor_delta = (
            np.abs(gray[:, 1:] - gray[:, :-1]).mean()
            + np.abs(gray[1:, :] - gray[:-1, :]).mean()
        ) / 2.0
        return float(neighbor_delta) > 52.0

    def _open_camera_capture(self) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
        """Find an available camera index and try Windows backends with MJPG."""
        if sys.platform.startswith("win"):
            backends = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
            ]
        else:
            backends = [(cv2.CAP_ANY, "Automatic")]

        indexes = list(dict.fromkeys([self.camera_index, *range(3)]))
        fallback: tuple[cv2.VideoCapture, np.ndarray, int] | None = None
        for camera_index in indexes:
            for backend, backend_name in backends:
                print(f"[Camera Worker] Trying camera {camera_index} via {backend_name}...")
                cap = cv2.VideoCapture(camera_index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue

                # MJPG reduces USB bandwidth and avoids broken raw YUY2 negotiation.
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                good_frame = None
                noisy_frames = 0
                valid_frames = 0
                for _ in range(3):
                    ok, probe = cap.read()
                    if not ok or probe is None:
                        continue
                    valid_frames += 1
                    noisy_frames += int(self._frame_has_sensor_noise(probe))
                    good_frame = probe

                if valid_frames and noisy_frames <= valid_frames // 2:
                    if fallback:
                        fallback[0].release()
                    self.camera_index = camera_index
                    print(f"[Camera Worker] Camera {camera_index} detected via {backend_name}: {good_frame.shape[1]}x{good_frame.shape[0]}")
                    return cap, good_frame

                if valid_frames:
                    print(f"[Camera Worker] Camera {camera_index} via {backend_name} returned noisy frames")
                    if fallback is None:
                        fallback = (cap, good_frame, camera_index)
                    else:
                        cap.release()
                else:
                    cap.release()

        if fallback:
            print(f"[Camera Warning] No clean feed found; using camera {fallback[2]} as a fallback")
            self.camera_index = fallback[2]
            return fallback[0], fallback[1]
        return None, None

    def _load_model(self) -> None:
        """Build the detector once. Caller must hold _model_lock."""
        configure_torch_threads()
        from ultralytics import YOLO

        model_path = str(ROOT / MODEL_NAME) if (ROOT / MODEL_NAME).is_file() else MODEL_NAME
        model = YOLO(model_path)
        model.add_callback("on_predict_start", configure_torch_threads)
        model.predict(
            source=np.zeros((320, 320, 3), dtype=np.uint8),
            imgsz=512,
            device=DETECTION_DEVICE,
            verbose=False,
        )
        self.model = model

    def _model_worker(self) -> None:
        started = time.perf_counter()
        try:
            with self._model_lock:
                if self.model is not None:
                    return
                print("[Model Loader] Warming the AI stack (first run can take a few minutes)...", flush=True)
                self._load_model()
            self.last_error = ""
            print(f"[Model Loader] AI detection ready in {time.perf_counter() - started:.1f}s.", flush=True)
        except Exception as exc:
            self.last_error = f"AI model failed to load: {exc}"
            print(f"[Model Loader] {self.last_error}. Live video keeps streaming without AI.", flush=True)

    def _ensure_model_loading(self) -> None:
        """Start one background model load at most; preload and the camera worker share it."""
        if self.model is not None:
            return
        with self._model_thread_guard:
            if self._model_thread is not None and self._model_thread.is_alive():
                return
            self._model_thread = threading.Thread(target=self._model_worker, daemon=True, name="model-loader")
            self._model_thread.start()

    def preload_model(self) -> None:
        """Warm the AI stack at server startup so the first camera start has no AI wait."""
        self._ensure_model_loading()
        GLOBAL_MICRO_AI.ensure_started()

    def _publish_frame(self, frame: np.ndarray) -> bytes | None:
        ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok_enc:
            return None
        jpeg_bytes = buf.tobytes()
        with self.lock:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.latest_jpeg = jpeg_bytes
        return jpeg_bytes

    def _worker_loop(self) -> None:
        print(f"\n[Camera Worker] Initializing camera {self.camera_index}...")
        try:
            cap, probe_frame = self._open_camera_capture()
        except Exception as exc:
            self.startup_error = f"Camera initialization failed: {exc}"
            print(f"[Camera Worker] {self.startup_error}")
            self.is_running = False
            self.camera_ready_event.set()
            return
        if cap is None or not cap.isOpened():
            print(f"[Camera Worker] Failed to open camera {self.camera_index}")
            self.startup_error = "No camera returned video frames. Check the USB connection and close Camera/Teams/Zoom, then retry."
            self.is_running = False
            self.camera_ready_event.set()
            return

        if not self.is_running:
            cap.release()
            return

        self.cap = cap
        if probe_frame is not None:
            self._publish_frame(probe_frame)

        # The live feed must not wait on the AI stack; stream now and detect as
        # soon as the background model load finishes.
        self.camera_ready_event.set()
        self._ensure_model_loading()

        fps_val = 0.0
        last_time = time.time()
        last_inference_error = ""
        read_failures = 0

        while self.is_running:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failures += 1
                if read_failures >= 50:
                    self.startup_error = "The camera stopped returning frames. Reconnect the USB camera and start it again."
                    self.last_error = self.startup_error
                    print(f"[Camera Worker] {self.startup_error}")
                    break
                time.sleep(0.05)
                continue
            read_failures = 0

            # Run the whole pipeline on a screen-sized frame. Detection, drawing,
            # recording, and streaming then share one resolution: far cheaper to
            # encode and buffer, with no accuracy loss at the model's 512px input.
            raw_h, raw_w = frame.shape[:2]
            if raw_w > 960 or raw_h > 540:
                scale = min(960.0 / raw_w, 540.0 / raw_h)
                frame = cv2.resize(
                    frame,
                    (max(2, int(raw_w * scale)), max(2, int(raw_h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            now = time.time()
            dt = now - last_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_val = inst_fps if fps_val == 0 else (0.85 * fps_val + 0.15 * inst_fps)
            last_time = now
            self.fps = fps_val

            if self.model is None:
                # Model still warming up in the background: keep the feed live.
                self._publish_frame(frame)
                time.sleep(0.01)
                continue

            try:
                self._inference_frame_count += 1
                detail_scan = self._inference_frame_count % 10 == 0
                inference_started = time.perf_counter()
                detections, tile_detections = detect_with_detail_tiles(
                    self.model,
                    frame,
                    confidence=self.confidence,
                    detail_scan=detail_scan,
                    previous_tile_detections=self._previous_tile_detections,
                )
                self.inference_ms = (time.perf_counter() - inference_started) * 1000.0
                if detail_scan:
                    self._previous_tile_detections = tile_detections
            except Exception as exc:
                self._publish_frame(frame)
                error_message = str(exc)
                self.last_error = error_message
                if error_message != last_inference_error:
                    print(f"[Camera Worker] AI prediction failed; continuing to stream camera frames: {exc}")
                    last_inference_error = error_message
                time.sleep(0.015)
                continue
            last_inference_error = ""
            if not self.is_running:
                break

            boxes_data = list(detections)

            # Secondary Precision Micro-AI Scan (pens, pencils, camera, keys, gadgets)
            micro_items = GLOBAL_MICRO_AI.scan_frame_for_micro_objects(
                frame,
                boxes_data,
                now,
                self.engine.latest_saved_clip,
            )
            reference_items = [
                item for item in micro_items if item.get("source") == "Reference Photo AI"
            ]
            for reference in reference_items:
                boxes_data = [
                    item for item in boxes_data
                    if item.get("label") == "person"
                    or not GLOBAL_MICRO_AI._overlaps(item.get("bbox", []), reference["bbox"])
                ]
            for mb in micro_items:
                if mb.get("source") != "Reference Photo AI" and any(
                    GLOBAL_MICRO_AI._overlaps(mb["bbox"], reference["bbox"])
                    for reference in reference_items
                ):
                    continue
                mx1, my1, mx2, my2 = mb["bbox"]
                mcx, mcy = (mx1 + mx2) // 2, (my1 + my2) // 2
                boxes_data.append({
                    "class_id": -1,
                    "label": mb["label"],
                    "display_name": mb.get("display_name", mb["label"].title()),
                    "bbox": mb["bbox"],
                    "conf": mb.get("conf", 0.88),
                    "center": (mcx, mcy),
                    "is_micro": True,
                    "source": mb.get("source", "Deep Neural Secondary AI"),
                    "reference_margin": mb.get("reference_margin", 0.0),
                })
            boxes_data = [
                item for item in boxes_data
                if (item.get("label") not in REFERENCE_REQUIRED_LABELS
                    or item.get("source") == "Reference Photo AI")
                and (item.get("label") not in SECONDARY_REQUIRED_LABELS
                     or item.get("source") in {"Reference Photo AI", "Open Vocabulary AI"})
            ]
            boxes_data = self._detection_stabilizer.update(boxes_data, now)
            GLOBAL_OBJECT_TRACKER.update_detections(frame, boxes_data, now)
            # Mark every confirmed detection on the live stream with a category-colored box
            for detection in boxes_data:
                draw_detection_box(frame, detection)
            # Highlight active user search target prominently on stream AND in recorded video
            if GLOBAL_ACTIVE_SEARCH_TARGET:
                for b in boxes_data:
                    label = b.get("label", "").lower()
                    if is_search_target_match(label, GLOBAL_ACTIVE_SEARCH_TARGET):
                        tx1, ty1, tx2, ty2 = b["bbox"]
                        tcx, tcy = b["center"]
                        # Draw prominent gold target reticle with crosshair
                        draw_corner_reticle(frame, (tx1, ty1), (tx2, ty2), COLOR_GOLD, 2, 16)
                        cv2.line(frame, (tcx - 8, tcy), (tcx + 8, tcy), COLOR_GOLD, 1)
                        cv2.line(frame, (tcx, tcy - 8), (tcx, tcy + 8), COLOR_GOLD, 1)
                        target_tag = f"TARGET: {label.upper()}"
                        cv2.rectangle(frame, (tx1, max(0, ty1 - 20)), (tx1 + 180, ty1), (15, 23, 42), -1)
                        cv2.rectangle(frame, (tx1, max(0, ty1 - 20)), (tx1 + 180, ty1), COLOR_GOLD, 1)
                        cv2.putText(frame, target_tag, (tx1 + 6, max(14, ty1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GOLD, 1)

            self._publish_frame(frame)

            self.action_detector.submit(frame, now)
            emergency_events = self.action_detector.drain_events()
            pose_snapshot = self.action_detector.status()
            pose_candidates = pose_snapshot["candidates"] if pose_snapshot["captured_at"] is not None and 0 <= now - pose_snapshot["captured_at"] <= 2.5 else []
            evaluation = self.engine.evaluate_core_rules(
                frame,
                boxes_data,
                now,
                emergency_events=emergency_events,
                pose_status=self.action_detector.state,
                pose_candidates=pose_candidates,
            )
            self.last_eval = evaluation
            self.engine.update_recording(frame, now=now)

            time.sleep(0.01)

        cap.release()
        if self.cap is cap:
            self.cap = None
        with self.lock:
            self.latest_jpeg = None
        self.is_running = False
        print("[Camera Worker] Camera released cleanly.")


GLOBAL_CAMERA = CameraManager()
GLOBAL_DELIVERY = ViolationDeliveryQueue(ROOT / "telegram_outbox.json")


_UPLOADED_VIDEO_ALERT_LISTENER: TelegramTwoWayListener | None = None


def _uploaded_video_listener() -> TelegramTwoWayListener | None:
    """Reuse the live camera listener, or build a quiet one for clip delivery."""
    global _UPLOADED_VIDEO_ALERT_LISTENER
    listener = getattr(GLOBAL_CAMERA, "listener", None)
    if listener is not None:
        return listener
    config = load_telegram_config()
    if not config:
        return None
    if _UPLOADED_VIDEO_ALERT_LISTENER is None:
        _UPLOADED_VIDEO_ALERT_LISTENER = TelegramTwoWayListener(config)
    return _UPLOADED_VIDEO_ALERT_LISTENER


def handle_uploaded_video_event(filename: str, event: dict[str, Any]) -> None:
    """Keep historical warnings local and queue their annotated violation clips."""
    event_type = str(event.get("type") or "")
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp = max(0.0, float(event.get("timestamp_seconds", 0.0)))
    minutes, seconds = divmod(int(timestamp), 60)
    hours, minutes = divmod(minutes, 60)
    video_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"
    plate_text = str(event.get("plate_text") or "Unreadable")
    source = f"Uploaded recording {Path(filename).name}"
    clip_name = str(event.get("clip_file") or "")
    details = f"{source}. Video time {video_time}. {str(event.get('details') or '').strip()}".strip()

    if event_type == "plate_read":
        append_incident_log(
            stamp,
            "UPLOADED VIDEO PLATE",
            f"{source}. Video time {video_time}. Plate {plate_text}. "
            f"OCR confidence {float(event.get('confidence', 0.0)) * 100:.0f}%.",
            "-",
        )
        return
    if event_type == "person_absence":
        append_incident_log(stamp, "UPLOADED VIDEO PERSON ABSENCE", details, "-")
        return

    vehicle_type = str(event.get("vehicle_type", "Vehicle")).strip()
    if event_type == "helmetless":
        title = "HELMET VIOLATION"
        violation = f"Motorcycle rider without a helmet (plate: {plate_text})"
    elif event_type == "red_light":
        title = "RED LIGHT VIOLATION"
        violation = f"{vehicle_type} crossed a red light at the stop line (plate: {plate_text})"
    elif event_type == "vehicle_speed" and event.get("overspeed"):
        title = "OVERSPEED"
        violation = (
            f"{vehicle_type} measured {float(event.get('speed_kmh', 0.0)):.1f} km/h "
            f"over the {float(event.get('speed_limit_kmh', 0.0)):.0f} km/h limit (plate: {plate_text})"
        )
    elif event_type == "vehicle_speed":
        append_incident_log(stamp, "UPLOADED VIDEO SPEED", details, clip_name or "-")
        return
    else:
        return

    if event.get("violation_alerted"):
        return
    event["violation_alerted"] = True
    event["clip_sent"] = False
    append_incident_log(stamp, f"UPLOADED VIDEO {title}", details, clip_name or "-")
    dispatch_camera_violation(
        f"Uploaded video (historical review) - {title}",
        f"{violation}. {details}",
    )
    caption = (
        f"VISIONAI - {title}\n"
        f"Event: {title}\n"
        f"Time: {stamp}\n"
        f"Video time: {video_time}\n"
        f"{violation}\n"
        f"Source: {Path(filename).name}\n"
        f"Details: {details}\n"
        "Attached: annotated incident clip. AI screening only; verify the footage."
    )
    clip_path = INCIDENTS_DIR / clip_name if clip_name else None

    event["delivery_status"] = "no_clip"
    if clip_path is not None and clip_path.is_file():
        GLOBAL_DELIVERY.enqueue(clip_path, caption, event=event)


GLOBAL_VIDEO_ANALYZER = UploadedVideoAnalyzer(on_event=handle_uploaded_video_event)
MAX_UPLOADED_VIDEO_BYTES = 200 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


HTML_COMMAND_CENTER = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VisionAI Enterprise // Sentinel Surveillance Platform</title>
  <link rel="icon" type="image/png" href="/logo.png?v=2">
  <link rel="apple-touch-icon" href="/logo.png?v=2">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      /* Zoho Enterprise Professional Design System */
      --zoho-canvas: #090F1F;
      --zoho-header: #0D162C;
      --zoho-ribbon: #111C36;
      --zoho-card: #142140;
      --zoho-card-elevated: #18284C;
      --zoho-border: #23355A;
      --zoho-border-subtle: rgba(255, 255, 255, 0.08);

      --zoho-blue: #0F6FDE;
      --zoho-blue-hover: #0C5CB8;
      --zoho-crimson: #E11D48;
      --zoho-crimson-hover: #BE123C;
      --zoho-emerald: #10B981;
      --zoho-gold: #F59E0B;
      --zoho-slate: #334155;

      --text-white: #FFFFFF;
      --text-primary: #F1F5F9;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;

      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, "SF Mono", monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--zoho-canvas);
      color: var(--text-primary);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* Top Executive Zoho Navigation Header */
    header.zoho-navbar {
      background: var(--zoho-header);
      border-bottom: 1px solid var(--zoho-border);
      padding: 12px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 50;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
    }
    .zoho-brand-group {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .zoho-logo-img {
      height: 42px;
      width: auto;
      object-fit: contain;
      background: #FFFFFF;
      padding: 3px 8px;
      border-radius: 6px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }
    .brand-divider {
      width: 1px;
      height: 32px;
      background: var(--zoho-border);
    }
    .zoho-brand-titles h1 {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-white);
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .zoho-brand-titles p {
      font-size: 11px;
      color: var(--text-secondary);
      font-weight: 500;
      margin-top: 2px;
    }
    .zoho-pill-group {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .zoho-nav-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--zoho-border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--text-secondary);
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
    }
    .status-dot-active { background: var(--zoho-emerald); }
    .status-dot-standby { background: var(--text-muted); }

    /* Zoho Ribbon Action Toolbar */
    .zoho-toolbar {
      background: var(--zoho-ribbon);
      border-bottom: 1px solid var(--zoho-border);
      padding: 14px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }
    .toolbar-left, .toolbar-right {
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .toolbar-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    select.zoho-select {
      background: var(--zoho-card);
      border: 1px solid var(--zoho-border);
      color: var(--text-white);
      padding: 8px 14px;
      border-radius: 6px;
      font-size: 13px;
      font-family: var(--font-sans);
      outline: none;
      cursor: pointer;
    }
    select.zoho-select:focus {
      border-color: var(--zoho-blue);
    }

    /* Zoho Professional Buttons */
    .btn-zoho {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 18px;
      border-radius: 6px;
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
      text-decoration: none;
    }
    .btn-zoho-primary {
      background: var(--zoho-blue);
      color: #FFFFFF;
    }
    .btn-zoho-primary:hover {
      background: var(--zoho-blue-hover);
    }
    .btn-zoho-emergency {
      background: var(--zoho-crimson);
      color: #FFFFFF;
    }
    .btn-zoho-emergency:hover {
      background: var(--zoho-crimson-hover);
    }
    .btn-zoho-secondary {
      background: var(--zoho-card);
      border-color: var(--zoho-border);
      color: var(--text-primary);
    }
    .btn-zoho-secondary:hover {
      background: var(--zoho-card-elevated);
      border-color: rgba(255, 255, 255, 0.2);
    }
    .btn-zoho-stop {
      background: var(--zoho-slate);
      color: #FFFFFF;
    }
    .btn-zoho-stop:hover {
      background: #475569;
    }

    /* Main Responsive Dashboard Container */
    .zoho-container {
      max-width: 1720px;
      margin: 0 auto;
      padding: 24px 32px;
      width: 100%;
      display: grid;
      grid-template-columns: 1fr 420px;
      gap: 24px;
      flex: 1;
    }
    @media (max-width: 1280px) {
      .zoho-container {
        grid-template-columns: 1fr;
      }
    }

    /* Zoho Card Component */
    .zoho-card {
      background: var(--zoho-card);
      border: 1px solid var(--zoho-border);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .zoho-card-header {
      padding: 14px 20px;
      border-bottom: 1px solid var(--zoho-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(255, 255, 255, 0.02);
    }
    .zoho-card-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 14px;
      font-weight: 700;
      color: var(--text-white);
      letter-spacing: -0.01em;
    }
    .zoho-card-body {
      padding: 20px;
    }

    /* Live Surveillance Feed Viewport */
    .stream-viewport {
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #060B17;
      border-radius: 6px;
      overflow: hidden;
      position: relative;
      border: 1px solid var(--zoho-border);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .stream-viewport img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    /* Color key for the live detection boxes */
    .detection-legend {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px 13px;
      margin-top: 10px;
      padding: 0 2px;
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-secondary);
    }
    .detection-legend .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      white-space: nowrap;
    }
    .detection-legend .legend-chip i {
      width: 9px;
      height: 9px;
      border-radius: 2px;
      display: inline-block;
    }
    .hero-launchpad {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      text-align: center;
      padding: 24px;
    }
    .btn-hero-start {
      background: var(--zoho-blue);
      color: #FFFFFF;
      padding: 14px 28px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 700;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 4px 12px rgba(15, 111, 222, 0.35);
      transition: all 0.2s ease;
    }
    .btn-hero-start:hover {
      background: var(--zoho-blue-hover);
      transform: translateY(-1px);
    }

    /* Active Search Target Highlight Banner */
    .active-target-banner {
      background: rgba(245, 158, 11, 0.10);
      border: 1px solid var(--zoho-gold);
      border-radius: 6px;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
      font-size: 12px;
    }
    .target-badge {
      background: var(--zoho-gold);
      color: #000;
      font-weight: 700;
      font-family: var(--font-mono);
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .btn-clear-target {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: var(--text-primary);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-clear-target:hover {
      background: rgba(255, 255, 255, 0.2);
    }

    /* Search Bar Component */
    .zoho-search-row {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .zoho-search-input {
      flex: 1;
      background: var(--zoho-card-elevated);
      border: 1px solid var(--zoho-border);
      color: var(--text-white);
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-family: var(--font-sans);
      outline: none;
    }
    .zoho-search-input:focus {
      border-color: var(--zoho-blue);
      box-shadow: 0 0 0 2px rgba(15, 111, 222, 0.25);
    }
    .zoho-quick-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      align-items: center;
    }
    .quick-pill {
      background: var(--zoho-card-elevated);
      border: 1px solid var(--zoho-border);
      color: var(--text-secondary);
      padding: 5px 12px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .quick-pill:hover {
      border-color: var(--zoho-blue);
      color: var(--text-white);
      background: rgba(15, 111, 222, 0.15);
    }

    /* Object Result Cards */
    .object-results-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }
    .object-card {
      background: var(--zoho-card-elevated);
      border: 1px solid var(--zoho-border);
      border-radius: 6px;
      padding: 14px;
      display: flex;
      gap: 14px;
      transition: border-color 0.15s ease;
    }
    .object-card:hover {
      border-color: var(--zoho-blue);
    }
    .object-thumb {
      width: 90px;
      height: 80px;
      background: #090F1F;
      border: 1px solid var(--zoho-border);
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      flex-shrink: 0;
    }
    .object-thumb img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .object-info {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 4px;
    }
    .object-title {
      font-size: 13px;
      font-weight: 700;
      color: var(--text-white);
    }
    .object-meta {
      font-size: 11px;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      line-height: 1.4;
    }
    .btn-play-clip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: var(--zoho-blue);
      color: #FFFFFF;
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      border: none;
      cursor: pointer;
      margin-top: 4px;
    }
    .btn-play-clip:hover {
      background: var(--zoho-blue-hover);
    }
    .btn-play-clip:disabled {
      background: var(--zoho-slate);
      color: var(--text-secondary);
      cursor: not-allowed;
      opacity: 0.8;
    }

    /* Spoken Warnings and Intercom */
    .preset-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .preset-pill {
      background: var(--zoho-card-elevated);
      border: 1px solid var(--zoho-border);
      color: var(--text-secondary);
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .preset-pill:hover {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text-white);
    }
    .intercom-row {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }

    /* 3-Minute Batch Circular Timer */
    .timer-flex {
      display: flex;
      align-items: center;
      gap: 20px;
    }
    .ring-box {
      position: relative;
      width: 80px;
      height: 80px;
    }
    .ring-box svg {
      width: 100%;
      height: 100%;
      transform: rotate(-90deg);
    }
    .ring-bg {
      stroke: var(--zoho-border);
      stroke-width: 6;
      fill: none;
    }
    .ring-prog {
      stroke: var(--zoho-blue);
      stroke-width: 6;
      stroke-linecap: round;
      fill: none;
      stroke-dasharray: 251.2;
      stroke-dashoffset: 251.2;
      transition: stroke-dashoffset 1s linear;
    }
    .ring-text {
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-family: var(--font-mono);
      font-size: 14px;
      font-weight: 700;
      color: var(--text-white);
    }

    /* Compliance Radar Checklist */
    .radar-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .radar-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 7px 10px;
      border-radius: 4px;
      background: rgba(255, 255, 255, 0.02);
      font-size: 12px;
    }
    .badge {
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 700;
    }
    .badge-ok { background: rgba(16, 185, 129, 0.15); color: var(--zoho-emerald); border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-danger { background: rgba(225, 29, 72, 0.15); color: var(--zoho-crimson); border: 1px solid rgba(225, 29, 72, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: var(--zoho-gold); border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-muted { background: rgba(148, 163, 184, 0.1); color: var(--text-muted); border: 1px solid var(--zoho-border); }

    /* Video Player & Playlist */
    .player-box {
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid var(--zoho-border);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .playlist-scroll {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 280px;
      overflow-y: auto;
    }
    .playlist-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      background: var(--zoho-card-elevated);
      border: 1px solid var(--zoho-border);
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .playlist-item:hover, .playlist-item.active {
      border-color: var(--zoho-blue);
      background: rgba(15, 111, 222, 0.12);
    }

    /* Enterprise Table */
    table.zoho-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    table.zoho-table th {
      background: var(--zoho-card-elevated);
      color: var(--text-secondary);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.04em;
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--zoho-border);
    }
    table.zoho-table td {
      padding: 10px 14px;
      border-bottom: 1px solid var(--zoho-border);
      color: var(--text-primary);
    }
    table.zoho-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(6px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 100;
      padding: 20px;
    }
    .modal-dialog {
      background: var(--zoho-card);
      border: 1px solid var(--zoho-border);
      border-radius: 8px;
      width: 100%;
      max-width: 560px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.4);
      overflow: hidden;
    }
    .clip-modal-backdrop {
      z-index: 120;
    }
    .clip-modal-dialog {
      max-width: 1000px;
    }
    .clip-modal-video {
      display: block;
      width: 100%;
      max-height: 72vh;
      min-height: 240px;
      background: #050914;
      object-fit: contain;
    }
    .clip-modal-empty {
      display: none;
      min-height: 240px;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: #050914;
      color: var(--text-secondary);
      text-align: center;
      font-size: 13px;
    }
    .modal-head {
      padding: 16px 20px;
      border-bottom: 1px solid var(--zoho-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--zoho-header);
    }
    .modal-content {
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .modal-foot {
      padding: 14px 20px;
      border-top: 1px solid var(--zoho-border);
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      background: var(--zoho-header);
    }

    /* VisionAI command-center visual system */
    :root {
      --zoho-canvas: #070b12;
      --zoho-header: #0a111d;
      --zoho-ribbon: #0b1420;
      --zoho-card: #101a28;
      --zoho-card-elevated: #152235;
      --zoho-border: #233346;
      --zoho-blue: #39d6c3;
      --zoho-blue-hover: #24b7a7;
      --zoho-crimson: #fb5574;
      --zoho-crimson-hover: #df3b5d;
      --zoho-emerald: #45e0a1;
      --zoho-gold: #f7bd61;
      --zoho-slate: #334155;
      --text-white: #f7fbff;
      --text-primary: #e6edf5;
      --text-secondary: #9cabbc;
      --text-muted: #718196;
    }

    html { scroll-behavior: smooth; }
    body {
      position: relative;
      background:
        radial-gradient(ellipse at 12% 0%, rgba(35, 177, 166, 0.13), transparent 34rem),
        radial-gradient(ellipse at 96% 28%, rgba(44, 104, 170, 0.11), transparent 34rem),
        var(--zoho-canvas);
      letter-spacing: 0.005em;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      z-index: -1;
      pointer-events: none;
      opacity: 0.16;
      background-image: linear-gradient(rgba(139, 173, 194, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(139, 173, 194, 0.08) 1px, transparent 1px);
      background-size: 56px 56px;
      mask-image: linear-gradient(to bottom, black, transparent 78%);
    }
    header.zoho-navbar {
      min-height: 78px;
      padding: 14px clamp(18px, 3.4vw, 56px);
      background: rgba(8, 15, 25, 0.88);
      border-bottom: 1px solid rgba(115, 155, 177, 0.2);
      backdrop-filter: blur(18px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
    }
    .zoho-logo-img {
      height: 44px;
      border-radius: 10px;
      padding: 5px 9px;
      box-shadow: 0 0 28px rgba(57, 214, 195, 0.13);
    }
    .brand-divider { background: linear-gradient(transparent, rgba(57, 214, 195, 0.55), transparent); }
    .zoho-brand-titles h1 {
      font-size: 17px;
      letter-spacing: -0.035em;
    }
    .zoho-brand-titles p {
      color: #91a4b7;
      letter-spacing: 0.035em;
      text-transform: uppercase;
      font-size: 9px;
    }
    .zoho-pill-group { gap: 8px; }
    .zoho-nav-pill {
      padding: 8px 11px;
      border-radius: 999px;
      border-color: rgba(125, 157, 181, 0.18);
      background: rgba(16, 29, 44, 0.72);
      color: #b7c5d3;
      font-size: 10px;
      white-space: nowrap;
    }
    .status-dot {
      box-shadow: 0 0 12px currentColor;
      background: var(--zoho-emerald);
    }
    .status-dot-standby { background: #64748b; box-shadow: none; }
    .zoho-toolbar {
      position: relative;
      top: auto;
      z-index: 40;
      padding: 11px clamp(18px, 3.4vw, 56px);
      background: rgba(10, 18, 29, 0.86);
      border-bottom: 1px solid rgba(115, 155, 177, 0.16);
      backdrop-filter: blur(18px);
    }
    .toolbar-left, .toolbar-right { gap: 9px; }
    .toolbar-label {
      color: #8497aa;
      font-size: 9px;
      letter-spacing: 0.14em;
    }
    select.zoho-select, .zoho-search-input {
      background-color: #111d2b;
      border-color: #2a3b4e;
      border-radius: 10px;
      color: var(--text-white);
    }
    .btn-zoho {
      min-height: 39px;
      padding: 9px 15px;
      border-radius: 10px;
      letter-spacing: 0.01em;
      transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease, border-color 160ms ease;
    }
    .btn-zoho-primary {
      background: linear-gradient(135deg, #39d6c3, #27b7b0);
      color: #041613;
      box-shadow: 0 7px 22px rgba(39, 183, 176, 0.2);
    }
    .btn-zoho-primary:hover {
      background: linear-gradient(135deg, #56e8d5, #39cfc1);
      box-shadow: 0 10px 28px rgba(39, 183, 176, 0.29);
    }
    .btn-zoho-emergency {
      background: linear-gradient(135deg, #fb5574, #d93459);
      box-shadow: 0 7px 22px rgba(251, 85, 116, 0.16);
    }
    .btn-zoho-emergency:hover { background: linear-gradient(135deg, #ff6a85, #e44767); }
    .btn-zoho-secondary {
      border-color: rgba(132, 160, 183, 0.2);
      background: rgba(19, 32, 48, 0.9);
    }
    .btn-zoho-secondary:hover {
      border-color: rgba(57, 214, 195, 0.55);
      background: rgba(26, 47, 63, 0.95);
    }
    .btn-zoho:hover { transform: translateY(-1px); }
    button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible {
      outline: 2px solid #66ead8;
      outline-offset: 3px;
    }
    .mission-banner {
      position: relative;
      isolation: isolate;
      display: flex;
      min-height: 164px;
      align-items: center;
      justify-content: space-between;
      gap: 28px;
      max-width: 1720px;
      margin: 22px auto 0;
      padding: 25px clamp(20px, 3.4vw, 48px);
      width: calc(100% - clamp(36px, 6.8vw, 112px));
      overflow: hidden;
      border: 1px solid rgba(105, 163, 181, 0.22);
      border-radius: 20px;
      background:
        radial-gradient(ellipse at 78% 50%, rgba(50, 183, 174, 0.16), transparent 20rem),
        linear-gradient(112deg, rgba(16, 31, 45, 0.98), rgba(11, 20, 32, 0.95) 55%, rgba(12, 31, 42, 0.96));
      box-shadow: 0 22px 65px rgba(0, 0, 0, 0.25), inset 0 1px rgba(255, 255, 255, 0.035);
    }
    .mission-banner::before {
      content: "";
      position: absolute;
      z-index: -1;
      width: 340px;
      height: 340px;
      right: 6%;
      top: -225px;
      border: 1px solid rgba(57, 214, 195, 0.14);
      border-radius: 50%;
      box-shadow: 0 0 0 34px rgba(57, 214, 195, 0.025), 0 0 0 68px rgba(57, 214, 195, 0.02);
    }
    .mission-copy { min-width: 0; }
    .mission-eyebrow {
      display: flex;
      align-items: center;
      gap: 9px;
      color: #75d9ce;
      font: 700 10px var(--font-mono);
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    .mission-eyebrow::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--zoho-emerald);
      box-shadow: 0 0 13px rgba(69, 224, 161, 0.8);
    }
    .mission-banner h2 {
      margin-top: 11px;
      color: #f5fbff;
      font-size: clamp(24px, 3vw, 37px);
      line-height: 1.08;
      letter-spacing: -0.055em;
      font-weight: 700;
    }
    .mission-banner p {
      margin-top: 9px;
      color: #9aabba;
      font-size: 13px;
      line-height: 1.55;
    }
    .mission-console {
      display: flex;
      flex: 0 0 auto;
      align-items: center;
      gap: 18px;
      padding: 13px 17px;
      border: 1px solid rgba(119, 170, 183, 0.19);
      border-radius: 14px;
      background: rgba(6, 15, 24, 0.47);
      backdrop-filter: blur(10px);
    }
    .mission-orbit {
      position: relative;
      display: grid;
      width: 58px;
      height: 58px;
      place-items: center;
      border: 1px solid rgba(57, 214, 195, 0.36);
      border-radius: 50%;
      color: #6de6d4;
      background: radial-gradient(circle, rgba(57, 214, 195, 0.15), transparent 68%);
      box-shadow: 0 0 28px rgba(57, 214, 195, 0.12), inset 0 0 18px rgba(57, 214, 195, 0.09);
    }
    .mission-orbit::before, .mission-orbit::after {
      content: "";
      position: absolute;
      inset: 7px -5px;
      border: 1px solid rgba(57, 214, 195, 0.2);
      border-radius: 50%;
      transform: rotate(52deg);
    }
    .mission-orbit::after { transform: rotate(-52deg); }
    .mission-orbit svg { width: 22px; height: 22px; }
    .mission-console-label {
      color: #e2edf3;
      font: 700 10px var(--font-mono);
      letter-spacing: 0.09em;
    }
    .mission-console-sub {
      margin-top: 5px;
      color: #8196a8;
      font: 10px var(--font-mono);
    }
    #telegramEmergencyPanel {
      margin: 16px clamp(18px, 3.4vw, 56px) 0 !important;
      border-radius: 14px !important;
      box-shadow: 0 12px 40px rgba(251, 85, 116, 0.14) !important;
    }
    .zoho-container {
      max-width: 1720px;
      padding: 22px clamp(18px, 3.4vw, 56px) 44px;
      grid-template-columns: minmax(0, 1fr) minmax(320px, 390px);
      gap: 20px;
      align-items: start;
    }
    .zoho-container > div { min-width: 0; }
    .zoho-container > div:first-child, .zoho-container > div:last-child { gap: 18px !important; }
    .zoho-card {
      position: relative;
      border-color: rgba(107, 141, 164, 0.2);
      border-radius: 15px;
      background: linear-gradient(145deg, rgba(17, 29, 44, 0.97), rgba(12, 22, 34, 0.97));
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.2), inset 0 1px rgba(255, 255, 255, 0.035);
      transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    }
    .zoho-card:hover {
      border-color: rgba(91, 178, 181, 0.37);
      box-shadow: 0 17px 42px rgba(0, 0, 0, 0.27), inset 0 1px rgba(255, 255, 255, 0.04);
    }
    .zoho-card-header {
      min-height: 54px;
      padding: 14px 18px;
      border-color: rgba(107, 141, 164, 0.15);
      background: linear-gradient(90deg, rgba(24, 43, 59, 0.52), rgba(16, 29, 43, 0.2));
    }
    .zoho-card-title {
      font-size: 13px;
      letter-spacing: -0.015em;
    }
    .zoho-card-title svg { color: #6ed9ce; }
    .zoho-card-body { padding: 17px; }
    .stream-viewport {
      position: relative;
      aspect-ratio: 16 / 9;
      min-height: 300px;
      border-color: rgba(95, 155, 167, 0.3);
      border-radius: 11px;
      background:
        radial-gradient(ellipse at 50% 48%, rgba(28, 89, 93, 0.22), transparent 44%),
        repeating-linear-gradient(0deg, transparent 0, transparent 39px, rgba(116, 179, 176, 0.035) 40px),
        repeating-linear-gradient(90deg, transparent 0, transparent 39px, rgba(116, 179, 176, 0.035) 40px),
        #07111b;
      box-shadow: inset 0 0 55px rgba(0, 0, 0, 0.42);
    }
    .stream-viewport::after {
      content: "";
      position: absolute;
      inset: 10px;
      pointer-events: none;
      border: 1px solid rgba(111, 201, 191, 0.1);
      border-radius: 7px;
    }
    .stream-viewport img { position: relative; z-index: 1; }
    .hero-launchpad { z-index: 2; }
    .btn-hero-start {
      padding: 15px 25px;
      border-radius: 11px;
      background: linear-gradient(135deg, #39d6c3, #28b6ad);
      color: #051715;
      box-shadow: 0 12px 34px rgba(39, 183, 176, 0.24);
    }
    .btn-hero-start:hover {
      background: linear-gradient(135deg, #60e7d5, #39cfc1);
      box-shadow: 0 15px 40px rgba(39, 183, 176, 0.3);
    }
    .active-target-banner {
      border-radius: 11px;
      background: rgba(247, 189, 97, 0.1);
      border-color: rgba(247, 189, 97, 0.62);
    }
    .target-badge { border-radius: 999px; background: var(--zoho-gold); }
    .zoho-search-row { gap: 11px; }
    .zoho-search-input {
      min-height: 44px;
      padding: 11px 15px;
      border-radius: 11px;
      background: rgba(8, 17, 27, 0.82);
    }
    .zoho-search-input:focus {
      border-color: var(--zoho-blue);
      box-shadow: 0 0 0 3px rgba(57, 214, 195, 0.13);
    }
    .quick-pill, .preset-pill {
      padding: 8px 12px;
      border-radius: 999px;
      border-color: rgba(126, 158, 178, 0.23);
      background: rgba(20, 36, 51, 0.85);
      color: #bdcbd7;
      transition: all 150ms ease;
    }
    .quick-pill:hover, .preset-pill:hover {
      border-color: rgba(57, 214, 195, 0.6);
      background: rgba(57, 214, 195, 0.1);
      color: #a4f5e9;
      transform: translateY(-1px);
    }
    .object-card {
      border-radius: 13px !important;
      border-color: rgba(107, 141, 164, 0.2) !important;
      background: linear-gradient(145deg, rgba(23, 39, 56, 0.96), rgba(15, 27, 41, 0.98)) !important;
    }
    .object-card:hover {
      border-color: rgba(57, 214, 195, 0.45) !important;
      box-shadow: 0 12px 30px rgba(0,0,0,0.24) !important;
    }
    .radar-list { gap: 3px; }
    .radar-row {
      min-height: 39px;
      padding: 8px 10px;
      border-color: rgba(116, 147, 168, 0.12);
      background: rgba(15, 27, 40, 0.58);
    }
    .radar-row:first-child { border-radius: 10px 10px 4px 4px; }
    .radar-row:last-child { border-radius: 4px 4px 10px 10px; }
    .badge { border-radius: 999px; letter-spacing: 0.025em; }
    .ring-box {
      filter: drop-shadow(0 0 18px rgba(57, 214, 195, 0.12));
    }
    .ring-prog { stroke: var(--zoho-blue); filter: drop-shadow(0 0 5px rgba(57, 214, 195, 0.45)); }
    .ring-bg { stroke: rgba(140, 171, 190, 0.14); }
    .ring-text { letter-spacing: -0.04em; }
    .playlist-item {
      padding: 12px 13px;
      border-radius: 10px;
      border-color: rgba(107, 141, 164, 0.16);
      background: rgba(18, 32, 47, 0.88);
    }
    .playlist-item:hover, .playlist-item.active {
      border-color: rgba(57, 214, 195, 0.5);
      background: rgba(57, 214, 195, 0.08);
    }
    .player-box {
      border-radius: 11px;
      border-color: rgba(107, 141, 164, 0.2);
      background: radial-gradient(ellipse at center, #101c29, #050a11 75%);
    }
    table.zoho-table th {
      background: rgba(21, 36, 52, 0.95);
      color: #9eafbf;
    }
    table.zoho-table th:first-child { border-radius: 8px 0 0 0; }
    table.zoho-table th:last-child { border-radius: 0 8px 0 0; }
    table.zoho-table td, table.zoho-table th { border-color: rgba(107, 141, 164, 0.15); }
    .modal-dialog {
      border-color: rgba(119, 165, 181, 0.25);
      border-radius: 16px;
      background: #101a28;
      box-shadow: 0 28px 90px rgba(0,0,0,0.62);
    }
    .modal-head, .modal-foot { background: #0b1420; }
    .clip-modal-video { background: #050a11; }
    @media (max-width: 1280px) {
      .zoho-container { grid-template-columns: minmax(0, 1fr) minmax(300px, 390px); }
    }
    @media (max-width: 980px) {
      .zoho-container { grid-template-columns: minmax(0, 1fr); }
      .zoho-container > div:last-child { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); align-items: start; }
      .zoho-container > div:last-child > .zoho-card:nth-child(2) { grid-row: span 2; }
      .zoho-brand-titles h1, .zoho-brand-titles p { white-space: nowrap; }
      .zoho-pill-group { gap: 5px; }
      .zoho-nav-pill { padding: 7px 8px; font-size: 9px; }
      .zoho-nav-pill:nth-child(3) { display: none; }
    }
    @media (max-width: 760px) {
      .zoho-pill-group { display: none; }
    }
    @media (max-width: 640px) {
      header.zoho-navbar {
        position: relative;
        align-items: flex-start;
        padding: 13px 16px;
      }
      .zoho-brand-group { gap: 10px; }
      .zoho-logo-img { width: 68px; height: 38px; object-fit: contain; }
      .brand-divider { height: 28px; }
      .zoho-brand-titles h1 { font-size: 14px; }
      .zoho-brand-titles p { max-width: 190px; font-size: 8px; line-height: 1.4; white-space: normal; }
      .zoho-pill-group { display: none; }
      .zoho-toolbar {
        position: relative;
        padding: 12px 16px;
        align-items: stretch;
      }
      .toolbar-left, .toolbar-right { width: 100%; }
      .toolbar-left { display: grid; grid-template-columns: 1fr 1fr; }
      .toolbar-label { grid-column: 1 / -1; }
      #cameraSelect { grid-column: 1 / -1; width: 100%; }
      #topToggleBtn, #topSirenBtn { justify-content: center; padding-inline: 8px; font-size: 11px; }
      .toolbar-right { display: grid; grid-template-columns: 1fr 1fr; }
      .toolbar-right .btn-zoho { justify-content: center; padding-inline: 8px; font-size: 11px; }
      .mission-banner {
        width: calc(100% - 28px);
        min-height: unset;
        margin-top: 14px;
        padding: 20px;
        border-radius: 16px;
      }
      .mission-banner h2 { max-width: 390px; font-size: 27px; }
      .mission-banner p { max-width: 300px; font-size: 12px; }
      .mission-console { display: none; }
      #telegramEmergencyPanel { margin: 12px 14px 0 !important; }
      .zoho-container { gap: 14px; padding: 14px 14px 30px; }
      .zoho-container > div:first-child, .zoho-container > div:last-child { gap: 14px !important; }
      .zoho-container > div:last-child { display: flex !important; }
      .zoho-card { border-radius: 13px; }
      .zoho-card-header { gap: 10px; padding: 12px 13px; }
      .zoho-card-title { gap: 8px; font-size: 12px; }
      .zoho-card-body { padding: 13px; }
      .stream-viewport { min-height: 220px; aspect-ratio: 4 / 3; }
      .btn-hero-start { max-width: 100%; padding: 13px 15px; font-size: 11px; }
      .zoho-search-row { align-items: stretch; }
      .zoho-search-row .btn-zoho { padding-inline: 12px; }
      .quick-pill { padding: 7px 10px; font-size: 10px; }
      .mission-copy { padding-right: 0; }
      .modal-backdrop { padding: 12px; }
      .clip-modal-video { min-height: 180px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
    }
    :root {
      --zoho-canvas: #f5f7fb;
      --zoho-header: #ffffff;
      --zoho-ribbon: #ffffff;
      --zoho-card: #ffffff;
      --zoho-card-elevated: #f8fafc;
      --zoho-border: #e4e7ec;
      --zoho-border-subtle: #eaecf0;
      --zoho-blue: #0f6fde;
      --zoho-blue-hover: #0b5fc2;
      --zoho-crimson: #d92d20;
      --zoho-crimson-hover: #b42318;
      --zoho-emerald: #12a36f;
      --zoho-gold: #b7791f;
      --zoho-slate: #475467;
      --text-white: #1d2939;
      --text-primary: #344054;
      --text-secondary: #667085;
      --text-muted: #98a2b3;
    }
    body {
      background: var(--zoho-canvas);
      color: var(--text-primary);
    }
    header.zoho-navbar {
      background: #ffffff;
      border-color: #e4e7ec;
      box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08);
    }
    .zoho-logo-img { box-shadow: none; border: 1px solid #eaecf0; }
    .brand-divider { background: #e4e7ec; }
    .zoho-brand-titles h1 { color: #1d2939; }
    .zoho-brand-titles p { color: #667085; }
    .zoho-nav-pill {
      background: #f8fafc;
      border-color: #e4e7ec;
      color: #475467;
    }
    .status-dot { box-shadow: none; }
    .zoho-toolbar {
      background: #ffffff;
      border-color: #e4e7ec;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .toolbar-label { color: #667085; }
    select.zoho-select, .zoho-search-input {
      background: #ffffff;
      border-color: #d0d5dd;
      color: #344054;
    }
    select.zoho-select option { background: #ffffff; color: #344054; }
    .btn-zoho-primary {
      background: #0f6fde;
      color: #ffffff;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.12);
    }
    .btn-zoho-primary:hover { background: #0b5fc2; }
    .btn-zoho-emergency {
      background: #d92d20;
      color: #ffffff;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.12);
    }
    .btn-zoho-emergency:hover { background: #b42318; }
    .btn-zoho-secondary {
      background: #ffffff;
      border-color: #d0d5dd;
      color: #344054;
    }
    .btn-zoho-secondary:hover {
      background: #f8fafc;
      border-color: #98a2b3;
    }
    .btn-zoho-stop { background: #475467; color: #ffffff; }
    .btn-zoho-stop:hover { background: #344054; }
    button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible {
      outline-color: #0f6fde;
    }
    .mission-banner {
      border-color: #d6e5f7;
      border-radius: 12px;
      background: linear-gradient(110deg, #ffffff 0%, #f1f7ff 72%, #eaf3ff 100%);
      box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06);
    }
    .mission-banner::before {
      border-color: rgba(15, 111, 222, 0.1);
      box-shadow: 0 0 0 34px rgba(15, 111, 222, 0.025), 0 0 0 68px rgba(15, 111, 222, 0.018);
    }
    .mission-eyebrow { color: #0f6fbd; }
    .mission-eyebrow::before { background: #12a36f; box-shadow: none; }
    .mission-banner h2 { color: #1d2939; }
    .mission-banner p { color: #667085; }
    .mission-console {
      border-color: #d6e5f7;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: none;
    }
    .mission-orbit {
      border-color: #c5dcf8;
      color: #0f6fde;
      background: #f1f7ff;
      box-shadow: none;
    }
    .mission-orbit::before, .mission-orbit::after { border-color: #c5dcf8; }
    .mission-console-label { color: #344054; }
    .mission-console-sub { color: #667085; }
    #telegramEmergencyPanel {
      background: #fff4f2 !important;
      border-color: #fecdca !important;
      box-shadow: 0 2px 8px rgba(16, 24, 40, 0.06) !important;
    }
    #emergencyPanelTitle { color: #b42318 !important; }
    .zoho-card {
      border-color: #e4e7ec;
      border-radius: 10px;
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(16, 24, 40, 0.07);
      transition: border-color 150ms ease, box-shadow 150ms ease;
    }
    .zoho-card:hover {
      border-color: #c8d8eb;
      box-shadow: 0 3px 10px rgba(16, 24, 40, 0.08);
    }
    .zoho-card-header {
      border-color: #eaecf0;
      background: #ffffff;
    }
    .zoho-card-title { color: #344054; }
    .zoho-card-title svg { color: #0f6fde; }
    .stream-viewport {
      border-color: #cbd5e1;
      border-radius: 8px;
      background:
        repeating-linear-gradient(0deg, transparent 0, transparent 39px, rgba(148, 163, 184, 0.06) 40px),
        repeating-linear-gradient(90deg, transparent 0, transparent 39px, rgba(148, 163, 184, 0.06) 40px),
        #101828;
      box-shadow: inset 0 0 35px rgba(0, 0, 0, 0.25);
    }
    .stream-viewport::after { border-color: rgba(255, 255, 255, 0.12); }
    .btn-hero-start {
      border-radius: 7px;
      background: #0f6fde;
      color: #ffffff;
      box-shadow: 0 2px 6px rgba(15, 111, 222, 0.22);
    }
    .btn-hero-start:hover {
      background: #0b5fc2;
      box-shadow: 0 3px 9px rgba(15, 111, 222, 0.25);
    }
    .active-target-banner {
      background: #fffaeb;
      border-color: #fedf89;
    }
    .target-badge { background: #fdb022; color: #344054; }
    .btn-clear-target {
      background: #ffffff;
      border-color: #e4e7ec;
      color: #475467;
    }
    .btn-clear-target:hover { background: #f8fafc; }
    .zoho-search-input { border-radius: 7px; }
    .zoho-search-input:focus {
      border-color: #0f6fde;
      box-shadow: 0 0 0 3px rgba(15, 111, 222, 0.12);
    }
    .quick-pill, .preset-pill {
      border-color: #e4e7ec;
      background: #f8fafc;
      color: #475467;
    }
    .quick-pill:hover, .preset-pill:hover {
      border-color: #b2d2f5;
      background: #eff6ff;
      color: #0b5fc2;
    }
    .object-card {
      border-color: #e4e7ec !important;
      border-radius: 8px !important;
      background: #ffffff !important;
    }
    .object-card:hover {
      border-color: #b2d2f5 !important;
      box-shadow: 0 2px 8px rgba(16, 24, 40, 0.07) !important;
    }
    .object-thumb { background: #f2f4f7; border-color: #e4e7ec; }
    .object-title { color: #1d2939; }
    .object-meta { color: #667085; }
    .btn-play-clip {
      border-radius: 6px;
      background: #0f6fde;
      color: #ffffff;
    }
    .btn-play-clip:hover { background: #0b5fc2; }
    .btn-play-clip:disabled {
      background: #f2f4f7;
      border-color: #e4e7ec;
      color: #667085;
    }
    .radar-row {
      border-color: #eaecf0;
      background: #f8fafc;
      color: #475467;
    }
    .ring-box { filter: none; }
    .ring-prog { stroke: #0f6fde; filter: none; }
    .ring-bg { stroke: #e4e7ec; }
    .ring-text { color: #1d2939; }
    .playlist-item {
      border-color: #e4e7ec;
      background: #ffffff;
    }
    .playlist-item:hover, .playlist-item.active {
      border-color: #b2d2f5;
      background: #eff6ff;
    }
    .player-box {
      border-color: #d0d5dd;
      background: #101828;
    }
    table.zoho-table th {
      background: #f8fafc;
      color: #475467;
    }
    table.zoho-table td, table.zoho-table th { border-color: #eaecf0; }
    table.zoho-table tr:hover td { background: #f8fafc; }
    .modal-dialog {
      border-color: #e4e7ec;
      border-radius: 12px;
      background: #ffffff;
      box-shadow: 0 16px 40px rgba(16, 24, 40, 0.2);
    }
    .modal-head, .modal-foot {
      border-color: #eaecf0;
      background: #ffffff;
    }
    .modal-content { color: #344054; }
    .clip-modal-video, .clip-modal-empty { background: #101828; }
    .clip-modal-empty { color: #e4e7ec; }
    .ccms-subnav {
      display: flex;
      position: fixed;
      z-index: 90;
      inset: 0 auto 0 0;
      width: 188px;
      flex-direction: column;
      align-items: stretch;
      gap: 3px;
      padding: 14px 10px;
      border: 0;
      border-right: 1px solid rgba(131, 185, 255, 0.09);
      background: linear-gradient(180deg, #0f1c31 0%, #14243d 42%, #101b2f 100%);
      color: #e8f0ff;
      box-shadow: 6px 0 22px rgba(7, 11, 18, 0.35);
    }
    .ccms-sidebar-brand {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 46px;
      margin: 2px 3px 14px;
      color: #ffffff;
      font-size: 16px;
      font-weight: 750;
    }
    .ccms-sidebar-brand .ccms-brand-tile {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      border-radius: 9px;
      background: linear-gradient(135deg, #1d5dc1 0%, #2f8fe0 55%, #39d6c3 100%);
      color: #ffffff;
      box-shadow: 0 4px 12px rgba(29, 93, 193, 0.45), inset 0 1px 0 rgba(255,255,255,0.25);
    }
    .ccms-sidebar-brand small { display: block; margin-top: 2px; color: #8fa5c2; font-size: 8.5px; font-weight: 450; letter-spacing: .02em; }
    .ccms-sidebar-group {
      display: flex;
      align-items: center;
      gap: 7px;
      margin: 13px 8px 4px;
      color: #6f87a6;
      font-size: 8.5px;
      font-weight: 700;
      letter-spacing: .14em;
      text-transform: uppercase;
    }
    .ccms-sidebar-group::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(131,185,255,0.22), transparent); }
    .ccms-sidebar-group:first-of-type { margin-top: 2px; }
    .ccms-subnav .ccms-nav-item {
      display: flex;
      width: 100%;
      min-height: 46px;
      align-items: center;
      gap: 9px;
      padding: 7px 8px;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      color: #d3deeb;
      text-align: left;
      font: 600 11.5px var(--font-sans);
      cursor: pointer;
      transition: background 140ms ease, border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
    }
    .ccms-subnav .ccms-nav-item:hover {
      border-color: rgba(255,255,255,0.07);
      background: rgba(255,255,255,0.055);
      color: #ffffff;
      transform: translateX(2px);
    }
    .ccms-subnav .ccms-nav-item.active {
      border-color: rgba(120, 180, 255, 0.38);
      background: linear-gradient(135deg, #1d5dc1 0%, #2f7fe0 100%);
      color: #ffffff;
      box-shadow: 0 4px 14px rgba(29, 93, 193, 0.42), inset 0 1px 0 rgba(255,255,255,0.18);
    }
    .ccms-nav-icon {
      display: flex;
      flex: 0 0 auto;
      align-items: center;
      justify-content: center;
      width: 27px;
      height: 27px;
      border-radius: 7px;
      background: rgba(131, 185, 255, 0.10);
      color: var(--ccms-accent, #83b9ff);
      transition: background 140ms ease, color 140ms ease;
    }
    .ccms-nav-item:hover .ccms-nav-icon { background: rgba(131, 185, 255, 0.18); }
    .ccms-nav-item.active .ccms-nav-icon { background: rgba(255,255,255,0.22); color: #ffffff; }
    .ccms-nav-text { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
    .ccms-nav-text small {
      margin-top: 1px;
      color: #8fa5c2;
      font-size: 8.5px;
      font-weight: 500;
      letter-spacing: .01em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .ccms-nav-item:hover .ccms-nav-text small { color: #a9c1dd; }
    .ccms-nav-item.active .ccms-nav-text small { color: #d8e9ff; }
    .ccms-nav-chip {
      flex: 0 0 auto;
      margin-left: auto;
      padding: 1px 5px;
      border-radius: 999px;
      background: rgba(131, 185, 255, 0.16);
      color: #9dc8ff;
      font-size: 7.5px;
      font-weight: 800;
      letter-spacing: .08em;
    }
    .ccms-nav-item.active .ccms-nav-chip { background: rgba(255,255,255,0.25); color: #ffffff; }
    .ccms-sidebar-spacer { flex: 1; }
    .ccms-sidebar-footer {
      margin: 8px 4px 2px;
      padding: 8px 10px;
      border: 1px solid rgba(131,185,255,0.10);
      border-radius: 8px;
      background: rgba(8, 14, 26, 0.5);
      color: #6f87a6;
      font-size: 8.5px;
      font-weight: 550;
      letter-spacing: .04em;
      line-height: 1.5;
    }
    .ccms-sidebar-footer b { color: #9dc8ff; font-weight: 700; }
    body[data-active-app="ccms"] header.zoho-navbar,
    body[data-active-app="ccms"] .zoho-toolbar,
    body[data-active-app="ccms"] #telegramEmergencyPanel,
    body[data-active-app="ccms"] #ccmsKpiStrip,
    body[data-active-app="ccms"] .zoho-container {
      width: auto;
      margin-left: 188px;
    }
    body[data-active-app="ccms"] .mission-banner { display: none; }
    body[data-active-app="ccms"] header.zoho-navbar { min-height: 58px; padding-top: 9px; padding-bottom: 9px; }
    body[data-active-app="ccms"] .zoho-toolbar { min-height: 48px; padding-top: 7px; padding-bottom: 7px; }
    body[data-active-app="ccms"] #telegramEmergencyPanel { margin-right: 18px !important; }
    .ccms-page-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin: 0 18px 0 190px;
      padding: 10px 0 3px;
      color: #1d2939;
    }
    .ccms-page-heading h2 { font-size: 17px; font-weight: 750; letter-spacing: -.025em; }
    .ccms-page-heading p { margin-top: 3px; color: #667085; font-size: 10px; }
    .ccms-operational { display: inline-flex; align-items: center; gap: 7px; color: #16834f; font-size: 10px; }
    .ccms-operational span { width: 7px; height: 7px; border-radius: 50%; background: #1fa36c; box-shadow: 0 0 0 3px #e4f6ec; }
    .ccms-operational[data-state="standby"] { color: #667085; }
    .ccms-operational[data-state="standby"] span { background: #98a2b3; box-shadow: 0 0 0 3px #f2f4f7; }
    .ccms-operational[data-state="connecting"] { color: #b54708; }
    .ccms-operational[data-state="connecting"] span { background: #f79009; box-shadow: 0 0 0 3px #fef0c7; }
    body[data-active-app="ccms"][data-ccms-section="search"] .ccms-page-heading,
    body[data-active-app="ccms"][data-ccms-section="search"] #ccmsKpiStrip { display: none; }
    body[data-active-app="ccms"] #ccmsKpiStrip {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 18px 0;
    }
    .ccms-kpi-card {
      display: flex;
      min-width: 0;
      min-height: 66px;
      align-items: center;
      gap: 11px;
      padding: 10px 12px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: #ffffff;
      box-shadow: 0 1px 3px rgba(16, 24, 40, .06);
    }
    .ccms-kpi-icon {
      display: grid;
      width: 36px;
      height: 36px;
      flex: 0 0 auto;
      place-items: center;
      border-radius: 9px;
      background: #edf5ff;
      color: #1672d3;
    }
    .ccms-kpi-card:nth-child(2) .ccms-kpi-icon { background: #f0edff; color: #7052ca; }
    .ccms-kpi-card:nth-child(3) .ccms-kpi-icon { background: #fff3e8; color: #d97706; }
    .ccms-kpi-card:nth-child(4) .ccms-kpi-icon { background: #eaf8f0; color: #17834f; }
    .ccms-kpi-card:nth-child(5) .ccms-kpi-icon { background: #eaf8f0; color: #17834f; }
    .ccms-kpi-label { color: #667085; font-size: 10px; font-weight: 600; }
    .ccms-kpi-value { margin-top: 2px; color: #1d2939; font-size: 19px; font-weight: 800; line-height: 1.1; }
    .ccms-kpi-note { margin-top: 2px; color: #667085; font-size: 9px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-container {
      display: grid;
      max-width: none;
      grid-template-columns: minmax(0, 1.85fr) minmax(290px, .95fr);
      align-items: start;
      gap: 10px;
      padding: 12px 18px 22px;
    }
    body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-container > div { display: contents !important; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsLiveCard { grid-column: 1; grid-row: 1; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #aiObjectSearchCard { grid-column: 1; grid-row: 2; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard { grid-column: 1; grid-row: 3; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #incidentPlayerCard { grid-column: 1; grid-row: 4; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard { grid-column: 2; grid-row: 1; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard { grid-column: 2; grid-row: 2; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsRadarCard { grid-column: 2; grid-row: 3; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsBatchCard { grid-column: 2; grid-row: 4; }
    body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-card { min-width: 0; border-radius: 8px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-card-header { min-height: 38px; padding: 9px 12px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-card-title { font-size: 12px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsLiveCard .zoho-card-body { padding: 8px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsLiveCard .stream-viewport { max-height: 340px; aspect-ratio: 16 / 8.5; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #aiObjectSearchCard .zoho-card-body { padding: 11px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] .ccms-search-intro { display: none; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #searchResultsContainer {
      grid-template-columns: repeat(4, minmax(0, 1fr));
      max-height: 218px;
      overflow-y: auto;
    }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard .zoho-card-body { max-height: 280px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard table { min-width: 0; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard th:nth-child(3),
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard td:nth-child(3),
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard th:nth-child(4),
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard td:nth-child(4) { display: none; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsRadarCard .zoho-card-body { max-height: 220px; overflow-y: auto; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsBatchCard { display: none; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard .zoho-card-body { padding: 12px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard .preset-group { gap: 5px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard .preset-pill { padding: 6px 8px; font-size: 9px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard .playlist-scroll { max-height: 245px; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard .playlist-item {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 0;
      border-width: 0 0 1px;
      background: #ffffff;
    }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard .playlist-item > div:first-child {
      display: flex;
      min-width: 0;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard .playlist-item > div:first-child > div:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard .playlist-item > div:first-child > div:last-child { flex: 0 0 auto; }
    body[data-active-app="ccms"][data-ccms-section="overview"] #incidentPlayerCard .player-box { min-height: 220px; }
    body[data-active-app="ccms"][data-ccms-section="search"] .ccms-subnav {
      width: 188px;
    }
    body[data-active-app="ccms"][data-ccms-section="overview"] .ccms-subnav,
    body[data-active-app="ccms"][data-ccms-section="search"] .ccms-subnav {
      justify-content: flex-start;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] .zoho-toolbar,
    body[data-active-app="ccms"][data-ccms-section="search"] .mission-banner { display: none; }
    body[data-active-app="ccms"][data-ccms-section="search"] .zoho-container {
      grid-template-columns: minmax(0, 1fr);
      max-width: 1440px;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] .zoho-container > div:first-child {
      min-width: 0;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] .zoho-container > div:first-child > .zoho-card:not(#aiObjectSearchCard),
    body[data-active-app="ccms"][data-ccms-section="search"] .zoho-container > div:last-child {
      display: none !important;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] #aiObjectSearchCard {
      display: flex;
      width: 100%;
      min-height: 58vh;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] #aiObjectSearchCard .zoho-card-body {
      padding: clamp(16px, 3vw, 30px);
    }
    body[data-active-app="ccms"][data-ccms-section="search"] #objectSearchInput {
      min-height: 46px;
      font-size: 14px;
    }
    body[data-active-app="ccms"][data-ccms-section="search"] #searchResultsContainer {
      grid-template-columns: repeat(auto-fill, minmax(min(100%, 340px), 1fr));
      margin-top: 20px;
    }
    .ccms-search-intro {
      margin: 0 0 18px;
      color: #667085;
      font-size: 12px;
      line-height: 1.55;
    }
    @media (max-width: 640px) {
      header.zoho-navbar { background: #ffffff; }
      .zoho-toolbar { background: #ffffff; }
      .mission-banner {
        background: linear-gradient(110deg, #ffffff, #f1f7ff);
        border-radius: 10px;
      }
      .ccms-subnav { padding-inline: 12px; }
      body[data-active-app="ccms"] header.zoho-navbar,
      body[data-active-app="ccms"] .zoho-toolbar,
      body[data-active-app="ccms"] .mission-banner,
      body[data-active-app="ccms"] #telegramEmergencyPanel,
      body[data-active-app="ccms"] #ccmsKpiStrip,
      body[data-active-app="ccms"] .zoho-container { margin-left: 64px; }
      body[data-active-app="ccms"] header.zoho-navbar { padding-right: 10px; padding-left: 10px; }
      body[data-active-app="ccms"] .zoho-brand-titles p { display: none; }
      .ccms-subnav { width: 64px; align-items: center; padding: 10px 5px; }
      .ccms-sidebar-brand { justify-content: center; margin-inline: 0; }
      .ccms-sidebar-brand span, .ccms-sidebar-group, .ccms-sidebar-footer { display: none; }
      .ccms-subnav .ccms-nav-item { width: 44px; min-height: 40px; justify-content: center; overflow: hidden; padding: 6px 4px; font-size: 0; }
      .ccms-nav-text { display: none; }
      .ccms-nav-chip { display: none; }
      body[data-active-app="ccms"] #ccmsKpiStrip { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; padding: 8px 9px 0; }
      .ccms-kpi-card { min-height: 58px; gap: 7px; padding: 8px; }
      .ccms-kpi-icon { width: 30px; height: 30px; }
      .ccms-kpi-value { font-size: 16px; }
      .ccms-page-heading { margin-inline: 9px; }
      body[data-active-app="ccms"] .ccms-page-heading { margin-left: 73px; margin-right: 9px; }
      .ccms-page-heading h2 { font-size: 14px; }
      body[data-active-app="ccms"][data-ccms-section="overview"] .zoho-container { grid-template-columns: minmax(0,1fr); gap: 9px; padding: 9px; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsLiveCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #aiObjectSearchCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #incidentPlayerCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard,
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsRadarCard { grid-column: 1; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsLiveCard { grid-row: 1; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsEmergencyCard { grid-row: 2; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #aiObjectSearchCard { grid-row: 3; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsAuditCard { grid-row: 4; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsClipsCard { grid-row: 5; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #ccmsRadarCard { grid-row: 6; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #incidentPlayerCard { grid-row: 7; }
      body[data-active-app="ccms"][data-ccms-section="overview"] #searchResultsContainer { grid-template-columns: repeat(2, minmax(0,1fr)); }
    }
  </style>
</head>
<body>
  <!-- Zoho-Format Top Navigation Header -->
  <header class="zoho-navbar">
    <div class="zoho-brand-group">
      <!-- Official VisionAI Platform Logo -->
      <img src="/logo.png?v=2" alt="VisionAI logo" class="zoho-logo-img">
      <div class="brand-divider"></div>
      <div class="zoho-brand-titles">
        <h1>VisionAI Enterprise</h1>
        <p>AI Surveillance &amp; Safety Platform</p>
      </div>
    </div>

    <div class="zoho-pill-group">
      <div class="zoho-nav-pill">
        <div class="status-dot" id="engineDot"></div>
        <span id="engineStatusText">CAMERA: STANDBY</span>
      </div>
      <div class="zoho-nav-pill">
        <span id="fpsCounter">FPS: 0.0</span>
      </div>
      <div class="zoho-nav-pill">
        <span>3-MIN BATCH: ACTIVE</span>
      </div>
      <div class="zoho-nav-pill" id="waHeaderPill" onclick="openWhatsAppModal()" style="cursor: pointer;" title="Configure WhatsApp Alerts">
        <div class="status-dot status-dot-standby" id="waDot"></div>
        <span id="waHeaderText">WHATSAPP: READY</span>
      </div>
    </div>
  </header>

  <nav class="ccms-subnav" aria-label="CCMS dashboard sections">
    <div class="ccms-sidebar-brand">
      <span class="ccms-brand-tile"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="2.5" y="5" width="13" height="14" rx="2"></rect><path d="m15.5 10 6-3v10l-6-3z"></path><circle cx="9" cy="12" r="2.5"></circle></svg></span>
      <span>CCMS<small>Control &amp; Monitoring</small></span>
    </div>

    <div class="ccms-sidebar-group">Monitoring</div>
    <button class="ccms-nav-item active" id="ccmsOverviewTab" type="button" aria-pressed="true" onclick="selectCcmsSection('overview')" style="--ccms-accent:#4da3ff;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1z"></path></svg></span>
      <span class="ccms-nav-text">Overview<small>Mission dashboard</small></span>
    </button>
    <button class="ccms-nav-item" id="ccmsLiveTab" type="button" onclick="focusCcmsPanel('ccmsLiveCard',this)" style="--ccms-accent:#39d6c3;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="6" width="13" height="12" rx="2"></rect><path d="m16 10 5-3v10l-5-3z"></path></svg></span>
      <span class="ccms-nav-text">Live Surveillance<small>Feed &amp; safety rules</small></span>
    </button>

    <div class="ccms-sidebar-group">Intelligence</div>
    <button class="ccms-nav-item" id="ccmsSearchTab" type="button" aria-pressed="false" onclick="selectCcmsSection('search')" style="--ccms-accent:#a78bfa;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg></span>
      <span class="ccms-nav-text">AI Object Search<small>Find any object</small></span>
      <span class="ccms-nav-chip">AI</span>
    </button>
    <button class="ccms-nav-item" id="ccmsAiChatTab" type="button" onclick="focusCcmsPanel('ccmsAiChatCard',this)" style="--ccms-accent:#45e0a1;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg></span>
      <span class="ccms-nav-text">Vision AI Assistant<small>Chat &amp; commands</small></span>
      <span class="ccms-nav-chip">AI</span>
    </button>
    <button class="ccms-nav-item" id="ccmsVideoAnalysisTab" type="button" onclick="focusCcmsPanel('ccmsVideoAnalysisCard',this)" style="--ccms-accent:#f7bd61;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M20.2 6 3 11l-.9-2.4c-.3-1.1.3-2.2 1.3-2.5l13.5-4c1.1-.3 2.2.3 2.5 1.3z"></path><path d="m6.2 5.3 3.1 3.1"></path><path d="m11.4 8.7 3.1 3.1"></path><path d="M3 11h18v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path></svg></span>
      <span class="ccms-nav-text">Video Analysis<small>Upload &amp; deep review</small></span>
    </button>

    <div class="ccms-sidebar-group">Records</div>
    <button class="ccms-nav-item" id="ccmsClipsTab" type="button" onclick="focusCcmsPanel('ccmsClipsCard',this)" style="--ccms-accent:#fb923c;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"></rect><path d="M7 4v16M17 4v16M3 9h18M3 15h18"></path></svg></span>
      <span class="ccms-nav-text">Incident Clips<small>Saved recordings</small></span>
    </button>
    <button class="ccms-nav-item" id="ccmsAlertsTab" type="button" onclick="focusCcmsPanel('ccmsAuditCard',this)" style="--ccms-accent:#fb5574;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M12 3 3 7v5c0 5 4 8 9 9 5-1 9-4 9-9V7z"></path><path d="m8 12 2.5 2.5L16 9"></path></svg></span>
      <span class="ccms-nav-text">Alerts &amp; Events<small>Violation &amp; audit feed</small></span>
    </button>

    <div class="ccms-sidebar-group">Control</div>
    <button class="ccms-nav-item" id="ccmsEmergencyTab" type="button" onclick="focusCcmsPanel('ccmsEmergencyCard',this)" style="--ccms-accent:#ff4d5e;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"></path><path d="M10 21h4"></path></svg></span>
      <span class="ccms-nav-text">Emergency Controls<small>Siren &amp; escalation</small></span>
    </button>
    <button class="ccms-nav-item" id="ccmsRadarTab" type="button" onclick="focusCcmsPanel('ccmsRadarCard',this)" style="--ccms-accent:#94a3b8;">
      <span class="ccms-nav-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><path d="M12 3v9l6 3M5.5 5.5l13 13"></path></svg></span>
      <span class="ccms-nav-text">Settings &amp; Monitors<small>Radar, confidence, system</small></span>
    </button>

    <div class="ccms-sidebar-spacer"></div>
    <div class="ccms-sidebar-footer"><b>VisionAI Sentinel</b><br>Local edge AI &bull; 80-class detection</div>
  </nav>

  <!-- Zoho Action Ribbon -->
  <div class="zoho-toolbar">
    <div class="toolbar-left">
      <span class="toolbar-label">FEED SOURCE:</span>
      <select id="cameraSelect" class="zoho-select">
        <option value="0">Camera 0 (Primary / Built-in)</option>
        <option value="1" selected>Camera 1 (External USB Camera)</option>
        <option value="2">Camera 2 (Secondary USB)</option>
      </select>

      <!-- START / STOP CAMERA BUTTON -->
      <button id="topToggleBtn" class="btn-zoho btn-zoho-primary" onclick="toggleCamera()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        <span id="topToggleText">START CAMERA</span>
      </button>

      <!-- EMERGENCY SIREN BUTTON (TORNADO WAIL) -->
      <button class="btn-zoho btn-zoho-emergency" id="topSirenBtn" onclick="soundEmergencySiren()" title="Sound Emergency Tornado Wail Siren">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
        <span>EMERGENCY SIREN</span>
      </button>
    </div>

    <div class="toolbar-right">
      <button class="btn-zoho btn-zoho-secondary" onclick="openWhatsAppModal()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>
        <span>WhatsApp Alerts</span>
      </button>
      <button class="btn-zoho btn-zoho-secondary" onclick="triggerManualDispatch()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"></path><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
        <span>Send Pending Batch Now</span>
      </button>
      <button class="btn-zoho btn-zoho-secondary" onclick="refreshData()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
        <span>Refresh</span>
      </button>
    </div>
  </div>

  <div class="ccms-page-heading" id="ccmsPageHeading">
    <div><h2>Centralized Control &amp; Monitoring System</h2><p>Main Facility &middot; Live surveillance and safety operations</p></div>
    <div class="ccms-operational"><span></span><b id="ccmsOperationalStatus">STANDBY</b></div>
  </div>
  <section id="ccmsKpiStrip" aria-label="CCMS system summary">
    <article class="ccms-kpi-card">
      <span class="ccms-kpi-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2.5" y="5" width="13" height="14" rx="2"></rect><path d="m15.5 10 6-3v10l-6-3z"></path></svg></span>
      <div><div class="ccms-kpi-label">Cameras</div><div class="ccms-kpi-value" id="ccmsKpiCameras">0 / 1</div><div class="ccms-kpi-note" id="ccmsKpiCameraNote">Standby</div></div>
    </article>
    <article class="ccms-kpi-card">
      <span class="ccms-kpi-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="7" y="7" width="10" height="10" rx="2"></rect><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"></path></svg></span>
      <div><div class="ccms-kpi-label">Tracked Objects</div><div class="ccms-kpi-value" id="ccmsKpiObjects">—</div><div class="ccms-kpi-note">Current object catalog</div></div>
    </article>
    <article class="ccms-kpi-card">
      <span class="ccms-kpi-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 3 7v5c0 5 4 8 9 9 5-1 9-4 9-9V7z"></path><path d="M12 8v5m0 3h.01"></path></svg></span>
      <div><div class="ccms-kpi-label">Incident Events</div><div class="ccms-kpi-value" id="ccmsKpiAlerts">0</div><div class="ccms-kpi-note">Recorded event log</div></div>
    </article>
    <article class="ccms-kpi-card">
      <span class="ccms-kpi-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"></rect><path d="m10 8 6 4-6 4z"></path></svg></span>
      <div><div class="ccms-kpi-label">Clips Stored</div><div class="ccms-kpi-value" id="ccmsKpiClips">0</div><div class="ccms-kpi-note">Recorded incidents</div></div>
    </article>
    <article class="ccms-kpi-card">
      <span class="ccms-kpi-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path></svg></span>
      <div><div class="ccms-kpi-label">System Uptime</div><div class="ccms-kpi-value" id="ccmsKpiUptime">—</div><div class="ccms-kpi-note">Dashboard process</div></div>
    </article>
  </section>

  <section class="mission-banner" aria-label="VisionAI command center overview">
    <div class="mission-copy">
      <div class="mission-eyebrow">VisionAI intelligence operations</div>
      <h2>See the signal.<br>Act in seconds.</h2>
      <p>Live surveillance, object intelligence and safety response — together in one CCMS command center.</p>
    </div>
    <div class="mission-console" aria-hidden="true">
      <div class="mission-orbit">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="8"></circle><circle cx="12" cy="12" r="3"></circle><path d="M12 2v3M22 12h-3M12 22v-3M2 12h3"></path>
        </svg>
      </div>
      <div>
        <div class="mission-console-label">VISIONAI CORE</div>
        <div class="mission-console-sub">CAMERA · OBJECTS · RESPONSE</div>
      </div>
    </div>
  </section>

  <section id="telegramEmergencyPanel" aria-live="assertive" style="display: none; margin: 16px 32px 0; padding: 16px 20px; border: 1px solid var(--zoho-crimson); border-left: 5px solid var(--zoho-crimson); border-radius: 8px; background: rgba(225, 29, 72, 0.14); box-shadow: 0 8px 24px rgba(0,0,0,0.22);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 16px;">
      <div style="min-width: 0;">
        <div id="emergencyPanelTitle" style="font-size: 13px; font-weight: 800; color: #FDA4AF; letter-spacing: 0.06em;">EMERGENCY ALERT</div>
        <div id="telegramEmergencyMessage" style="margin-top: 6px; color: var(--text-white); font-size: 15px; font-weight: 600; overflow-wrap: anywhere;"></div>
        <div id="telegramEmergencyTime" style="margin-top: 6px; color: var(--text-secondary); font-size: 11px; font-family: var(--font-mono);"></div>
      </div>
      <button class="btn-zoho btn-zoho-secondary" type="button" onclick="dismissTelegramEmergency()" style="flex: 0 0 auto;">Dismiss</button>
    </div>
  </section>

  <!-- Main Container -->
  <div class="zoho-container">
    <!-- Left Column: Video Feed, Player, Object Search, Intercom & Audit -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      
      <!-- Live Video Monitor Card -->
      <div class="zoho-card" id="ccmsLiveCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3"></circle></svg>
            <span>Live Edge Surveillance Stream</span>
          </div>
          <span id="streamStatusBadge" style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted);">CAMERA STANDBY</span>
        </div>
        <div class="zoho-card-body" style="padding: 12px;">
          <div class="stream-viewport">
            <img id="liveStreamImg" src="" alt="Live Surveillance Feed" style="display: none;">
            <div id="streamHeroLaunchpad" class="hero-launchpad">
              <button class="btn-hero-start" onclick="toggleCamera()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                <span>START LIVE CAMERA SURVEILLANCE</span>
              </button>
              <p style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">
                Select Camera 0 or 1 above and click to initialize AI surveillance engine.
              </p>
            </div>
          </div>
          <div class="detection-legend" aria-label="Detection box color key">
            <span class="legend-chip"><i style="background:#22c55e;"></i>Person</span>
            <span class="legend-chip"><i style="background:#f59e0b;"></i>Vehicle</span>
            <span class="legend-chip"><i style="background:#06b6d4;"></i>Electronics</span>
            <span class="legend-chip"><i style="background:#8b5cf6;"></i>Furniture</span>
            <span class="legend-chip"><i style="background:#84cc16;"></i>Kitchen</span>
            <span class="legend-chip"><i style="background:#ec4899;"></i>Animal</span>
            <span class="legend-chip"><i style="background:#ef4444;"></i>Hazard / Tool</span>
            <span class="legend-chip"><i style="background:#0ea5e9;"></i>Accessory</span>
          </div>
        </div>
      </div>

      <!-- 10-Second Incident Video Player Card -->
      <div class="zoho-card" id="incidentPlayerCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="2" width="20" height="20" rx="2"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line></svg>
            <span>10-Second Incident Video Player</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span id="videoPlaybackStatus" style="font-size: 10px; font-family: var(--font-mono); font-weight: 700; padding: 2px 8px; border-radius: 4px; background: rgba(15, 111, 222, 0.1); color: var(--zoho-blue); border: 1px solid rgba(15, 111, 222, 0.25);">STANDBY</span>
            <span id="activeVideoTitle" style="font-size: 11px; font-family: var(--font-mono); color: var(--text-secondary);">No clip selected</span>
            <a id="videoDirectLink" href="#" target="_blank" style="display: none; font-size: 11px; color: var(--zoho-blue); text-decoration: none; font-weight: 700; margin-left: 6px;">[OPEN IN NEW TAB]</a>
          </div>
        </div>
        <div class="zoho-card-body" style="padding: 12px;">
          <div class="player-box">
            <video id="incidentVideoPlayer" controls playsinline preload="auto" style="display: none; width: 100%; height: 100%; object-fit: contain;">
              Your browser does not support HTML5 video playback.
            </video>
            <div id="playerEmptyText" style="color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); padding: 24px; text-align: center;">
              Click any captured video clip from the playlist or object search below to review footage.
            </div>
          </div>
        </div>
      </div>

      <!-- Vision AI Interactive Assistant (Manual Input / Query Section) -->
      <div class="zoho-card" id="ccmsAiChatCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            <span>Vision AI Assistant &bull; Manual Inquiry &amp; Commands</span>
          </div>
          <span style="font-size: 11px; font-family: var(--font-mono); color: var(--zoho-emerald); font-weight: 700;">ACTIVE AI COPILOT</span>
        </div>
        <div class="zoho-card-body">
          <p class="ccms-search-intro">Talk directly with the Vision AI system. Ask about system status, tracked objects, incidents, camera controls, or ask questions regarding this surveillance workspace.</p>
          <div id="aiChatHistory" style="max-height: 220px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; padding: 12px; background: rgba(12,18,32,0.94); border: 1px solid var(--zoho-border); border-radius: 6px; margin-bottom: 12px; font-size: 13px; color: #e6edf5;">
            <div style="background: rgba(30,41,59,0.7); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--zoho-blue);">
              <strong style="color: var(--zoho-blue);">Vision AI:</strong> Hello! I am your Vision AI Assistant. I can run commands and answer live questions. Try <em>"status"</em>, <em>"anything wrong?"</em>, <em>"last hour"</em>, <em>"dispatch now"</em>, <em>"snapshot"</em>, <em>"mute 10 min"</em>, <em>"unmute siren"</em>, <em>"confidence 40"</em>, <em>"target bottle"</em>, or <em>"where is the phone?"</em>. Type <em>"help"</em> for the full list.
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <input type="text" id="aiChatInput" class="zoho-search-input" placeholder="Type your question or request to Vision AI (e.g. 'where is the cell phone?', 'status', 'who is sleeping?')..." onkeydown="if(event.key==='Enter') sendAiChatMessage()">
            <button class="btn-zoho btn-zoho-primary" onclick="sendAiChatMessage()" style="flex-shrink: 0;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
              <span>Ask AI</span>
            </button>
          </div>
        </div>
      </div>

      <!-- AI Object Search & Live Detection Video Retriever Card -->
      <div class="zoho-card" id="aiObjectSearchCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <span>AI Object Search & Live Detection Video Retriever</span>
          </div>
          <span id="objectAiStatus" style="font-size: 11px; font-family: var(--font-mono); color: var(--zoho-gold); font-weight: 700;">OBJECT AI WARMING UP</span>
        </div>
        <div class="zoho-card-body">
          <p class="ccms-search-intro">Search the tracked object catalog for recent detections, locations, model scores, snapshots, and available incident clips.</p>
          <!-- Active Target Notification Banner -->
          <div id="activeTargetBanner" class="active-target-banner" style="display: none;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span class="target-badge" id="activeTargetBadge">TARGET</span>
              <span>Actively marked with gold target brackets on live stream and in recorded video clips.</span>
            </div>
            <button type="button" class="btn-clear-target" onclick="clearSearchTarget()">Clear Target</button>
          </div>

          <div class="zoho-search-row">
            <input type="text" id="objectSearchInput" class="zoho-search-input" placeholder="Search any object or micro-item (e.g. pen, pencil, camera, keys, badge, book bag, bottle, phone)..." onkeydown="if(event.key==='Enter') searchObjects()">
            <button class="btn-zoho btn-zoho-primary" onclick="searchObjects()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
              <span>Search</span>
            </button>
          </div>

          <div class="zoho-quick-pills">
            <span style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Quick Search:</span>
            <button type="button" class="quick-pill" onclick="quickSearch('pen')">Pen / Pencil</button>
            <button type="button" class="quick-pill" onclick="quickSearch('camera')">Camera / Lens</button>
            <button type="button" class="quick-pill" onclick="quickSearch('keys')">Keys / Access Badge</button>
            <button type="button" class="quick-pill" onclick="quickSearch('book bag')">Book Bag / Backpack</button>
            <button type="button" class="quick-pill" onclick="quickSearch('phone')">Cell Phone</button>
            <button type="button" class="quick-pill" onclick="quickSearch('bottle')">Water Bottle</button>
            <button type="button" class="quick-pill" onclick="quickSearch('laptop')">Laptop / PC</button>
            <button type="button" class="quick-pill" onclick="quickSearch('')">Show All Tracked</button>
          </div>

          <!-- Dynamic Search Results Grid -->
          <div id="searchResultsContainer" class="object-results-grid"></div>
        </div>
      </div>

      <div class="zoho-card" id="ccmsVideoAnalysisCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"></rect><path d="m10 8 6 4-6 4z"></path></svg>
            <span>Uploaded Video Analysis</span>
          </div>
          <span style="font-size: 11px; font-family: var(--font-mono); color: var(--zoho-blue); font-weight: 700;">LOCAL AI REVIEW</span>
        </div>
        <div class="zoho-card-body">
          <p class="ccms-search-intro">Screen an uploaded recording for every visible number plate, motorcycle riders without helmets, vehicles crossing a red light, and automatic per-vehicle speeds. Every finding is marked with labeled rectangles and a confidence score, and each violation is cut into an annotated 10-second clip and sent to the Telegram bot.</p>
          <div style="padding: 11px 13px; border: 1px solid rgba(15,111,222,.25); border-radius: 6px; background: rgba(15,111,222,.06); color: var(--text-secondary); font-size: 12px; line-height: 1.6;">
            No manual setup is needed. While the traffic signal stays red, VisionAI derives the <strong>stop line automatically from the signal's position</strong>; any vehicle that crosses it triggers the <strong>emergency siren, a spoken warning, and an immediate 10-second annotated clip to the Telegram bot</strong>. Vehicle speed is estimated automatically from each vehicle's on-screen size, and every number plate is read with its video timestamp. YOLO11 tracks vehicles and traffic lights; a plate detector plus enhanced OCR reads plates; an open-vocabulary model checks riders' heads for helmets. OCR, signal colors, and speeds are AI estimates, so verify every result. This generic dashboard does not issue official citations.
          </div>
          <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 14px;">
            <input id="uploadedVideoFile" type="file" accept=".mp4,.mov,.mkv,.avi,.webm,.m4v,video/*" onchange="handleUploadedVideoSelection(this)" style="max-width: 100%; color: var(--text-secondary); font-size: 12px;">
            <button id="analyzeUploadedVideoBtn" type="button" class="btn-zoho btn-zoho-primary" onclick="analyzeUploadedVideo()" disabled>ANALYZE VIDEO</button>
            <button type="button" class="btn-zoho btn-zoho-secondary" onclick="clearUploadedVideo()">Clear</button>
          </div>
          <div id="uploadedVideoFileInfo" role="status" style="margin-top: 8px; color: var(--text-muted); font-size: 11px;">Choose an MP4, MOV, MKV, AVI, WEBM, or M4V file (maximum 200 MB).</div>
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:12px;">
            <label style="font-size:11px;color:var(--text-secondary);">Speed limit (km/h)
              <input id="uploadedSpeedLimit" type="number" min="1" max="300" step="1" value="60" style="width:76px;margin-left:5px;padding:7px;border:1px solid var(--zoho-border);border-radius:5px;">
            </label>
          </div>
          <div id="uploadedVideoPreviewWrap" style="display:none;position:relative;width:fit-content;max-width:100%;margin-top:12px;">
            <video id="uploadedVideoPreview" controls playsinline preload="metadata" style="display:block;width:auto;max-width:100%;max-height:340px;border-radius:6px;background:#080d18;"></video>
          </div>
          <div id="uploadedVideoAnalysisProgress" style="display: none; margin-top: 14px;">
            <div style="display: flex; justify-content: space-between; gap: 12px; color: var(--text-secondary); font-size: 11px;">
              <span id="uploadedVideoAnalysisStatus" role="status">Preparing upload...</span>
              <span id="uploadedVideoAnalysisPercent">0%</span>
            </div>
            <progress id="uploadedVideoAnalysisMeter" value="0" max="100" style="width: 100%; height: 8px; margin-top: 7px;"></progress>
          </div>
          <div id="uploadedVideoAnalysisResults" aria-live="polite" style="margin-top: 12px;"></div>
        </div>
      </div>

      <!-- Emergency Siren & Voice Intercom Card -->
      <div class="zoho-card" id="ccmsEmergencyCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            <span>Emergency Siren & Voice Intercom</span>
          </div>
          <span id="speakerStatusBadge" style="font-size: 11px; font-family: var(--font-mono); color: var(--zoho-emerald); font-weight: 700;">STATION READY</span>
        </div>
        <div class="zoho-card-body">
          <div style="display: flex; gap: 12px; align-items: center; justify-content: space-between; flex-wrap: wrap;">
            <div style="display: flex; gap: 10px;">
              <button class="btn-zoho btn-zoho-emergency" onclick="soundEmergencySiren()">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
                <span>EMERGENCY SIREN</span>
              </button>
              <button class="btn-zoho btn-zoho-secondary" onclick="silenceEmergencySiren()">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="1" y1="1" x2="23" y2="23"></line><path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path></svg>
                <span>Silence Siren</span>
              </button>
            </div>
            <div style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono);">
              Audio sounds through connected facility speakers and local browser.
            </div>
          </div>

          <!-- Spoken Warning Presets -->
          <div style="margin-top: 16px;">
            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">
              Spoken Warning Presets:
            </div>
            <div class="preset-group">
              <button class="preset-pill" onclick="speakVoiceMessage('Warning: High-visibility safety vest is required in this facility immediately.')">PPE Vest Warning</button>
              <button class="preset-pill" onclick="speakVoiceMessage('Danger: Machine danger perimeter breached. Step back immediately.')">Perimeter Breach Warning</button>
              <button class="preset-pill" onclick="speakVoiceMessage('Attention staff: Customer waiting. Please attend to the billing desk immediately.')">Cashier Desk Call</button>
              <button class="preset-pill" onclick="speakVoiceMessage('Notice: Personal mobile phone usage is prohibited while on active duty.')">Phone Distraction Warning</button>
              <button class="preset-pill" onclick="speakVoiceMessage('Emergency alert: Immediate security inspection requested on floor.')">General Floor Alert</button>
            </div>
          </div>

          <!-- Custom Voice Intercom -->
          <div class="intercom-row">
            <input type="text" id="customVoiceInput" class="zoho-search-input" placeholder="Type custom voice warning announcement to speak through facility speakers...">
            <button class="btn-zoho btn-zoho-secondary" onclick="sendCustomVoice()">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon></svg>
              <span>Broadcast Voice</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Security & Compliance Audit Trail Card -->
      <div class="zoho-card" id="ccmsAuditCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
            <span>Security & Compliance Audit Trail</span>
          </div>
        </div>
        <div class="zoho-card-body" style="padding: 0; max-height: 260px; overflow-y: auto;">
          <table class="zoho-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Violation Event</th>
                <th>Details</th>
                <th>Recorded Footage</th>
              </tr>
            </thead>
            <tbody id="auditTableBody">
              <tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No records found.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

    </div>

    <!-- Right Column: 3-Min Batch Timer, Safety Radar & Captured Clips -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      
      <!-- 3-Minute Batch Timer Card -->
      <div class="zoho-card" id="ccmsBatchCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            <span>3-Minute Batch Cycle</span>
          </div>
        </div>
        <div class="zoho-card-body">
          <div class="timer-flex">
            <div class="ring-box">
              <svg viewBox="0 0 96 96">
                <circle class="ring-bg" cx="48" cy="48" r="40"></circle>
                <circle id="cycleProgressCircle" class="ring-prog" cx="48" cy="48" r="40"></circle>
              </svg>
              <div class="ring-text" id="ringCountdown">03:00</div>
            </div>
            <div>
              <div style="font-size: 13px; font-weight: 700; color: var(--text-white);">CYCLE COUNTDOWN</div>
              <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px; line-height: 1.4;">
                All pending violation clips are sent with captions every 3 minutes, including while the camera is off. No immediate Telegram warning texts.
                <div id="telegramBatchStatus" style="margin-top:6px;">TELEGRAM: 0 CLIPS QUEUED</div>
                <div style="margin-top:6px;">Camera confirmation: 12s continuous evidence (15s perimeter dwell). Ambiguous colour/posture checks stay local.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Safety Compliance Radar Card -->
      <div class="zoho-card" id="ccmsRadarCard">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            <span>Safety Compliance Radar</span>
          </div>
          <span id="radarMonitorStatus" style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); font-weight: 700;">13 MONITORS STANDBY</span>
        </div>
        <div class="zoho-card-body">
          <div class="radar-list" id="radarList">
            <div class="radar-row"><span>PPE Safety Vest</span><span class="badge badge-muted">STANDBY</span></div>
            <div class="radar-row"><span>Danger Perimeter</span><span class="badge badge-ok">SECURE</span></div>
            <div class="radar-row"><span>Shift Vigor</span><span class="badge badge-ok">ACTIVE</span></div>
            <div class="radar-row"><span>Duty Desk</span><span class="badge badge-muted">STANDBY</span></div>
            <div class="radar-row"><span>Phone Distraction</span><span class="badge badge-ok">CLEAR</span></div>
            <div class="radar-row"><span>Queue Monitor</span><span class="badge badge-ok">CLEAR</span></div>
            <div class="radar-row"><span>Fall Watch</span><span class="badge badge-ok">NORMAL</span></div>
            <div class="radar-row"><span>Conduct Watch</span><span class="badge badge-ok">NORMAL</span></div>
            <div class="radar-row"><span>Inactivity Watch</span><span class="badge badge-muted">STANDBY</span></div>
            <div class="radar-row"><span>Loitering Watch</span><span class="badge badge-ok">CLEAR</span></div>
            <div class="radar-row"><span>Truck Turnaround</span><span class="badge badge-ok">NO VEHICLE</span></div>
            <div class="radar-row"><span>VIP Entrance</span><span class="badge badge-ok">NO ENTRY</span></div>
            <div class="radar-row"><span>Cleanliness Check</span><span class="badge badge-ok">CLEAN</span></div>
          </div>
        </div>
      </div>

      <!-- Captured Video Clips Playlist Card -->
      <div class="zoho-card" id="ccmsClipsCard" style="flex: 1;">
        <div class="zoho-card-header">
          <div class="zoho-card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2"></rect></svg>
            <span>Captured Video Clips</span>
          </div>
          <span id="clipCountLabel" style="font-size: 11px; font-family: var(--font-mono); color: var(--text-secondary);">0 Files</span>
        </div>
        <div class="zoho-card-body">
          <div class="playlist-scroll" id="playlist">
            <p style="text-align: center; color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); padding: 24px;">
              No incident video clips recorded yet.
            </p>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- WhatsApp Configuration Modal Dialog -->
  <div id="waModal" class="modal-backdrop">
    <div class="modal-dialog">
      <div class="modal-head">
        <div style="display: flex; align-items: center; gap: 10px;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--zoho-emerald)" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>
          <span style="font-weight: 700; font-size: 14px; color: var(--text-white);">Configure WhatsApp Direct Alerts</span>
        </div>
        <button onclick="closeWhatsAppModal()" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 20px; line-height: 1;">&times;</button>
      </div>
      <div class="modal-content">
        <div style="font-size: 12px; color: var(--text-secondary); line-height: 1.5;">
          Dispatch validated 10s video alerts and security violations directly to WhatsApp.
        </div>
        <div>
          <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Select Alert Provider:</label>
          <select id="waProviderSelect" class="zoho-select" style="width: 100%;" onchange="onWaProviderChange()">
            <option value="callmebot">CallMeBot (Free Personal Instant Link - No Account Required)</option>
            <option value="twilio">Twilio WhatsApp API (Send to ANY Mobile Number Worldwide)</option>
          </select>
        </div>
        <div>
          <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Recipient WhatsApp Phone Number (with Country Code):</label>
          <input type="text" id="waPhoneInput" class="zoho-search-input" style="width: 100%;" placeholder="+919876543210 or +14155552671">
        </div>

        <!-- CallMeBot Provider Section -->
        <div id="callmebotSection">
          <div>
            <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">CallMeBot API Key:</label>
            <input type="text" id="waApiKeyInput" class="zoho-search-input" style="width: 100%;" placeholder="Enter numeric API key (e.g. 123456)">
          </div>
          <div style="background: rgba(15, 111, 222, 0.08); border: 1px solid var(--zoho-border); border-radius: 6px; padding: 12px; margin-top: 8px; font-size: 11px; color: var(--text-secondary); line-height: 1.5;">
            <div style="color: var(--zoho-blue); font-weight: 700; margin-bottom: 4px;">WHAT TO ENTER IN API KEY (10-SECOND FREE SETUP):</div>
            1. Save phone number <strong style="color: var(--text-white);">+34 644 44 49 64</strong> in your WhatsApp.<br>
            2. Send this text to that number: <strong style="color: var(--zoho-gold);">I allow callmebot to send me messages</strong><br>
            3. The bot replies instantly: <code style="color: var(--zoho-emerald); font-weight: 700;">API Key: XXXXXX</code>. Paste that numeric code into the API Key box above!
          </div>
        </div>

        <!-- Twilio Provider Section -->
        <div id="twilioSection" style="display: none; flex-direction: column; gap: 12px;">
          <div>
            <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Twilio Account SID:</label>
            <input type="text" id="waTwilioSidInput" class="zoho-search-input" style="width: 100%;" placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
          </div>
          <div>
            <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Twilio Auth Token / API Secret:</label>
            <input type="password" id="waTwilioTokenInput" class="zoho-search-input" style="width: 100%;" placeholder="Enter your Twilio Auth Token">
          </div>
          <div>
            <label style="display: block; font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px;">Twilio WhatsApp Sender Number:</label>
            <input type="text" id="waTwilioFromInput" class="zoho-search-input" style="width: 100%;" placeholder="+14155238886">
          </div>
          <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid var(--zoho-border); border-radius: 6px; padding: 12px; font-size: 11px; color: var(--text-secondary); line-height: 1.5;">
            <div style="color: var(--zoho-emerald); font-weight: 700; margin-bottom: 4px;">SEND TO ANY ARBITRARY NUMBER:</div>
            Twilio WhatsApp Business API allows you to send instant alerts to ANY client, security guard, or manager phone number worldwide without them needing to save a number first.
          </div>
        </div>

        <div id="waStatusMsg" style="font-size: 11px; font-family: var(--font-mono); padding: 10px 14px; border-radius: 6px; display: none;"></div>
      </div>
      <div class="modal-foot">
        <button class="btn-zoho btn-zoho-secondary" onclick="testWhatsAppLive()">
          <span>SEND LIVE TEST ALERT</span>
        </button>
        <button class="btn-zoho btn-zoho-primary" onclick="saveWhatsAppModal()">
          <span>SAVE CONFIGURATION</span>
        </button>
      </div>
    </div>
  </div>

  <div id="clipModal" class="modal-backdrop clip-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="clipModalTitle" onclick="if(event.target === this) closeClipModal()">
    <div class="modal-dialog clip-modal-dialog">
      <div class="modal-head">
        <div style="min-width: 0;">
          <div id="clipModalTitle" style="font-weight: 700; font-size: 14px; color: var(--text-white);">Object Clip</div>
          <div id="clipModalStatus" role="status" style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono); margin-top: 4px;">Loading clip...</div>
        </div>
        <button type="button" aria-label="Close video player" onclick="closeClipModal()" style="background: none; border: none; color: var(--text-secondary); cursor: pointer; font-size: 24px; line-height: 1;">&times;</button>
      </div>
      <video id="objectClipPlayer" class="clip-modal-video" controls playsinline preload="metadata">
        Your browser does not support HTML5 video playback.
      </video>
      <div id="clipModalEmptyState" class="clip-modal-empty" role="status"></div>
      <div class="modal-foot" style="justify-content: space-between;">
        <a id="clipModalDirectLink" href="#" target="_blank" rel="noopener" style="font-size: 12px; color: var(--zoho-blue); text-decoration: none; font-weight: 700;">OPEN VIDEO DIRECTLY</a>
        <button type="button" class="btn-zoho btn-zoho-secondary" onclick="closeClipModal()">CLOSE</button>
      </div>
    </div>
  </div>

  <script>
    let isCameraActive = false;
    let currentSearchTarget = '';

    // =========================================================================
    // Camera Start / Stop Lifecycle
    // =========================================================================
    function setCameraButtonLabel(button, label) {
      let text = button.querySelector('#topToggleText');
      if (!text) {
        text = document.createElement('span');
        text.id = 'topToggleText';
        button.appendChild(text);
      }
      text.innerText = label;
    }

    async function toggleCamera() {
      const topBtn = document.getElementById('topToggleBtn');
      const camSelect = document.getElementById('cameraSelect');

      if (!isCameraActive) {
        topBtn.disabled = true;
        setCameraButtonLabel(topBtn, 'INITIALIZING...');
        try {
          const resp = await fetch('/api/camera/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ camera: parseInt(camSelect.value) })
          });
          const data = await resp.json();
          if (data.ok) {
            isCameraActive = true;
            if (data.camera !== undefined) camSelect.value = String(data.camera);
            topBtn.className = 'btn-zoho btn-zoho-stop';
            topBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg><span id="topToggleText">STOP CAMERA</span>';
            camSelect.disabled = true;

            const streamImg = document.getElementById('liveStreamImg');
            const heroLaunchpad = document.getElementById('streamHeroLaunchpad');
            streamImg.src = '/video_feed?t=' + Date.now();
            streamImg.style.display = 'block';
            heroLaunchpad.style.display = 'none';

            document.getElementById('streamStatusBadge').innerText = 'LIVE SURVEILLANCE ACTIVE';
            document.getElementById('streamStatusBadge').style.color = 'var(--zoho-emerald)';
            document.getElementById('engineDot').className = 'status-dot status-dot-active';
            document.getElementById('engineStatusText').innerText = 'CAMERA: ONLINE';
          } else {
            alert(data.message || 'Camera could not be detected. Check the USB connection and close other camera apps.');
            setCameraButtonLabel(topBtn, 'START CAMERA');
          }
        } catch (e) {
          isCameraActive = false;
          topBtn.className = 'btn-zoho btn-zoho-primary';
          setCameraButtonLabel(topBtn, 'START CAMERA');
          alert('Failed to connect to camera: ' + e);
        } finally {
          topBtn.disabled = false;
        }
      } else {
        topBtn.disabled = true;
        setCameraButtonLabel(topBtn, 'STOPPING...');

        const streamImg = document.getElementById('liveStreamImg');
        streamImg.src = '';
        streamImg.style.display = 'none';

        try {
          const resp = await fetch('/api/camera/stop', { method: 'POST' });
          await resp.json();
        } catch (e) {
          console.error('Stop request error:', e);
        }

        isCameraActive = false;
        topBtn.className = 'btn-zoho btn-zoho-primary';
        topBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span id="topToggleText">START CAMERA</span>';
        camSelect.disabled = false;

        const heroLaunchpad = document.getElementById('streamHeroLaunchpad');
        heroLaunchpad.style.display = 'flex';

        document.getElementById('streamStatusBadge').innerText = 'CAMERA STANDBY';
        document.getElementById('streamStatusBadge').style.color = 'var(--text-muted)';
        document.getElementById('engineDot').className = 'status-dot status-dot-standby';
        document.getElementById('engineStatusText').innerText = 'CAMERA: STANDBY';
        document.getElementById('fpsCounter').innerText = 'FPS: 0.0';
        topBtn.disabled = false;
      }
    }

    async function triggerManualDispatch() {
      try {
        const resp = await fetch('/api/dispatch_now', { method: 'POST' });
        const data = await resp.json();
        alert(data.message || '3-minute cycle dispatch executed.');
        refreshData();
      } catch (e) {
        alert('Dispatch error: ' + e);
      }
    }

    // =========================================================================
    // AI Object Search & Real-Time Target Marking
    // =========================================================================
    async function searchObjects(term) {
      const query = (term !== undefined ? term : document.getElementById('objectSearchInput').value).trim();
      const container = document.getElementById('searchResultsContainer');
      container.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-secondary); font-size: 12px; padding: 20px;">Scanning object catalog...</div>';

      try {
        const resp = await fetch('/api/objects/search?q=' + encodeURIComponent(query));
        const data = await resp.json();
        currentSearchTarget = query;
          document.getElementById('ccmsKpiObjects').textContent =
            String(Array.isArray(data.results) ? data.results.length : 0);
          updateTargetBanner(query);
        renderSearchResults(data.results || [], query);
        if ((data.results || []).some(item => item.clip_recording || (query && isCameraActive && !item.video_clip)) && currentSearchTarget === query) {
          setTimeout(() => {
            if (currentSearchTarget === query) searchObjects(query);
          }, 2500);
        }
      } catch (e) {
        container.innerHTML = '<div style="grid-column: 1/-1; color: var(--zoho-crimson); font-size: 12px; padding: 20px;">Search request failed: ' + e + '</div>';
      }
    }

    async function captureObjectClip(target) {
      document.getElementById('objectSearchInput').value = target;
      const response = await fetch('/api/objects/set_target', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({target})
      });
      if (!response.ok) return;
      if (!isCameraActive) await toggleCamera();
      await searchObjects(target);
      const banner = document.getElementById('activeTargetBadge');
      banner.innerText = 'TARGET: ' + target.toUpperCase() + ' — Wait for MULTI-OBJECT AI READY, then hold the object clearly in view for 10 seconds.';
    }

    function quickSearch(term) {
      document.getElementById('objectSearchInput').value = term;
      searchObjects(term);
    }

    function updateTargetBanner(target) {
      const banner = document.getElementById('activeTargetBanner');
      const badge = document.getElementById('activeTargetBadge');
      if (target) {
        badge.innerText = 'TARGET: ' + target.toUpperCase();
        banner.style.display = 'flex';
      } else {
        banner.style.display = 'none';
      }
    }

    async function clearSearchTarget() {
      try {
        await fetch('/api/objects/set_target', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ target: '' })
        });
      } catch (e) {}
      currentSearchTarget = '';
      document.getElementById('objectSearchInput').value = '';
      updateTargetBanner('');
      searchObjects('');
    }

    function renderSearchResults(items, query) {
      const container = document.getElementById('searchResultsContainer');
      if (!items || items.length === 0) {
        container.innerHTML = `
          <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); font-size: 12px; padding: 24px; background: rgba(255, 255, 255, 0.02); border-radius: 6px; border: 1px dashed var(--zoho-border);">
            No objects found matching "${escapeHtml(query)}". Try searching for <strong>pen</strong>, <strong>pencil</strong>, <strong>camera</strong>, <strong>keys</strong>, <strong>book bag</strong>, or <strong>phone</strong>.
          </div>
        `;
        return;
      }

      container.innerHTML = items.map(item => {
        const safeName = escapeHtml(item.display_name || item.label || 'Object');
        const snapSrc = item.snapshot_file ? ('/snapshot/' + encodeURIComponent(item.snapshot_file)) : '';
        const thumbHtml = snapSrc
          ? `<img src="${snapSrc}" alt="${safeName}" onerror="this.parentElement.innerHTML='<span style=\\'font-size:10px;color:var(--text-muted);\\'>SNAPSHOT</span>'">`
          : `<span style="font-size: 10px; color: var(--text-muted); font-family: var(--font-mono);">SNAPSHOT</span>`;
        const clipName = item.video_clip || '';
        const safeClip = escapeHtml(clipName);
        const clipButton = clipName
          ? `<button type="button" class="btn-play-clip" data-clip="${safeClip}" data-label="${safeName}" onclick="playObjectClip(this.dataset.clip, this.dataset.label)">
               <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
               <span>PLAY LAST LOCATION CLIP</span>
             </button>`
          : `<button type="button" class="btn-play-clip" data-target="${escapeHtml(item.label || '')}" onclick="captureObjectClip(this.dataset.target)">CAPTURE LOCATION CLIP</button>`;
        const clipNote = !clipName && item.clip_recording
          ? `<span style="display: block; font-size: 10px; color: var(--text-muted); font-family: var(--font-mono);">Recording this object's clip…</span>`
          : '';

        return `
          <div class="object-card">
            <div class="object-thumb">${thumbHtml}</div>
            <div class="object-info">
              <div>
                <div class="object-title">${safeName}</div>
                <div class="object-meta">Last Seen: <strong style="color: var(--text-white);">${escapeHtml(item.last_seen_time || 'Recent')}</strong></div>
                <div class="object-meta">Location: <span style="color: var(--zoho-gold);">${escapeHtml(item.location_desc || 'Within FOV')}</span></div>
                <div class="object-meta">${item.confidence === 'Reviewed' ? 'Verification' : 'Model score (not accuracy)'}: <span style="color: var(--zoho-emerald);">${escapeHtml(item.confidence || 'Unverified')}</span></div>
              </div>
              <div>
                ${clipButton}
                ${clipNote}
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function playObjectClip(clipFilename, label) {
      const modal = document.getElementById('clipModal');
      const player = document.getElementById('objectClipPlayer');
      const status = document.getElementById('clipModalStatus');
      const directLink = document.getElementById('clipModalDirectLink');
      const emptyState = document.getElementById('clipModalEmptyState');
      const showEmptyState = function(message) {
        player.pause();
        player.removeAttribute('src');
        player.style.display = 'none';
        player.load();
        emptyState.innerText = message;
        emptyState.style.display = 'flex';
        directLink.style.display = 'none';
      };
      const selectedClip = clipFilename && clipFilename !== '-' ? clipFilename : '';
      if (!selectedClip) {
        document.getElementById('clipModalTitle').innerText = (label || 'Object') + ' - No Matching Clip';
        status.innerText = 'No matching recording is available.';
        status.style.color = 'var(--zoho-gold)';
        showEmptyState('No clip of this object has been saved. Start the camera and keep the object visible to record a matching clip.');
        modal.style.display = 'flex';
        return;
      }

      const cleanFile = String(selectedClip).split('?')[0].split('/').pop().split('\\\\').pop();
      if (!cleanFile) return;

      const videoUrl = '/video/' + encodeURIComponent(cleanFile);

      document.getElementById('clipModalTitle').innerText = (label || 'Object') + ' - Last Detected Clip';
      status.innerText = 'Loading ' + cleanFile;
      status.style.color = 'var(--text-secondary)';
      emptyState.style.display = 'none';
      player.style.display = 'block';
      directLink.href = videoUrl;
      directLink.style.display = 'inline-block';
      modal.style.display = 'flex';

      player.pause();
      player.removeAttribute('src');
      player.load();
      player.onerror = function() {
        const errorCode = player.error ? player.error.code : 0;
        const message = errorCode === 4
          ? 'Video format is not supported. Use OPEN VIDEO DIRECTLY to check the file.'
          : 'Could not load this object clip. Check that the clip file still exists.';
        status.innerText = message;
        status.style.color = 'var(--zoho-crimson)';
        emptyState.innerText = message;
        emptyState.style.display = 'flex';
        player.style.display = 'none';
      };
      player.onplaying = function() {
        status.innerText = cleanFile;
        status.style.color = 'var(--zoho-emerald)';
      };
      player.oncanplay = function() {
        status.innerText = 'Clip ready - use the video controls to play.';
      };
      player.src = videoUrl;
      player.load();

      const playback = player.play();
      if (playback && typeof playback.catch === 'function') {
        playback.catch(function(error) {
          if (error.name !== 'NotAllowedError' && error.name !== 'AbortError') {
            status.innerText = 'Playback failed. Use the video controls or OPEN VIDEO DIRECTLY.';
            status.style.color = 'var(--zoho-crimson)';
          } else if (error.name === 'NotAllowedError') {
            status.innerText = 'Clip loaded - press Play to start.';
          }
        });
      }
    }

    function closeClipModal() {
      const modal = document.getElementById('clipModal');
      const player = document.getElementById('objectClipPlayer');
      player.pause();
      player.removeAttribute('src');
      player.load();
      player.style.display = 'block';
      document.getElementById('clipModalEmptyState').style.display = 'none';
      modal.style.display = 'none';
    }

    document.addEventListener('keydown', function(event) {
      if (event.key === 'Escape' && document.getElementById('clipModal').style.display === 'flex') {
        closeClipModal();
      }
    });

    // =========================================================================
    // Authentic Mechanical Tornado Air-Raid Wail Sound Synthesizer
    // =========================================================================
    let audioCtx = null;
    let sirenOsc1 = null;
    let sirenOsc2 = null;
    let sirenGain = null;
    let sirenTimer = null;

    function playLocalBrowserSiren(durationSec = 4.0) {
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) return;
        if (!audioCtx) audioCtx = new AudioContext();
        if (audioCtx.state === 'suspended') audioCtx.resume();

        stopLocalBrowserSiren();

        const now = audioCtx.currentTime;
        sirenOsc1 = audioCtx.createOscillator();
        sirenOsc2 = audioCtx.createOscillator();
        sirenGain = audioCtx.createGain();

        // Realistic mechanical tornado air-raid harmonics
        sirenOsc1.type = 'sawtooth';
        sirenOsc2.type = 'sine';
        sirenOsc2.detune.setValueAtTime(14, now); // acoustic horn phasing

        sirenGain.gain.setValueAtTime(0.001, now);
        sirenGain.gain.linearRampToValueAtTime(0.35, now + 0.3); // smooth wind-up

        // Undulating tornado air-raid wail: sweeps between 420 Hz and 840 Hz (2.4s cycle)
        const cycle = 2.4;
        const totalCycles = Math.ceil(durationSec / cycle);
        let t = now;
        sirenOsc1.frequency.setValueAtTime(420, t);
        sirenOsc2.frequency.setValueAtTime(420, t);

        for (let i = 0; i < totalCycles; i++) {
          sirenOsc1.frequency.linearRampToValueAtTime(840, t + cycle * 0.5);
          sirenOsc2.frequency.linearRampToValueAtTime(840, t + cycle * 0.5);
          sirenOsc1.frequency.linearRampToValueAtTime(420, t + cycle);
          sirenOsc2.frequency.linearRampToValueAtTime(420, t + cycle);
          t += cycle;
        }

        sirenOsc1.connect(sirenGain);
        sirenOsc2.connect(sirenGain);
        sirenGain.connect(audioCtx.destination);

        sirenOsc1.start(now);
        sirenOsc2.start(now);

        sirenTimer = setTimeout(() => {
          stopLocalBrowserSiren();
        }, durationSec * 1000);
      } catch (e) {
        console.log('Siren audio notice:', e);
      }
    }

    function stopLocalBrowserSiren() {
      if (sirenTimer) {
        clearTimeout(sirenTimer);
        sirenTimer = null;
      }
      if (sirenGain && audioCtx) {
        try {
          sirenGain.gain.linearRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);
        } catch(e){}
      }
      setTimeout(() => {
        if (sirenOsc1) { try { sirenOsc1.stop(); }catch(e){} sirenOsc1.disconnect(); sirenOsc1 = null; }
        if (sirenOsc2) { try { sirenOsc2.stop(); }catch(e){} sirenOsc2.disconnect(); sirenOsc2 = null; }
        sirenGain = null;
      }, 250);
    }

    // =========================================================================
    // Emergency Siren & Voice Intercom Handlers
    // =========================================================================
    async function soundEmergencySiren() {
      const badge = document.getElementById('speakerStatusBadge');
      badge.innerText = 'SIREN BLASTING...';
      badge.style.color = 'var(--zoho-crimson)';
      playLocalBrowserSiren(4.0);

      try {
        const resp = await fetch('/api/speaker/siren', { method: 'POST' });
        const data = await resp.json();
        console.log(data.message);
      } catch (e) {
        console.error('Siren trigger error:', e);
      }

      setTimeout(() => {
        badge.innerText = 'STATION READY';
        badge.style.color = 'var(--zoho-emerald)';
      }, 4000);
    }

    async function silenceEmergencySiren() {
      stopLocalBrowserSiren();
      const badge = document.getElementById('speakerStatusBadge');
      badge.innerText = 'SIREN SILENCED';
      badge.style.color = 'var(--text-muted)';
      try {
        await fetch('/api/speaker/stop', { method: 'POST' });
      } catch (e) {}
      setTimeout(() => {
        badge.innerText = 'STATION READY';
        badge.style.color = 'var(--zoho-emerald)';
      }, 1500);
    }

    async function speakVoiceMessage(text) {
      if (!text) return;
      const badge = document.getElementById('speakerStatusBadge');
      badge.innerText = 'BROADCASTING VOICE...';
      badge.style.color = 'var(--zoho-blue)';

      try {
        const resp = await fetch('/api/speaker/voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        const data = await resp.json();
        console.log(data.message);
      } catch (e) {
        console.error('Voice warning error:', e);
      }

      setTimeout(() => {
        badge.innerText = 'STATION READY';
        badge.style.color = 'var(--zoho-emerald)';
      }, 3000);
    }

    function sendCustomVoice() {
      const input = document.getElementById('customVoiceInput');
      const val = (input.value || '').trim();
      if (!val) {
        alert('Please enter a voice warning to speak.');
        return;
      }
      speakVoiceMessage(val);
      input.value = '';
    }

    // =========================================================================
    // WhatsApp Configuration Modal & Direct Alerts
    // =========================================================================
    function onWaProviderChange() {
      const prov = document.getElementById('waProviderSelect').value;
      const cbSec = document.getElementById('callmebotSection');
      const twSec = document.getElementById('twilioSection');
      if (prov === 'twilio') {
        cbSec.style.display = 'none';
        twSec.style.display = 'flex';
      } else {
        cbSec.style.display = 'block';
        twSec.style.display = 'none';
      }
    }

    async function openWhatsAppModal() {
      const modal = document.getElementById('waModal');
      modal.style.display = 'flex';
      try {
        const resp = await fetch('/api/whatsapp/status');
        const data = await resp.json();
        if (data.phone) document.getElementById('waPhoneInput').value = data.phone;
        if (data.provider) document.getElementById('waProviderSelect').value = data.provider;
        if (data.account_sid) document.getElementById('waTwilioSidInput').value = data.account_sid;
        if (data.from_whatsapp) document.getElementById('waTwilioFromInput').value = data.from_whatsapp;
        document.getElementById('waApiKeyInput').placeholder = data.has_api_key ? 'Saved - leave blank to keep' : 'Enter numeric API key (e.g. 123456)';
        document.getElementById('waTwilioTokenInput').placeholder = data.has_auth_token ? 'Saved - leave blank to keep' : 'Enter your Twilio Auth Token';
        onWaProviderChange();
      } catch (e) {}
    }

    function closeWhatsAppModal() {
      document.getElementById('waModal').style.display = 'none';
      const statusBox = document.getElementById('waStatusMsg');
      statusBox.style.display = 'none';
    }

    async function saveWhatsAppModal() {
      const provider = document.getElementById('waProviderSelect').value;
      const phone = document.getElementById('waPhoneInput').value.trim();
      const apiKey = document.getElementById('waApiKeyInput').value.trim();
      const sid = document.getElementById('waTwilioSidInput').value.trim();
      const token = document.getElementById('waTwilioTokenInput').value.trim();
      const fromWa = document.getElementById('waTwilioFromInput').value.trim();
      const statusBox = document.getElementById('waStatusMsg');

      const payload = {
        provider: provider,
        phone: phone,
        api_key: apiKey,
        account_sid: sid,
        auth_token: token,
        from_whatsapp: fromWa,
        enabled: true
      };

      try {
        const resp = await fetch('/api/whatsapp/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        statusBox.style.display = 'block';
        statusBox.style.background = 'rgba(16, 185, 129, 0.12)';
        statusBox.style.border = '1px solid var(--zoho-emerald)';
        statusBox.style.color = 'var(--zoho-emerald)';
        statusBox.innerText = data.message || 'Configuration saved.';
        checkWhatsAppHeaderStatus();
        setTimeout(closeWhatsAppModal, 1200);
      } catch (e) {
        statusBox.style.display = 'block';
        statusBox.style.background = 'rgba(225, 29, 72, 0.12)';
        statusBox.style.border = '1px solid var(--zoho-crimson)';
        statusBox.style.color = 'var(--zoho-crimson)';
        statusBox.innerText = 'Save error: ' + e;
      }
    }

    async function testWhatsAppLive() {
      const provider = document.getElementById('waProviderSelect').value;
      const phone = document.getElementById('waPhoneInput').value.trim();
      const apiKey = document.getElementById('waApiKeyInput').value.trim();
      const sid = document.getElementById('waTwilioSidInput').value.trim();
      const token = document.getElementById('waTwilioTokenInput').value.trim();
      const fromWa = document.getElementById('waTwilioFromInput').value.trim();
      const statusBox = document.getElementById('waStatusMsg');

      if (!phone) {
        alert('Please enter recipient WhatsApp phone number with country code (e.g. +919876543210).');
        return;
      }

      statusBox.style.display = 'block';
      statusBox.style.background = 'rgba(15, 111, 222, 0.12)';
      statusBox.style.border = '1px solid var(--zoho-blue)';
      statusBox.style.color = 'var(--zoho-blue)';
      statusBox.innerText = 'Dispatching live test alert to ' + phone + ' via ' + provider.toUpperCase() + '...';

      const payload = {
        provider: provider,
        phone: phone,
        api_key: apiKey,
        account_sid: sid,
        auth_token: token,
        from_whatsapp: fromWa
      };

      try {
        const resp = await fetch('/api/whatsapp/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();
        if (data.ok) {
          statusBox.style.background = 'rgba(16, 185, 129, 0.12)';
          statusBox.style.border = '1px solid var(--zoho-emerald)';
          statusBox.style.color = 'var(--zoho-emerald)';
          statusBox.innerText = 'SUCCESS: ' + data.message;
          checkWhatsAppHeaderStatus();
        } else {
          statusBox.style.background = 'rgba(225, 29, 72, 0.12)';
          statusBox.style.border = '1px solid var(--zoho-crimson)';
          statusBox.style.color = 'var(--zoho-crimson)';
          statusBox.innerText = 'DISPATCH ERROR: ' + data.message;
        }
      } catch (e) {
        statusBox.style.background = 'rgba(225, 29, 72, 0.12)';
        statusBox.style.border = '1px solid var(--zoho-crimson)';
        statusBox.style.color = 'var(--zoho-crimson)';
        statusBox.innerText = 'Network error: ' + e;
      }
    }

    async function checkWhatsAppHeaderStatus() {
      try {
        const resp = await fetch('/api/whatsapp/status');
        const data = await resp.json();
        const dot = document.getElementById('waDot');
        const text = document.getElementById('waHeaderText');
        if (data.configured) {
          dot.className = 'status-dot status-dot-active';
          text.innerText = 'WHATSAPP: LINKED (' + data.phone + ')';
        } else {
          dot.className = 'status-dot status-dot-standby';
          text.innerText = 'WHATSAPP: READY';
        }
      } catch (e) {}
    }

    // =========================================================================
    // Telemetry, Video Player & Audit Trail Refresh
    // =========================================================================
    let lastEmergencyEventId = 0;
    async function refreshTelegramEmergencyAlerts() {
      try {
        const response = await fetch('/api/telegram_emergencies?after=' + lastEmergencyEventId);
        if (!response.ok) throw new Error('HTTP ' + response.status);
        const data = await response.json();
        if (!Array.isArray(data.events) || data.events.length === 0) return;

        const latest = data.events[data.events.length - 1];
        lastEmergencyEventId = Math.max(lastEmergencyEventId, ...data.events.map(event => Number(event.id) || 0));
        const isViolation = latest.event === 'CAMERA VIOLATION';
        document.getElementById('emergencyPanelTitle').textContent = isViolation
          ? 'VIOLATION WARNING'
          : (latest.event === 'CAMERA EMERGENCY' ? 'CAMERA SAFETY WARNING · VERIFY' : 'TELEGRAM EMERGENCY ALERT');
        document.getElementById('telegramEmergencyMessage').textContent = latest.details
          || (isViolation ? 'Verify the recorded evidence.' : 'Emergency assistance requested.');
        document.getElementById('telegramEmergencyTime').textContent =
          (isViolation ? 'Local violation warning • '
            : (latest.event === 'CAMERA EMERGENCY' ? 'Local camera warning • clip queued after recording • ' : 'Received from Telegram • '))
          + (latest.timestamp || '');
        document.getElementById('telegramEmergencyPanel').style.display = 'block';
      } catch (error) {
        console.error('Telegram emergency alert refresh failed:', error);
      }
    }

    function dismissTelegramEmergency() {
      document.getElementById('telegramEmergencyPanel').style.display = 'none';
    }

    function setDashboardView(view) {
      const ccmsSection = view === 'ccms-search' ? 'search' : 'overview';
      document.body.dataset.activeApp = 'ccms';
      document.body.dataset.ccmsSection = ccmsSection;
      document.getElementById('ccmsOverviewTab').classList.toggle('active', ccmsSection === 'overview');
      document.getElementById('ccmsSearchTab').classList.toggle('active', ccmsSection === 'search');
      document.getElementById('ccmsOverviewTab').setAttribute('aria-pressed', String(ccmsSection === 'overview'));
      document.getElementById('ccmsSearchTab').setAttribute('aria-pressed', String(ccmsSection === 'search'));
      updateCcmsSidebarState(ccmsSection === 'search' ? 'ccmsSearchTab' : 'ccmsOverviewTab');
    }

    function updateCcmsSidebarState(activeId) {
      document.querySelectorAll('.ccms-subnav .ccms-nav-item').forEach(button => {
        const active = button.id === activeId;
        button.classList.toggle('active', active);
        if (button.id === 'ccmsOverviewTab' || button.id === 'ccmsSearchTab') {
          button.setAttribute('aria-pressed', String(active));
        }
      });
    }

    function selectCcmsSection(section) {
      const selectedSection = section === 'search' ? 'search' : 'overview';
      setDashboardView(selectedSection === 'search' ? 'ccms-search' : 'ccms');
      updateCcmsSidebarState(selectedSection === 'search' ? 'ccmsSearchTab' : 'ccmsOverviewTab');
      history.replaceState(null, '', selectedSection === 'search' ? '#ccms-search' : '#ccms');
      if (selectedSection === 'search') {
        requestAnimationFrame(() => {
          const searchInput = document.getElementById('objectSearchInput');
          if (searchInput && searchInput.getClientRects().length) searchInput.focus();
        });
      }
    }

    function focusCcmsPanel(panelId, button) {
      setDashboardView('ccms');
      history.replaceState(null, '', '#ccms');
      updateCcmsSidebarState(button && button.id ? button.id : '');
      const panel = document.getElementById(panelId);
      if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function formatUptime(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const days = Math.floor(total / 86400);
      const hours = Math.floor((total % 86400) / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      if (days) return days + 'd ' + hours + 'h';
      if (hours) return hours + 'h ' + minutes + 'm';
      return minutes + 'm';
    }

    let selectedUploadedVideo = null;
    let uploadedVideoPreviewUrl = '';
    let uploadedVideoAnalysisTimer = null;
    const maxUploadedVideoBytes = 200 * 1024 * 1024;
    const allowedUploadedVideoExtensions = ['.avi', '.m4v', '.mkv', '.mov', '.mp4', '.webm'];

    function handleUploadedVideoSelection(input) {
      const file = input.files && input.files[0];
      const info = document.getElementById('uploadedVideoFileInfo');
      const preview = document.getElementById('uploadedVideoPreview');
      const analyzeButton = document.getElementById('analyzeUploadedVideoBtn');
      if (!file) return;
      const extension = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
      if (!allowedUploadedVideoExtensions.includes(extension)) {
        clearUploadedVideo();
        info.innerText = 'Unsupported video type. Choose MP4, MOV, MKV, AVI, WEBM, or M4V.';
        input.value = '';
        selectedUploadedVideo = null;
        analyzeButton.disabled = true;
        return;
      }
      if (file.size > maxUploadedVideoBytes) {
        clearUploadedVideo();
        info.innerText = 'This file exceeds the 200 MB upload limit.';
        input.value = '';
        selectedUploadedVideo = null;
        analyzeButton.disabled = true;
        return;
      }
      selectedUploadedVideo = file;
      if (uploadedVideoPreviewUrl) URL.revokeObjectURL(uploadedVideoPreviewUrl);
      uploadedVideoPreviewUrl = URL.createObjectURL(file);
      preview.onerror = () => {
        info.innerText = file.name + ' · Preview unavailable for this codec; the file can still be analyzed.';
      };
      preview.src = uploadedVideoPreviewUrl;
      document.getElementById('uploadedVideoPreviewWrap').style.display = 'block';
      info.innerText = file.name + ' · ' + (file.size / (1024 * 1024)).toFixed(1) + ' MB';
      analyzeButton.disabled = false;
      document.getElementById('uploadedVideoAnalysisResults').replaceChildren();
    }

    function clearUploadedVideo() {
      const input = document.getElementById('uploadedVideoFile');
      const preview = document.getElementById('uploadedVideoPreview');
      if (uploadedVideoPreviewUrl) URL.revokeObjectURL(uploadedVideoPreviewUrl);
      uploadedVideoPreviewUrl = '';
      selectedUploadedVideo = null;
      input.value = '';
      preview.onerror = null;
      preview.pause();
      preview.removeAttribute('src');
      preview.load();
      document.getElementById('uploadedVideoPreviewWrap').style.display = 'none';
      document.getElementById('uploadedVideoFileInfo').innerText = 'Choose an MP4, MOV, MKV, AVI, WEBM, or M4V file (maximum 200 MB).';
      document.getElementById('analyzeUploadedVideoBtn').disabled = true;
      document.getElementById('uploadedVideoAnalysisResults').replaceChildren();
    }

    function setUploadedVideoProgress(status, percent) {
      const progressWrap = document.getElementById('uploadedVideoAnalysisProgress');
      const meter = document.getElementById('uploadedVideoAnalysisMeter');
      progressWrap.style.display = 'block';
      meter.value = Math.max(0, Math.min(100, Number(percent) || 0));
      document.getElementById('uploadedVideoAnalysisPercent').innerText = Math.round(meter.value) + '%';
      document.getElementById('uploadedVideoAnalysisStatus').innerText = status;
    }

    function formatVideoTimestamp(seconds) {
      const total = Math.max(0, Math.floor(Number(seconds) || 0));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const remainder = total % 60;
      return hours > 0
        ? String(hours).padStart(2, '0') + ':' + String(minutes).padStart(2, '0') + ':' + String(remainder).padStart(2, '0')
        : String(minutes).padStart(2, '0') + ':' + String(remainder).padStart(2, '0');
    }

    function renderUploadedVideoAnalysis(data) {
      const results = document.getElementById('uploadedVideoAnalysisResults');
      results.replaceChildren();
      if (data.status === 'error') {
        results.innerText = data.error || 'Video analysis failed.';
        results.style.color = 'var(--zoho-crimson)';
        return;
      }
      results.style.color = 'var(--text-secondary)';
      const events = Array.isArray(data.events) ? data.events : [];
      const heading = document.createElement('div');
      heading.style.cssText = 'font-size:12px;font-weight:700;margin-bottom:8px;';
      heading.innerText = 'Analysis complete · ' + formatVideoTimestamp(data.duration_seconds || 0) +
        ' · ' + events.length + ' finding' + (events.length === 1 ? '' : 's');
      results.appendChild(heading);
      if (!events.length) {
        const empty = document.createElement('div');
        empty.style.cssText = 'padding:10px 12px;border:1px solid var(--zoho-border);border-radius:6px;font-size:12px;';
        empty.innerText = 'No plates, helmet violations, red-light crossings, speed events, or person-absence events were detected. AI results can be missed; review the video manually.';
        results.appendChild(empty);
        return;
      }
      const list = document.createElement('ul');
      list.style.cssText = 'margin:0;padding:0;list-style:none;display:grid;gap:7px;';
      events.forEach(event => {
        const item = document.createElement('li');
        const isViolation = event.type === 'helmetless' || event.type === 'red_light' || event.overspeed;
        const accent = isViolation ? 'var(--zoho-crimson)' : (event.type === 'plate_read' ? 'var(--zoho-blue)' : 'var(--zoho-border)');
        item.style.cssText = 'padding:10px 12px;border:1px solid rgba(15,111,222,.25);border-left:3px solid ' + accent + ';border-radius:5px;font-size:12px;';
        const title = document.createElement('strong');
        title.innerText = formatVideoTimestamp(event.timestamp_seconds) + ' · ' + (event.label || 'Event detected');
        const details = document.createElement('div');
        details.style.cssText = 'margin-top:4px;color:var(--text-muted);';
        let detailText = event.details || '';
        if (event.type === 'vehicle_speed') {
          detailText = 'Plate: ' + (event.plate_text || 'Unreadable') +
            ' · Speed: ' + Number(event.speed_kmh || 0).toFixed(1) + ' km/h' +
            ' · Limit: ' + Number(event.speed_limit_kmh || 0).toFixed(0) + ' km/h' +
            ' · ' + (event.overspeed ? (event.clip_sent ? 'OVERSPEED · clip sent to Telegram' : 'OVERSPEED · ' + (event.delivery_status || 'local only') + '') : 'Within configured limit');
        } else if (event.type === 'plate_read') {
          detailText = 'Plate: ' + (event.plate_text || 'Unreadable') +
            ' · OCR confidence: ' + Math.round(Number(event.confidence || 0) * 100) + '%' +
            ' · Video time: ' + formatVideoTimestamp(event.timestamp_seconds);
        } else if (event.type === 'helmetless') {
          detailText = 'NO HELMET · Plate: ' + (event.plate_text || 'Unreadable') +
            ' · Head confidence: ' + Math.round(Number(event.confidence || 0) * 100) + '%';
        } else if (event.type === 'red_light') {
          detailText = 'RED LIGHT · Plate: ' + (event.plate_text || 'Unreadable') +
            ' · Signal confidence: ' + Math.round(Number(event.signal_confidence || 0) * 100) + '%';
        }
        details.innerText = detailText;
        item.append(title, details);
        if (event.clip_file) {
          const link = document.createElement('a');
          link.href = '/video/' + encodeURIComponent(event.clip_file);
          link.target = '_blank';
          link.rel = 'noopener';
          link.style.cssText = 'display:inline-block;margin-top:6px;font-size:11px;font-weight:700;color:var(--zoho-blue);text-decoration:none;';
          link.innerText = 'Open annotated clip · ' + (event.clip_sent ? 'sent to Telegram' : (event.delivery_status || 'local only'));
          item.appendChild(link);
        }
        if (event.snapshot) {
          const image = document.createElement('img');
          image.src = event.snapshot;
          image.alt = 'Annotated detection at ' + formatVideoTimestamp(event.timestamp_seconds);
          image.style.cssText = 'display:block;width:100%;max-width:640px;margin-top:8px;border-radius:4px;';
          item.appendChild(image);
        }
        list.appendChild(item);
      });
      results.appendChild(list);
      const note = document.createElement('div');
      note.style.cssText = 'margin-top:8px;color:var(--text-muted);font-size:11px;';
      note.innerText = 'Review the video around each timestamp. Rectangles and confidence scores are AI estimates, not confirmed incidents.';
      results.appendChild(note);
    }

    async function pollUploadedVideoAnalysis(jobId) {
      try {
        const response = await fetch('/api/video-analysis/status?job_id=' + encodeURIComponent(jobId), { cache: 'no-store' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Could not read the analysis status.');
        if (data.status === 'queued' || data.status === 'running') {
          setUploadedVideoProgress('AI analysis · ' + formatVideoTimestamp(data.processed_seconds || 0) + ' / ' + formatVideoTimestamp(data.duration_seconds || 0), data.progress);
          uploadedVideoAnalysisTimer = setTimeout(() => pollUploadedVideoAnalysis(jobId), 1000);
          return;
        }
        document.getElementById('analyzeUploadedVideoBtn').disabled = !selectedUploadedVideo;
        setUploadedVideoProgress(data.status === 'completed' ? 'Analysis complete' : 'Analysis failed', data.progress);
        renderUploadedVideoAnalysis(data);
        if (data.status === 'completed' && (data.events || []).some(event => ['queued', 'retry_pending'].includes(event.delivery_status))) {
          uploadedVideoAnalysisTimer = setTimeout(() => pollUploadedVideoAnalysis(jobId), 3000);
        }
      } catch (error) {
        document.getElementById('analyzeUploadedVideoBtn').disabled = !selectedUploadedVideo;
        setUploadedVideoProgress('Could not read analysis status: ' + error.message, 0);
      }
    }

    function analyzeUploadedVideo() {
      if (!selectedUploadedVideo) return;
      const speedLimitKmh = Number(document.getElementById('uploadedSpeedLimit').value);
      if (!Number.isFinite(speedLimitKmh) || speedLimitKmh < 1 || speedLimitKmh > 300) {
        setUploadedVideoProgress('Enter a speed limit between 1 and 300 km/h.', 0);
        return;
      }
      if (!window.confirm(
        'This uploaded clip may be historical. AI screening attempts plate OCR, helmet checks, red-light crossing checks and uncalibrated speed estimates. Readings are not guaranteed and require human review. Potential violations produce local warnings and annotated clips queued for the next 3-minute Telegram batch, not immediate messages. The screening speed limit is ' +
        speedLimitKmh + ' km/h. Continue?'
      )) return;
      const analyzeButton = document.getElementById('analyzeUploadedVideoBtn');
      const results = document.getElementById('uploadedVideoAnalysisResults');
      if (uploadedVideoAnalysisTimer) clearTimeout(uploadedVideoAnalysisTimer);
      analyzeButton.disabled = true;
      results.replaceChildren();
      setUploadedVideoProgress('Uploading video...', 0);
      const xhr = new XMLHttpRequest();
      const parameters = new URLSearchParams({
        filename: selectedUploadedVideo.name,
        speed_limit_kmh: String(speedLimitKmh)
      });
      xhr.open('POST', '/api/video-analysis?' + parameters.toString());
      xhr.setRequestHeader('Content-Type', 'application/octet-stream');
      xhr.upload.onprogress = event => {
        if (event.lengthComputable) {
          setUploadedVideoProgress('Uploading video...', event.loaded * 100 / event.total);
        }
      };
      xhr.onerror = () => {
        analyzeButton.disabled = !selectedUploadedVideo;
        setUploadedVideoProgress('Upload failed. Check the connection and try again.', 0);
      };
      xhr.onload = () => {
        let data;
        try {
          data = JSON.parse(xhr.responseText);
        } catch {
          analyzeButton.disabled = !selectedUploadedVideo;
          setUploadedVideoProgress('The server returned an invalid response.', 0);
          return;
        }
        if (xhr.status !== 202 || !data.job_id) {
          analyzeButton.disabled = !selectedUploadedVideo;
          setUploadedVideoProgress(data.error || 'The video could not be queued.', 0);
          return;
        }
        setUploadedVideoProgress('Upload complete · preparing AI analysis...', 0);
        pollUploadedVideoAnalysis(data.job_id);
      };
      xhr.send(selectedUploadedVideo);
    }

    function updateCcmsStatusMetrics(status) {
      const cameraOnline = Boolean(status.running && status.has_frame);
      document.getElementById('ccmsKpiCameras').textContent = cameraOnline ? '1 / 1' : '0 / 1';
      document.getElementById('ccmsKpiCameraNote').textContent = cameraOnline
        ? 'Camera ' + String(status.camera) + ' online'
        : status.running ? 'Waiting for video frame' : 'Standby';
      document.getElementById('ccmsKpiAlerts').textContent =
        Array.isArray(status.violations) ? String(status.violations.length) : '0';
      document.getElementById('ccmsKpiUptime').textContent = formatUptime(status.uptime_sec);
      const operational = document.querySelector('.ccms-operational');
      operational.dataset.state = cameraOnline ? 'operational' : status.running ? 'connecting' : 'standby';
      document.getElementById('ccmsOperationalStatus').textContent =
        cameraOnline ? 'OPERATIONAL' : status.running ? 'CONNECTING' : 'STANDBY';
    }

    window.addEventListener('hashchange', function() {
      const requestedView = window.location.hash.slice(1).toLowerCase();
      setDashboardView(requestedView === 'ccms-search' ? 'ccms-search' : 'ccms');
    });
    let refreshInFlight = false;
    let lastRadarSignature = '';
    let lastPlaylistSignature = '';
    let lastAuditSignature = '';
    let lastIncidentsRefresh = 0;
    async function refreshData() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      try {
        await refreshTelegramEmergencyAlerts();
        const sResp = await fetch('/api/status');
        const sData = await sResp.json();
        updateCcmsStatusMetrics(sData);
        const ai = sData.open_vocabulary || {};
        const aiStatus = document.getElementById('objectAiStatus');
        const objectAiReady = sData.primary_ai_ready && ai.state === 'ready' && ai.reference_photos === 'ready';
        const objectAiUnavailable = ['error', 'missing'].includes(ai.state) || ['error', 'missing'].includes(ai.reference_photos);
        aiStatus.innerText = objectAiReady ? 'MULTI-OBJECT AI READY' : objectAiUnavailable ? 'REFERENCE AI UNAVAILABLE' : 'OBJECT AI WARMING UP';
        aiStatus.style.color = objectAiReady ? 'var(--zoho-emerald)' : 'var(--zoho-gold)';
        aiStatus.title = objectAiReady ? 'Confirmed objects receive individual boxes. Hold small objects clearly in view for two detail scans.' : 'Basic camera detection can start before reference-object detection is ready.';
        const topBtn = document.getElementById('topToggleBtn');
        const camSelect = document.getElementById('cameraSelect');
        const streamImg = document.getElementById('liveStreamImg');
        const heroLaunchpad = document.getElementById('streamHeroLaunchpad');
        const wasCameraActive = topBtn.classList.contains('btn-zoho-stop');
        isCameraActive = Boolean(sData.running);

        if (sData.running) {
          if (!wasCameraActive) {
            topBtn.className = 'btn-zoho btn-zoho-stop';
            topBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"></rect></svg><span id="topToggleText">STOP CAMERA</span>';
          }
          camSelect.disabled = true;
          document.getElementById('engineDot').className = 'status-dot status-dot-active';
          if (!streamImg.src || !streamImg.style.display || streamImg.style.display === 'none') {
            streamImg.src = '/video_feed?t=' + Date.now();
          }
          streamImg.style.display = 'block';
          heroLaunchpad.style.display = 'none';
          document.getElementById('radarMonitorStatus').innerText = sData.has_frame ? '13 MONITORS ACTIVE' : 'WAITING FOR CAMERA';
          document.getElementById('radarMonitorStatus').style.color = sData.has_frame ? 'var(--zoho-emerald)' : 'var(--zoho-gold)';
          const streamStatus = document.getElementById('streamStatusBadge');
          streamStatus.innerText = sData.has_frame ? 'LIVE • ' + aiStatus.innerText : 'WAITING FOR VIDEO FRAME';
          streamStatus.style.color = sData.has_frame && objectAiReady ? 'var(--zoho-emerald)' : 'var(--zoho-gold)';
          document.getElementById('engineStatusText').innerText = sData.has_frame ? 'CAMERA: ONLINE' : 'CAMERA: WAITING FOR VIDEO';
          document.getElementById('fpsCounter').innerText = 'FPS: ' + sData.fps.toFixed(1);

        }
        else {
          if (wasCameraActive) {
            topBtn.className = 'btn-zoho btn-zoho-primary';
            topBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg><span id="topToggleText">START CAMERA</span>';
          }
          camSelect.disabled = false;
          streamImg.src = '';
          streamImg.style.display = 'none';
          heroLaunchpad.style.display = 'flex';
          document.getElementById('streamStatusBadge').innerText = 'CAMERA STANDBY';
          document.getElementById('streamStatusBadge').style.color = 'var(--text-muted)';
          document.getElementById('engineStatusText').innerText = 'CAMERA: STANDBY';
          document.getElementById('engineDot').className = 'status-dot status-dot-standby';
          document.getElementById('radarList').innerHTML = '';
          document.getElementById('radarMonitorStatus').innerText = '13 MONITORS STANDBY';
          document.getElementById('radarMonitorStatus').style.color = 'var(--text-muted)';
          document.getElementById('fpsCounter').innerText = 'FPS: 0.0';
        }

        const batch = sData.telegram_batch || {};
        const secondsLeft = Math.ceil(sData.remaining);
        document.getElementById('ringCountdown').innerText = String(Math.floor(secondsLeft / 60)).padStart(2, '0') + ':' + String(secondsLeft % 60).padStart(2, '0');
        document.getElementById('cycleProgressCircle').style.strokeDashoffset = 2 * Math.PI * 40 * (1 - Math.min(1, sData.elapsed / 180));
        document.getElementById('telegramBatchStatus').innerText = 'TELEGRAM: ' + (batch.sending ? 'SENDING BATCH' : (batch.pending_count || 0) + ' CLIPS QUEUED') + ' · ' + (batch.last_result || 'No batch sent');

        if (sData.radar) {
          const radarSignature = JSON.stringify(sData.radar);
          if (radarSignature !== lastRadarSignature) {
            lastRadarSignature = radarSignature;
            const rContainer = document.getElementById('radarList');
            rContainer.innerHTML = Object.entries(sData.radar).map(([key, item]) => `
              <div class="radar-row">
                <span>${key}</span>
                <span class="badge badge-${item[1]}">${item[0]}</span>
              </div>
            `).join('');
          }
        }

        const nowMs = Date.now();
        if (nowMs - lastIncidentsRefresh < 4000) return;
        lastIncidentsRefresh = nowMs;

        const iResp = await fetch('/api/incidents');
        const iData = await iResp.json();

        const pList = document.getElementById('playlist');
        document.getElementById('clipCountLabel').innerText = iData.clips.length + ' Files';
        document.getElementById('ccmsKpiClips').textContent = String(iData.clips.length);

        const playlistSignature = iData.clips.map(c => c.filename).join('|');
        if (playlistSignature !== lastPlaylistSignature) {
          lastPlaylistSignature = playlistSignature;
          if (iData.clips.length === 0) {
            pList.innerHTML = '<p style="text-align: center; color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); padding: 24px;">No incident video clips recorded yet.</p>';
          } else {
            pList.innerHTML = iData.clips.map(c => {
              const isActive = (c.filename === currentPlayingFile) ? ' active' : '';
              const safeTitle = (c.title || 'Incident Clip').replace(/"/g, '&quot;');
              const safeFile = (c.filename || '').replace(/"/g, '&quot;');
              return `
                <div class="playlist-item${isActive}" data-file="${safeFile}" data-title="${safeTitle}" onclick="playIncidentClip(this.dataset.file, this.dataset.title)">
                  <div>
                    <div style="font-size: 13px; font-weight: 700; color: var(--text-white);">${safeTitle}</div>
                    <div style="font-size: 11px; color: var(--text-secondary); font-family: var(--font-mono);">${c.time} &bull; MP4</div>
                  </div>
                  <span style="font-size: 11px; color: var(--zoho-blue); font-weight: 700;">PLAY</span>
                </div>
              `;
            }).join('');
          }
        } else {
          document.querySelectorAll('#playlist .playlist-item').forEach(item => {
            item.classList.toggle('active', item.dataset.file === currentPlayingFile);
          });
        }

        const tBody = document.getElementById('auditTableBody');
        const auditSignature = JSON.stringify(iData.records);
        if (auditSignature !== lastAuditSignature) {
          lastAuditSignature = auditSignature;
          if (iData.records.length === 0) {
            tBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted); padding: 24px;">No records found.</td></tr>';
          } else {
            tBody.innerHTML = iData.records.map(r => {
              const safeVid = (r.video || '').replace(/"/g, '&quot;');
              const safeEvt = (r.event || '').replace(/"/g, '&quot;');
              const hasVid = r.video && r.video !== '-' && r.video.endsWith('.mp4');
              const vidCell = hasVid
                ? `<button type="button" class="btn-play-clip" data-file="${safeVid}" data-title="${safeEvt}" onclick="playIncidentClip(this.dataset.file, this.dataset.title)">PLAY</button>`
                : `<span style="color: var(--text-muted);">-</span>`;
              return `
                <tr>
                  <td style="font-family: var(--font-mono); font-size: 11px;">${r.timestamp}</td>
                  <td style="font-weight: 700; color: var(--text-white);">${r.event}</td>
                  <td style="color: var(--text-secondary);">${r.details}</td>
                  <td>${vidCell}</td>
                </tr>
              `;
            }).join('');
          }
        }
      } catch (e) {
        console.error(e);
      } finally {
        refreshInFlight = false;
      }
    }

    let currentPlayingFile = '';

    function playIncidentClip(rawFilename, title) {
      const player = document.getElementById('incidentVideoPlayer');
      const emptyText = document.getElementById('playerEmptyText');
      const activeTitle = document.getElementById('activeVideoTitle');
      const statusBadge = document.getElementById('videoPlaybackStatus');
      const directLink = document.getElementById('videoDirectLink');

      if (!rawFilename || rawFilename === '-') return;

      // Extract clean filename (no query params, no path traversal)
      const cleanFile = rawFilename.split('?')[0].split('/').pop().split('\\\\').pop();
      if (!cleanFile) return;

      currentPlayingFile = cleanFile;
      if (emptyText) emptyText.style.display = 'none';
      if (player) player.style.display = 'block';

      if (activeTitle) {
        activeTitle.innerText = (title || 'Incident Clip') + ' [' + cleanFile + ']';
      }
      if (directLink) {
        directLink.href = '/video/' + encodeURIComponent(cleanFile);
        directLink.style.display = 'inline-block';
      }
      if (statusBadge) {
        statusBadge.innerText = 'LOADING';
        statusBadge.style.color = 'var(--zoho-gold)';
        statusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
      }

      // Safe clean re-initialization of video element
      try {
        player.pause();
        player.removeAttribute('src');
        player.load();
      } catch (e) {}

      player.src = '/video/' + encodeURIComponent(cleanFile);

      player.onloadedmetadata = function() {
        try {
          player.currentTime = 0;
        } catch (e) {}
      };

      player.onloadeddata = function() {
        if (statusBadge) {
          statusBadge.innerText = 'PLAYING';
          statusBadge.style.color = 'var(--zoho-emerald)';
          statusBadge.style.borderColor = 'rgba(16, 185, 129, 0.4)';
        }
        const p = player.play();
        if (p !== undefined) {
          p.catch(function(err) {
            console.log('Unmuted autoplay restricted by browser policy, attempting muted playback:', err);
            player.muted = true;
            player.play().catch(function(err2) {
              console.log('Browser requires explicit click on controls:', err2);
              if (statusBadge) {
                statusBadge.innerText = 'CLICK PLAY TO WATCH';
                statusBadge.style.color = 'var(--zoho-blue)';
                statusBadge.style.borderColor = 'rgba(15, 111, 222, 0.4)';
              }
            });
          });
        }
      };

      player.onerror = function() {
        if (statusBadge) {
          statusBadge.innerText = 'PLAYBACK ERROR';
          statusBadge.style.color = 'var(--zoho-crimson)';
          statusBadge.style.borderColor = 'rgba(225, 29, 72, 0.4)';
        }
      };

      player.onended = function() {
        if (statusBadge) {
          statusBadge.innerText = 'REPLAY';
          statusBadge.style.color = 'var(--text-secondary)';
          statusBadge.style.borderColor = 'rgba(148, 163, 184, 0.2)';
        }
      };

      player.load();

      document.querySelectorAll('.playlist-item').forEach(el => {
        if (el.dataset.file === cleanFile) {
          el.classList.add('active');
        } else {
          el.classList.remove('active');
        }
      });
    }

    function escapeHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }

    // Send AI Chat Message
    function sendAiChatMessage() {
      const input = document.getElementById('aiChatInput');
      const text = input ? input.value.trim() : '';
      if (!text) return;
      input.value = '';

      const history = document.getElementById('aiChatHistory');
      if (history) {
        const userDiv = document.createElement('div');
        userDiv.style.cssText = 'background: rgba(15,111,222,0.18); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--zoho-blue); align-self: flex-end; max-width: 85%;';
        userDiv.innerHTML = '<strong style="color: var(--zoho-blue);">You:</strong> ' + escapeHtml(text);
        history.appendChild(userDiv);
        history.scrollTop = history.scrollHeight;
      }

      fetch('/api/ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text })
      })
      .then(res => res.json())
      .then(data => {
        if (history) {
          history.appendChild(renderAiReply(data));
          history.scrollTop = history.scrollHeight;
        }
      })
      .catch(err => {
        if (history) {
          const errDiv = document.createElement('div');
          errDiv.style.cssText = 'background: rgba(225,29,72,0.15); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--zoho-crimson);';
          errDiv.innerHTML = '<strong style="color: var(--zoho-crimson);">Error:</strong> Could not connect to AI assistant.';
          history.appendChild(errDiv);
        }
      });
    }

    function renderAiReply(data) {
      const aiDiv = document.createElement('div');
      aiDiv.style.cssText = 'background: rgba(30,41,59,0.7); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--zoho-emerald); max-width: 90%;';

      const textDiv = document.createElement('div');
      textDiv.innerHTML = '<strong style="color: var(--zoho-emerald);">Vision AI:</strong> ' + escapeHtml(data.reply || data.response || 'Understood.');
      aiDiv.appendChild(textDiv);

      const actions = Array.isArray(data.actions) ? data.actions : [];
      if (actions.length) {
        const bar = document.createElement('div');
        bar.style.cssText = 'display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;';
        actions.forEach(item => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = item.label || 'Action';
          btn.style.cssText = 'background: rgba(37,99,235,0.18); border: 1px solid rgba(59,130,246,0.45); color: #93c5fd; font-size: 11px; font-weight: 600; padding: 5px 10px; border-radius: 999px; cursor: pointer;';
          btn.onmouseenter = () => { btn.style.background = 'rgba(37,99,235,0.35)'; };
          btn.onmouseleave = () => { btn.style.background = 'rgba(37,99,235,0.18)'; };
          btn.onclick = () => runAiAction(item);
          bar.appendChild(btn);
        });
        aiDiv.appendChild(bar);
      }
      return aiDiv;
    }

    function runAiAction(item) {
      if (!item) return;
      if (item.clip) {
        playIncidentClip(item.clip, item.label);
        const player = document.getElementById('incidentVideoPlayer');
        if (player) player.scrollIntoView({ behavior: 'smooth', block: 'center' });
        return;
      }
      if (item.url) {
        window.open(item.url, '_blank');
        return;
      }
      if (item.command) {
        const input = document.getElementById('aiChatInput');
        if (input) input.value = item.command;
        sendAiChatMessage();
      }
    }

    for (const button of document.querySelectorAll('.ccms-nav-item')) {
      const label = button.querySelector('.ccms-nav-text').firstChild.textContent.trim();
      button.setAttribute('aria-label', label);
      button.title = label;
    }
    // Auto-refresh telemetry every 2 seconds
    const requestedView = window.location.hash.slice(1).toLowerCase();
    setDashboardView(requestedView === 'ccms-search' ? 'ccms-search' : 'ccms');
    refreshData();
    setInterval(refreshData, 2000);

    // Initial check on load
    checkWhatsAppHeaderStatus();
    searchObjects(''); // Show initial tracked objects
  </script>
</body>
</html>
"""


class CommandCenterHandler(http.server.BaseHTTPRequestHandler):
    def _send_json_response(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_COMMAND_CENTER.encode("utf-8"))
            return

        if path == "/api/video-analysis/status":
            query = urllib.parse.parse_qs(parsed.query)
            job_id = query.get("job_id", [None])[0]
            self._send_json_response(200, GLOBAL_VIDEO_ANALYZER.status(job_id))
            return

        if path in ("/logo.png", "/favicon.ico", "/assets/logo.png"):
            logo_path = ROOT / "visionai_logo.png"
            if logo_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                # Short cache so a replaced logo shows up immediately, ?v= busts old copies.
                self.send_header("Cache-Control", "no-cache, max-age=0, must-revalidate")
                self.send_header("Content-Length", str(logo_path.stat().st_size))
                self.end_headers()
                with logo_path.open("rb") as lf:
                    self.wfile.write(lf.read())
                return
            else:
                self.send_error(404, "Logo not found")
                return

        if path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            try:
                while GLOBAL_CAMERA.is_running:
                    with GLOBAL_CAMERA.lock:
                        frame_bytes = GLOBAL_CAMERA.latest_jpeg
                    if frame_bytes:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.04)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass
            return

        if path == "/api/status":
            delivery = GLOBAL_DELIVERY.status()
            remaining = delivery["remaining_seconds"]
            elapsed = delivery["interval_seconds"] - remaining
            with GLOBAL_CAMERA.lock:
                has_frame = GLOBAL_CAMERA.latest_jpeg is not None
                resolution = (
                    f"{GLOBAL_CAMERA.frame_width}x{GLOBAL_CAMERA.frame_height}"
                    if has_frame else ""
                )
            resp_data = {
                "running": GLOBAL_CAMERA.is_running,
                "uptime_sec": max(0, int(time.time() - APP_START_TIME)),
                "has_frame": has_frame,
                "camera": GLOBAL_CAMERA.camera_index,
                "resolution": resolution,
                "fps": round(GLOBAL_CAMERA.fps, 1),
                "inference_ms": round(GLOBAL_CAMERA.inference_ms, 1),
                "device": DETECTION_DEVICE or "auto",
                "last_error": GLOBAL_CAMERA.last_error,
                "primary_ai_ready": GLOBAL_CAMERA.model is not None,
                "open_vocabulary": GLOBAL_MICRO_AI.status(),
                "elapsed": elapsed,
                "remaining": remaining,
                "cycle_active": True,
                "telegram_batch": delivery,
                "confirmation_seconds": GLOBAL_CAMERA.engine.confirmation_sec,
                "has_clip": bool(GLOBAL_CAMERA.engine.pending_video_path is not None),
                "violations": GLOBAL_CAMERA.last_eval.get("active_violations", []),
                "radar": (
                    GLOBAL_CAMERA.last_eval.get("radar_status", {})
                    if GLOBAL_CAMERA.is_running
                    else {name: ("CAMERA OFFLINE", "muted") for name in RADAR_MONITORS}
                ),
            }
            self._send_json_response(200, resp_data)
            return

        if path == "/api/telegram_emergencies":
            try:
                after_id = int(urllib.parse.parse_qs(parsed.query).get("after", ["0"])[0])
            except ValueError:
                self.send_error(400, "Invalid emergency event cursor")
                return
            with TELEGRAM_EMERGENCY_EVENTS_LOCK:
                events = [event for event in TELEGRAM_EMERGENCY_EVENTS if event["id"] > after_id]
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"events": events}).encode("utf-8"))
            return

        if path == "/api/incidents":
            now_mono = time.monotonic()
            with INCIDENTS_CACHE_LOCK:
                cached = INCIDENTS_CACHE["payload"]
                if cached is not None and now_mono - INCIDENTS_CACHE["at"] < 2.0:
                    self._send_json_response(200, cached)
                    return

            clips = []
            if INCIDENTS_DIR.exists():
                for p in sorted(INCIDENTS_DIR.glob("*.mp4"), key=os.path.getmtime, reverse=True):
                    if GLOBAL_OBJECT_TRACKER.is_object_clip_quarantined(p.name):
                        continue
                    name_parts = p.stem.split("_", 3)
                    title = name_parts[3].replace("_", " ").title() if len(name_parts) >= 4 else p.stem
                    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    clips.append({
                        "filename": p.name,
                        "title": title,
                        "time": mtime,
                    })

            records = []
            if LOG_FILE.exists():
                try:
                    with LOG_FILE.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in collections.deque(reader, maxlen=20):
                            records.append({
                                "timestamp": row.get("timestamp", "-"),
                                "event": row.get("event", "-"),
                                "details": row.get("details", "-"),
                                "video": row.get("video_clip", "-"),
                            })
                    records.reverse()
                except Exception:
                    pass

            resp_data = {"clips": clips, "records": records}
            with INCIDENTS_CACHE_LOCK:
                INCIDENTS_CACHE["at"] = now_mono
                INCIDENTS_CACHE["payload"] = resp_data
            self._send_json_response(200, resp_data)
            return

        if path.startswith("/snapshot/"):
            clean_name = Path(urllib.parse.unquote(path[10:])).name
            file_path = SNAPSHOTS_DIR / clean_name
            if file_path.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache, max-age=0, must-revalidate")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.end_headers()
                with file_path.open("rb") as snapshot_file:
                    self.wfile.write(snapshot_file.read())
                return
            self.send_error(404, "Snapshot not found")
            return

        if path.startswith("/video/"):
            raw_filename = path[7:]  # strips "/video/"
            filename = urllib.parse.unquote(raw_filename)
            if "?" in filename:
                filename = filename.split("?")[0]
            clean_name = Path(filename).name
            file_path = INCIDENTS_DIR / clean_name

            if file_path.exists() and file_path.is_file():
                file_size = file_path.stat().st_size
                range_header = self.headers.get("Range")

                if range_header and range_header.startswith("bytes="):
                    range_spec = range_header[6:].strip()
                    parts = range_spec.split("-")
                    if not parts[0] and parts[1]:
                        # Suffix range: bytes=-N
                        suffix = int(parts[1])
                        start = max(0, file_size - suffix)
                        end = file_size - 1
                    elif parts[0] and not parts[1]:
                        # Prefix range: bytes=N-
                        start = int(parts[0])
                        end = file_size - 1
                    else:
                        # Closed range: bytes=N-M
                        start = int(parts[0])
                        end = min(file_size - 1, int(parts[1]))

                    if start > end or start >= file_size:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.end_headers()
                        return

                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "Range, Content-Range, Content-Type")
                    self.end_headers()

                    try:
                        with file_path.open("rb") as f:
                            f.seek(start)
                            bytes_remaining = length
                            chunk_size = 64 * 1024
                            while bytes_remaining > 0:
                                read_bytes = min(chunk_size, bytes_remaining)
                                data = f.read(read_bytes)
                                if not data:
                                    break
                                self.wfile.write(data)
                                bytes_remaining -= len(data)
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        pass
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "Range, Content-Range, Content-Type")
                    self.end_headers()

                    try:
                        with file_path.open("rb") as f:
                            chunk_size = 64 * 1024
                            while True:
                                data = f.read(chunk_size)
                                if not data:
                                    break
                                self.wfile.write(data)
                    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                        pass
                return
            else:
                self.send_error(404, "Clip not found")
                return

        if path == "/api/whatsapp/status":
            cfg = whatsapp_manager.load_whatsapp_config() or {}
            self._send_json_response(200, {
                "configured": bool(cfg.get("phone")),
                "phone": cfg.get("phone", ""),
                "provider": cfg.get("provider", "callmebot"),
                "has_api_key": bool(cfg.get("api_key")),
                "account_sid": cfg.get("account_sid", ""),
                "has_auth_token": bool(cfg.get("auth_token")),
                "from_whatsapp": cfg.get("from_whatsapp", "+14155238886"),
                "enabled": cfg.get("enabled", True),
            })
            return

        if path == "/api/objects/search":
            query = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            global GLOBAL_ACTIVE_SEARCH_TARGET
            GLOBAL_ACTIVE_SEARCH_TARGET = query.strip().lower() if query.strip() else None
            results = GLOBAL_OBJECT_TRACKER.search(query)
            self._send_json_response(200, {
                "results": results,
                "query": query,
                "count": len(results),
                "active_target": GLOBAL_ACTIVE_SEARCH_TARGET or ""
            })
            return

        if path == "/api/objects/list":
            results = GLOBAL_OBJECT_TRACKER.get_all_tracked()
            self._send_json_response(200, {"results": results, "count": len(results)})
            return

        if path.startswith("/snapshot/"):
            raw_name = path[10:]
            clean_name = Path(urllib.parse.unquote(raw_name)).name
            snap_path = SNAPSHOT_DIR / clean_name
            if snap_path.exists() and snap_path.is_file():
                try:
                    snap_bytes = snap_path.read_bytes()
                except OSError:
                    self.send_error(404, "Snapshot not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(snap_bytes)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(snap_bytes)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass
                return
            else:
                self.send_error(404, "Snapshot not found")
                return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/video-analysis":
            if GLOBAL_CAMERA.is_running:
                self._send_json_response(409, {
                    "error": "Stop live camera surveillance before analyzing an uploaded video.",
                })
                return
            if GLOBAL_VIDEO_ANALYZER.is_active:
                self._send_json_response(409, {
                    "error": "Another uploaded video is already being analyzed.",
                })
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json_response(400, {"error": "Invalid upload size."})
                return
            if content_length <= 0:
                self._send_json_response(400, {"error": "The uploaded video is empty."})
                return
            if content_length > MAX_UPLOADED_VIDEO_BYTES:
                self._send_json_response(413, {"error": "Video uploads are limited to 200 MB."})
                return

            query = urllib.parse.parse_qs(parsed.query)
            raw_filename = urllib.parse.unquote(query.get("filename", ["video"])[0])
            filename = Path(raw_filename).name
            extension = Path(filename).suffix.lower()
            if extension not in ALLOWED_VIDEO_EXTENSIONS:
                self._send_json_response(415, {
                    "error": "Unsupported video format. Use MP4, MOV, MKV, AVI, WEBM, or M4V.",
                })
                return
            try:
                analysis_settings = validate_analysis_settings({
                    "speed_limit_kmh": float(query.get("speed_limit_kmh", ["60"])[0]),
                })
            except (ValueError, TypeError) as exc:
                self._send_json_response(400, {"error": f"Invalid speed limit: {exc}"})
                return

            upload_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="ccms_video_",
                    suffix=extension,
                    delete=False,
                ) as upload_file:
                    upload_path = upload_file.name
                    remaining = content_length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("The upload ended before the complete video was received.")
                        upload_file.write(chunk)
                        remaining -= len(chunk)
                job_id = GLOBAL_VIDEO_ANALYZER.start(upload_path, filename, analysis_settings)
                self._send_json_response(202, {"job_id": job_id, "status": "queued"})
            except RuntimeError as exc:
                self._send_json_response(409, {"error": str(exc)})
                if upload_path:
                    Path(upload_path).unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                self._send_json_response(400, {"error": f"Could not accept the uploaded video: {exc}"})
                if upload_path:
                    Path(upload_path).unlink(missing_ok=True)
            return

        if path == "/api/camera/start":
            if GLOBAL_VIDEO_ANALYZER.is_active:
                self._send_json_response(409, {
                    "ok": False,
                    "message": "Stop uploaded-video analysis before starting live camera surveillance.",
                })
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            cam_id = data.get("camera", 0)
            conf = data.get("confidence", GLOBAL_CAMERA.confidence)
            ok, msg = GLOBAL_CAMERA.start(cam_id, conf)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "ok": ok,
                "message": msg,
                "camera": GLOBAL_CAMERA.camera_index,
            }).encode("utf-8"))
            return

        if path == "/api/camera/stop":
            print(
                f"[Camera Control] Stop requested over HTTP from {self.client_address[0]} "
                f"(UA: {self.headers.get('User-Agent', '?')[:90]}, Referer: {self.headers.get('Referer', '-')[:90]}).",
                flush=True,
            )
            ok, msg = GLOBAL_CAMERA.stop()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}).encode("utf-8"))
            return

        if path == "/api/dispatch_now":
            started = GLOBAL_DELIVERY.dispatch(time.monotonic(), force=True)
            message = "Pending clip batch started; delivery acknowledgements will appear in the dashboard." if started else "No pending clips, or another batch is still sending. No duplicate was sent."
            self._send_json_response(200, {"ok": started, "message": message})
            return

        if path == "/api/objects/set_target":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            target = data.get("target", "").strip().lower()
            global GLOBAL_ACTIVE_SEARCH_TARGET
            GLOBAL_ACTIVE_SEARCH_TARGET = target if target else None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "target": GLOBAL_ACTIVE_SEARCH_TARGET or ""}).encode("utf-8"))
            return

        # Vision AI Assistant Interactive Chat API
        if path == "/api/ai/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            msg = str(data.get("message", "")).strip().lower()
            reply = ""
            actions: list[dict[str, str]] = []

            def action(label: str, command: str = "", clip: str = "", url: str = "") -> None:
                actions.append({"label": label, "command": command, "clip": clip, "url": url})

            def has_word(*words: str) -> bool:
                return any(re.search(rf"\b{re.escape(word)}\b", msg) for word in words)

            def add_clip_action() -> None:
                clip = latest_incident_clip()
                if clip:
                    action("Open Latest Clip", clip=clip["filename"])

            if not msg:
                reply = "Please enter a question or instruction. Type 'help' to see what I can do."
            elif has_word("unmute") or "resume siren" in msg or "enable siren" in msg:
                speaker_alarm.clear_mute()
                handle_two_way_event("SIREN UNMUTED (AI CHAT)", "User re-enabled the on-site siren via Vision AI Chat.")
                reply = "Siren is active again — emergency alarms will sound normally."
            elif re.search(r"\bmute(?:\s+(?:the\s+)?(?:siren|alarm))?(?:\s+for)?\s+\d", msg):
                mute_match = re.search(
                    r"\bmute(?:\s+(?:the\s+)?(?:siren|alarm))?(?:\s+for)?\s+(\d+(?:\.\d+)?)\s*(second|sec|s|minute|min|m|hour|hr|h)\b",
                    msg,
                )
                seconds = 600.0
                if mute_match:
                    amount = float(mute_match.group(1))
                    unit = mute_match.group(2)
                    if unit.startswith("h"):
                        seconds = amount * 3600.0
                    elif unit.startswith("m"):
                        seconds = amount * 60.0
                    else:
                        seconds = amount
                seconds = min(max(seconds, 0.0), 12 * 3600.0)
                if seconds < 5:
                    reply = "Mute time is too short. Ask for at least 5 seconds, for example 'mute 10 min'."
                else:
                    speaker_alarm.mute_for(seconds)
                    handle_two_way_event("SIREN MUTED (AI CHAT)", f"Siren muted for {seconds / 60:.1f} min via Vision AI Chat.")
                    reply = (
                        f"On-site siren muted for {seconds / 60:.1f} min. Recording and Telegram emergency alerts "
                        "continue; only the local speaker is quiet."
                    )
                    action("Unmute Siren", command="unmute siren")
            elif has_word("mute"):
                speaker_alarm.mute_for(600)
                handle_two_way_event("SIREN MUTED (AI CHAT)", "Siren muted for 10 min via Vision AI Chat.")
                reply = "On-site siren muted for 10 min. Recording and Telegram emergency alerts continue."
                action("Unmute Siren", command="unmute siren")
            elif (
                has_word("silence", "quiet")
                or "stop alarm" in msg or "stop siren" in msg or "stop the siren" in msg
                or re.search(r"\b(?:siren|alarm)\b\s*(?:stop|off|silence|quiet)\b", msg)
            ):
                speaker_alarm.stop_alarm()
                handle_two_way_event("SIREN SILENCED (AI CHAT)", "User silenced alarm via Vision AI Chat.")
                reply = "On-site emergency siren has been silenced."
                action("Mute 10 Min", command="mute 10 min")
            elif has_word("siren", "alarm", "beep"):
                if speaker_alarm.is_muted():
                    reply = (
                        f"The siren is muted for another {speaker_alarm.mute_remaining() / 60:.1f} min. "
                        "Say 'unmute siren' first."
                    )
                    action("Unmute Siren", command="unmute siren")
                elif speaker_alarm.play_siren_alarm(duration=2.5):
                    handle_two_way_event("EMERGENCY SIREN (AI CHAT)", "User requested alarm via Vision AI Chat.")
                    reply = "Siren sounded through the on-site speaker (2.5 s test)."
                    action("Silence Siren", command="siren stop")
                else:
                    reply = "Siren did not start because the anti-loop cooldown is still active. It clears automatically."
            elif has_word("dispatch") or "send clip" in msg or "send the clip" in msg or "send last clip" in msg or "send incident" in msg or "send to telegram" in msg:
                started = GLOBAL_DELIVERY.dispatch(time.monotonic(), force=True)
                reply = "Pending violation clip batch started; check delivery status on the dashboard." if started else "No pending clips, or a batch is already sending. Telegram stays quiet between batches."
                add_clip_action()
            elif has_word("snapshot", "screenshot", "photo", "selfie") or "capture frame" in msg:
                snap_url, snap_message = save_live_snapshot()
                reply = snap_message
                if snap_url:
                    action("Open Snapshot", url=snap_url)
                    action("Take Another", command="snapshot")
            elif re.search(r"\b(?:confidence|threshold|sensitivity)\b[^\d]{0,20}\d", msg):
                conf_match = re.search(
                    r"\b(?:confidence|threshold|sensitivity)\b[^\d]{0,20}(\d{1,3}(?:\.\d+)?)\s*(%|percent)?",
                    msg,
                )
                value = float(conf_match.group(1)) if conf_match else 0.25
                if (conf_match and conf_match.group(2)) or value > 1.0:
                    value = value / 100.0
                if not 0.05 <= value <= 0.95:
                    reply = "Confidence must be between 5% and 95%, for example 'confidence 40' or 'confidence 0.4'."
                else:
                    GLOBAL_CAMERA.confidence = value
                    reply = (
                        f"Detection confidence set to {value * 100:.0f}%. It applies from the next frame and is kept "
                        "when the camera restarts."
                    )
            elif "start camera" in msg or "turn on camera" in msg:
                ok, cmsg = GLOBAL_CAMERA.start(GLOBAL_CAMERA.camera_index, GLOBAL_CAMERA.confidence)
                reply = f"Camera start requested: {cmsg}"
            elif "stop camera" in msg or "turn off camera" in msg:
                ok, cmsg = GLOBAL_CAMERA.stop()
                reply = f"Camera stop requested: {cmsg}"
            elif re.search(r"\b(?:clear|reset|remove|stop)\b.*\btarget\b", msg) or "stop searching" in msg:
                set_search_target_from_telegram("")
                reply = "HUD search target cleared. The live view shows all detections again."
            elif "latest clip" in msg or "last clip" in msg or "show clip" in msg or "play clip" in msg or "open clip" in msg:
                clip = latest_incident_clip()
                if clip:
                    reply = f"The newest clip is '{clip['title']}' ({clip['filename']}). It is opening in the player."
                    action("Open Latest Clip", clip=clip["filename"])
                else:
                    reply = "No incident clips have been recorded yet."
            elif (
                "last hour" in msg or "past hour" in msg or "today" in msg
                or has_word("incidents", "summary", "activity", "history", "log")
                or "what happened" in msg
            ):
                hours = 1.0
                day_match = re.search(r"(\d+)\s*(?:day|days|d)\b", msg)
                hour_match = re.search(r"(\d+)\s*(?:hour|hours|hr|hrs|h)\b", msg)
                if day_match:
                    hours = float(day_match.group(1)) * 24.0
                elif hour_match:
                    hours = float(hour_match.group(1))
                elif "today" in msg:
                    hours = 24.0
                reply = summarize_incident_log(hours)
                add_clip_action()
                action("Take Snapshot", command="snapshot")
            elif has_word("status", "health", "fps", "alert", "alerts", "violation", "violations", "wrong", "problem"):
                reply = camera_status_summary()
                action("Take Snapshot", command="snapshot")
                action("Last Hour", command="last hour")
                if GLOBAL_CAMERA.engine.pending_video_path is not None:
                    action("Send Clip Now", command="dispatch now")
            elif has_word("fall", "collapse"):
                reply = (
                    "Fall Watch uses pose estimation: when a person transitions from standing to horizontal for "
                    "0.8 seconds or more, the siren, spoken warning and Telegram clip fire automatically."
                )
                add_clip_action()
                action("Take Snapshot", command="snapshot")
            elif has_word("sleep", "sleeping"):
                reply = (
                    "Shift Vigor and Sleeping Watch flag prolonged slumping at a desk and lying on the floor for "
                    "more than 6 seconds, then raise a health alert."
                )
                add_clip_action()
            elif has_word("hand", "raise", "fight", "attack", "aggressive", "aggression"):
                reply = (
                    "Conduct Watch detects a hand raised above shoulder level toward another person, or a fast "
                    "arm extension toward them, and triggers the emergency siren and Telegram alert once per event."
                )
                add_clip_action()
            elif has_word("plate", "car", "bike", "vehicle"):
                reply = (
                    "Vehicle Analysis detects cars and motorcycles, reads number plates with OCR including the "
                    "timestamp, and estimates speed."
                )
                add_clip_action()
            elif (
                msg in {"help", "commands", "?", "what can you do", "what can you do?"}
                or has_word("hi", "hello", "hey")
            ):
                reply = (
                    "I can run commands and answer live questions. Try 'status', 'anything wrong?', 'last hour', "
                    "'dispatch now', 'snapshot', 'mute 10 min', 'unmute siren', 'test siren', 'confidence 40', "
                    "'target bottle', 'clear target', 'start camera'. You can also ask where a detected object is, "
                    "e.g. 'where is the phone?'."
                )
                action("Status", command="status")
                action("Last Hour", command="last hour")
                action("Take Snapshot", command="snapshot")
            else:
                target_match = re.search(
                    r"\b(?:target|track|find|locate|show(?: me)?)\s+(?:the\s+)?([a-z0-9 _-]{2,30})$", msg
                )
                target_clean = target_match.group(1).strip() if target_match else ""
                if target_clean in {"status", "clips", "incidents", "alerts", "camera", "help", "commands", "snapshot"}:
                    target_clean = ""
                if target_clean:
                    set_search_target_from_telegram(target_clean)
                    search_res = GLOBAL_OBJECT_TRACKER.search(target_clean)
                    if search_res:
                        top = search_res[0]
                        reply = (
                            f"HUD is now tracking '{target_clean}'. Last seen as "
                            f"{top.get('display_name', target_clean)} at {top.get('location_desc', 'monitored view')} "
                            f"({top.get('last_seen_time', 'recently')}, score {top.get('confidence', 'n/a')})."
                        )
                    else:
                        reply = (
                            f"HUD is now tracking '{target_clean}'. I have not seen it yet; the reticle will lock "
                            "when it appears."
                        )
                    action("Clear Target", command="clear target")
                else:
                    search_res = GLOBAL_OBJECT_TRACKER.search(msg)
                    if search_res:
                        top = search_res[0]
                        obj_name = top.get("display_name", top.get("label", "item"))
                        loc = top.get("location_desc", "monitored view")
                        seen = top.get("last_seen_time", "recently")
                        conf = top.get("confidence", "92%")
                        reply = f"Found {obj_name}! Location: {loc}. Last seen at: {seen}. Confidence score: {conf}."
                        action("Track It", command=f"target {obj_name}".lower())
                    else:
                        reply = (
                            f"I received your inquiry: '{msg}'. I can run commands like 'status', 'last hour', "
                            "'dispatch now', 'snapshot', 'mute 10 min', or search detected items such as phone, "
                            "bottle and keys."
                        )
                        action("Status", command="status")
                        action("Last Hour", command="last hour")
                        action("Take Snapshot", command="snapshot")

            self._send_json_response(200, {"ok": True, "reply": reply, "actions": actions})
            return

        # Emergency Siren trigger (Tornado air-raid wail)
        if path == "/api/speaker/siren":
            speaker_alarm.play_siren_alarm(duration=4.0)
            handle_two_way_event("EMERGENCY SIREN", "Emergency siren sounded via dashboard.")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": "Emergency siren activated."}).encode("utf-8"))
            return

        # Voice Announcement trigger
        if path == "/api/speaker/voice":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            msg_text = data.get("text", "Warning: Security violation detected.").strip()
            speaker_alarm.trigger_two_way_alert(msg_text)
            handle_two_way_event("VOICE INTERCOM", f"Spoken on speaker: \"{msg_text}\"")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": f"Broadcasted voice warning: \"{msg_text}\""}).encode("utf-8"))
            return

        # Emergency Siren Silence
        if path == "/api/speaker/stop":
            speaker_alarm.stop_alarm()
            handle_two_way_event("SIREN SILENCED", "Emergency siren silenced.")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": "Emergency siren silenced."}).encode("utf-8"))
            return

        # WhatsApp Direct Alert Live Test
        if path == "/api/whatsapp/test":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            wa_cfg = whatsapp_manager.load_whatsapp_config() or {}
            if data.get("phone"):
                wa_cfg["phone"] = data["phone"]
            if data.get("api_key"):
                wa_cfg["api_key"] = data["api_key"]
            if data.get("provider"):
                wa_cfg["provider"] = data["provider"]
            if data.get("account_sid"):
                wa_cfg["account_sid"] = data["account_sid"]
            if data.get("auth_token"):
                wa_cfg["auth_token"] = data["auth_token"]
            if data.get("from_whatsapp"):
                wa_cfg["from_whatsapp"] = data["from_whatsapp"]
            wa_cfg["enabled"] = True

            test_text = "VISIONAI - LIVE WHATSAPP ALERT TEST\nDirect security alert link verified. Events will be delivered here."
            ok, msg = whatsapp_manager.send_whatsapp_alert(wa_cfg, test_text)
            if ok:
                whatsapp_manager.save_whatsapp_config(wa_cfg)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}).encode("utf-8"))
            return

        # WhatsApp Save Configuration
        if path == "/api/whatsapp/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            wa_cfg = whatsapp_manager.load_whatsapp_config() or {}
            for key in ("provider", "phone", "api_key", "account_sid", "auth_token", "from_whatsapp"):
                if data.get(key):
                    wa_cfg[key] = data[key]
            if "enabled" in data:
                wa_cfg["enabled"] = bool(data["enabled"])
            whatsapp_manager.save_whatsapp_config(wa_cfg)
            self._send_json_response(200, {"ok": True, "message": "WhatsApp configuration saved."})
            return

        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args) -> None:
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a stale process keep shadowing the same
    # port, so only reuse addresses on POSIX where it is safe.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc_value = sys.exc_info()[1]
        if isinstance(exc_value, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def main() -> int:
    print("Initializing VisionAI Sentinel Command Center...", flush=True)
    global PORT
    local_ip = get_local_ip()
    httpd: ThreadedHTTPServer | None = None
    requested_port = PORT
    for candidate_port in range(requested_port, requested_port + 21):
        try:
            httpd = ThreadedHTTPServer(("0.0.0.0", candidate_port), CommandCenterHandler)
            PORT = candidate_port
            break
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, 10048) and getattr(exc, "winerror", None) != 10013:
                raise
            print(f"Port {candidate_port} is already in use; trying the next port.", flush=True)
    if httpd is None:
        raise OSError(f"No available port found between {requested_port} and {requested_port + 20}.")

    if GLOBAL_CAMERA.listener is not None:
        GLOBAL_CAMERA.listener.port = PORT

    print("=" * 65, flush=True)
    print("   SERVER READY - VISIONAI SENTINEL COMMAND CENTER", flush=True)
    print("=" * 65, flush=True)
    print(f"   Local Access:    http://127.0.0.1:{PORT}", flush=True)
    print(f"   Mobile/Network:  http://{local_ip}:{PORT}", flush=True)
    print("   Dashboard is now LIVE and ready!", flush=True)
    print("=" * 65, flush=True)

    if not os.environ.get("VISIONAI_NO_BROWSER"):
        def _open_dashboard() -> None:
            try:
                time.sleep(0.5)
                import webbrowser
                webbrowser.open(f"http://127.0.0.1:{PORT}")
            except Exception:
                pass

        threading.Thread(target=_open_dashboard, daemon=True, name="dashboard-opener").start()

    threading.Thread(target=GLOBAL_CAMERA.preload_model, daemon=True, name="model-preloader").start()
    GLOBAL_DELIVERY.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Command Center Server...", flush=True)
    finally:
        GLOBAL_DELIVERY.stop()
        httpd.server_close()
        GLOBAL_CAMERA.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
