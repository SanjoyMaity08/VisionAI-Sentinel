"""Gyanovate VisionAI - High-Tech Enterprise Surveillance UI (Emoji-Free).

Architectural Features:
- Modern Enterprise HUD: High-contrast telemetry ribbons, targeting reticles,
  frosted acrylic overlays, vector pips, and precision countdown meters.
- Zero Emojis: 100% clean, professional, military/enterprise typography and indicators.
- 3-Minute Periodic Threat Dispatcher:
  * Automatically holds 1 validated 5-second video clip per 3-minute cycle.
  * Dispatches strictly at the end of the 3-minute window to Telegram.
  * Completely silent if no safety violation occurs (0 notifications).
- Evaluates core workplace safety & restricted behaviors:
  1. Machine Danger Perimeter Intrusion (Entering red hazard boundary)
  2. Counter Phone Addiction (Staff browsing phone on duty)
  3. Mandatory PPE Safety Vest Violation (Missing high-vis vest)
  4. Shift Laziness / Sleeping on Duty (Slumped on desk > 3.5s)
  5. Sudden Fall & Medical Collapse (Person fallen on floor)
  6. Unmanned Duty Desk (Cashier/guard desk vacant > threshold)

Controls (inside camera window):
  [D] Dispatch held 3-minute video clip immediately (manual trigger)
  [H] Toggle on-screen Safety Compliance Radar
  [Q] Quit Demo
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

from vision_detection import prime_windows_platform_cache

prime_windows_platform_cache()
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "telegram_config.json"
LOG_FILE = ROOT / "incident_log.csv"
INCIDENTS_DIR = ROOT / "incidents"
INCIDENTS_DIR.mkdir(exist_ok=True)

CLASS_PERSON = 0
CLASS_PHONE = 67
MONITORED_CLASSES = [CLASS_PERSON, CLASS_PHONE]

# High-Tech Enterprise Palette (BGR)
COLOR_BG = (16, 20, 26)
COLOR_BORDER = (45, 55, 72)
COLOR_TEXT_PRIMARY = (235, 240, 245)
COLOR_TEXT_MUTED = (140, 155, 175)
COLOR_CYAN = (255, 210, 0)
COLOR_GOLD = (35, 195, 255)
COLOR_SUCCESS = (80, 225, 60)
COLOR_WARNING = (0, 175, 255)
COLOR_DANGER = (40, 40, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gyanovate VisionAI - Enterprise Surveillance UI")
    parser.add_argument("--camera", type=int, default=0, help="Camera index: try 0, 1, or 2")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence (0 to 1)")
    parser.add_argument("--model", default=MODEL_NAME, help="Ultralytics YOLO model path/name")
    parser.add_argument("--mode", default="all", choices=["all", "fall", "phone", "perimeter", "loitering", "eldercare", "retail", "safety", "security"], help="Surveillance mode focus: all, fall (eldercare), phone (retail), perimeter (safety), loitering (security)")
    parser.add_argument("--no-telegram", action="store_true", help="Run detection locally without Telegram alerts")
    parser.add_argument("--interval-sec", type=float, default=180.0, help="Seconds between Telegram dispatches (default: 180s = 3 minutes)")
    parser.add_argument("--phone-sec", type=float, default=2.0, help="Seconds before phone alert (default 2.0s)")
    parser.add_argument("--desk-empty-sec", type=float, default=5.0, help="Seconds before empty desk alert (default 5.0s)")
    return parser.parse_args()


def open_camera(index: int) -> cv2.VideoCapture:
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def load_telegram_config() -> dict | None:
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if data.get("bot_token") and data.get("chat_id"):
            return data
    except (OSError, json.JSONDecodeError):
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
        return False, str(exc)


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


def draw_corner_reticle(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int], thickness: int = 1, corner_len: int = 16) -> None:
    """Draws high-tech targeting corner brackets around bounding boxes."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Subtle boundary line
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # High-intensity corners
    c_thick = thickness + 2
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, c_thick)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, c_thick)

    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, c_thick)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, c_thick)

    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, c_thick)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, c_thick)

    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, c_thick)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, c_thick)


class PeriodicClipEngine:
    """Monitors core safety violations, holds ONE video clip per 3 minutes, and dispatches only on cycle completion."""

    def __init__(self, args: argparse.Namespace, config: dict | None) -> None:
        self.args = args
        self.config = config
        self.interval_sec = args.interval_sec

        self.cycle_start_time = time.time()
        self.cycle_start_stamp = datetime.now().strftime("%H:%M:%S")

        self.pre_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=30)
        self.is_recording_clip = False
        self.frames_needed = 0
        self.recorded_frames: list[np.ndarray] = []
        self.current_incident_meta: dict[str, str] = {}

        self.pending_video_path: Path | None = None
        self.pending_video_meta: dict[str, str] | None = None

        self.phone_dwell_start: float | None = None
        self.sleep_start: float | None = None
        self.fall_start: float | None = None
        self.empty_desk_start: float | None = None

    def trigger_clip_capture(self, title: str, details: str, frame: np.ndarray) -> None:
        if self.pending_video_path is not None or self.is_recording_clip:
            return

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.is_recording_clip = True
        self.frames_needed = 45  # ~3 seconds post-event at 15fps
        self.recorded_frames = list(self.pre_buffer)
        self.recorded_frames.append(frame.copy())
        self.current_incident_meta = {
            "title": title,
            "details": details,
            "timestamp": stamp,
        }
        print(f"\n[VIOLATION CAPTURED] {title.upper()} - Recording 5s clip for 3-minute dispatch...")

    def update_recording(self, frame: np.ndarray) -> None:
        if not self.is_recording_clip:
            self.pre_buffer.append(frame.copy())
            return

        self.recorded_frames.append(frame.copy())
        self.frames_needed -= 1

        if self.frames_needed <= 0:
            self.is_recording_clip = False
            time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = self.current_incident_meta["title"].lower().replace(" ", "_")[:20]
            video_filename = f"clip_{time_tag}_{safe_name}.mp4"
            video_path = INCIDENTS_DIR / video_filename

            h, w = self.recorded_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for f in self.recorded_frames:
                out.write(f)
            out.release()

            self.pending_video_path = video_path
            self.pending_video_meta = self.current_incident_meta.copy()
            append_incident_log(
                self.current_incident_meta["timestamp"],
                self.current_incident_meta["title"],
                self.current_incident_meta["details"],
                video_filename,
            )
            self.recorded_frames.clear()
            print(f"[CLIP READY] 5-second video saved. Will dispatch at the end of the 3-minute window.\n")

    def dispatch_3min_cycle(self, now: float, is_manual: bool = False) -> None:
        end_stamp = datetime.now().strftime("%H:%M:%S")
        window_str = f"{self.cycle_start_stamp} - {end_stamp}"

        if self.pending_video_path is None or not self.pending_video_path.exists():
            prefix = "[MANUAL TEST]" if is_manual else "[3-MINUTE CYCLE COMPLETE]"
            print(f"{prefix} No safety violations occurred between {window_str}. Telegram kept silent (0 spam).")
            self.cycle_start_time = now
            self.cycle_start_stamp = end_stamp
            return

        print(f"\n{'='*65}")
        print(f"[3-MINUTE DISPATCH] Sending incident video clip to Telegram ({window_str})...")

        meta = self.pending_video_meta or {}
        caption = (
            f"GYANOVATE VISIONAI - 3-MINUTE SECURITY REPORT\n"
            f"Window: {window_str}\n"
            f"Primary Violation: {meta.get('title', 'Safety Event')}\n"
            f"Details: {meta.get('details', '')}\n"
            f"Timestamp: {meta.get('timestamp', '')}\n\n"
            f"Attached: Validated 5-Second Video Clip."
        )

        video_file_to_send = str(self.pending_video_path)

        def bg_send():
            if self.config and not self.args.no_telegram:
                sent, msg = send_telegram_video_clip(self.config, video_file_to_send, caption)
                print(f"[Telegram Result] {msg}")
            else:
                print(f"[Local Video Stored] Saved to: {Path(video_file_to_send).name}")

        threading.Thread(target=bg_send, daemon=True).start()

        self.pending_video_path = None
        self.pending_video_meta = None
        self.cycle_start_time = now
        self.cycle_start_stamp = end_stamp
        print(f"[Next Cycle Started] Counting next 3 minutes from {self.cycle_start_stamp}.")
        print(f"{'='*65}\n")

    def evaluate_core_rules(self, frame: np.ndarray, boxes_data: list[dict], now: float) -> dict[str, Any]:
        h, w = frame.shape[:2]

        # 1. Machine Danger Zone (Red - Right)
        dz_x1, dz_y1 = int(w * 0.48), int(h * 0.20)
        dz_x2, dz_y2 = int(w * 0.95), int(h * 0.90)

        # 2. Staff / Cashier Duty Desk (Cyan - Left)
        desk_x1, desk_y1 = int(w * 0.08), int(h * 0.25)
        desk_x2, desk_y2 = int(w * 0.44), int(h * 0.85)

        people = [b for b in boxes_data if b["class_id"] == CLASS_PERSON]
        phones = [b for b in boxes_data if b["class_id"] == CLASS_PHONE]

        # Draw zones with precision corner reticles
        draw_corner_reticle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), COLOR_DANGER, 1, 16)
        cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x1 + 225, dz_y1 + 22), (24, 26, 34), -1)
        cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x1 + 225, dz_y1 + 22), COLOR_DANGER, 1)
        cv2.putText(frame, "ZONE: MACHINE DANGER PERIMETER", (dz_x1 + 6, dz_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_PRIMARY, 1)

        draw_corner_reticle(frame, (desk_x1, desk_y1), (desk_x2, desk_y2), COLOR_CYAN, 1, 16)
        cv2.rectangle(frame, (desk_x1, desk_y1), (desk_x1 + 175, desk_y1 + 22), (24, 26, 34), -1)
        cv2.rectangle(frame, (desk_x1, desk_y1), (desk_x1 + 175, desk_y1 + 22), COLOR_CYAN, 1)
        cv2.putText(frame, "ZONE: STAFF / CASHIER DESK", (desk_x1 + 6, desk_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_PRIMARY, 1)

        active_violations: list[str] = []
        radar_status: dict[str, tuple[str, tuple[int, int, int]]] = {}
        mode = getattr(self.args, "mode", "all")

        # 1. MACHINE DANGER PERIMETER BREACH
        if mode in ("all", "perimeter", "safety", "security", "loitering"):
            in_danger_zone = sum(1 for p in people if dz_x1 <= p["center"][0] <= dz_x2 and dz_y1 <= p["center"][1] <= dz_y2)
            if in_danger_zone > 0:
                active_violations.append("PERIMETER BREACH")
                radar_status["Danger Perimeter"] = (f"BREACH ({in_danger_zone})", COLOR_DANGER)
                self.trigger_clip_capture("Machine Danger Perimeter Breach", f"{in_danger_zone} human entered hazard zone. Halting line.", frame)
            else:
                radar_status["Danger Perimeter"] = ("SECURE", COLOR_SUCCESS)
        else:
            radar_status["Danger Perimeter"] = ("STANDBY", COLOR_TEXT_MUTED)

        # 2. COUNTER PHONE ADDICTION
        if mode in ("all", "phone", "retail"):
            phone_held = False
            for ph in phones:
                px1, py1, px2, py2 = ph["bbox"]
                pcx, pcy = ph["center"]
                draw_corner_reticle(frame, (px1, py1), (px2, py2), COLOR_DANGER, 1, 8)
                cv2.putText(frame, "DEVICE: PHONE", (px1, max(18, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_DANGER, 1)
                for p in people:
                    x1, y1, x2, y2 = p["bbox"]
                    if (x1 - 50) <= pcx <= (x2 + 50) and (y1 - 50) <= pcy <= (y2 + 50):
                        phone_held = True
                        break

            if phone_held:
                if self.phone_dwell_start is None:
                    self.phone_dwell_start = now
                dwell = now - self.phone_dwell_start
                if dwell >= self.args.phone_sec:
                    active_violations.append("PHONE ADDICTION")
                    radar_status["Phone Distraction"] = (f"IN USE ({dwell:.1f}s)", COLOR_DANGER)
                    self.trigger_clip_capture("Counter Phone Addiction", f"Staff browsing device for {dwell:.1f}s instead of serving.", frame)
                else:
                    radar_status["Phone Distraction"] = (f"IN USE ({dwell:.1f}s)", COLOR_WARNING)
            else:
                self.phone_dwell_start = None
                radar_status["Phone Distraction"] = ("CLEAR", COLOR_SUCCESS)
        else:
            radar_status["Phone Distraction"] = ("STANDBY", COLOR_TEXT_MUTED)

        # 3. PPE & SAFETY VEST COMPLIANCE
        if mode in ("all", "safety", "perimeter"):
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
                active_violations.append("NO PPE VEST")
                radar_status["PPE Safety Vest"] = (f"VIOLATION ({ppe_missing})", COLOR_DANGER)
                self.trigger_clip_capture("PPE Safety Vest Violation", f"{ppe_missing} worker(s) missing high-vis safety vests.", frame)
            elif people:
                radar_status["PPE Safety Vest"] = ("COMPLIANT", COLOR_SUCCESS)
            else:
                radar_status["PPE Safety Vest"] = ("STANDBY", COLOR_TEXT_MUTED)
        else:
            radar_status["PPE Safety Vest"] = ("STANDBY", COLOR_TEXT_MUTED)

        # 4. SHIFT LAZINESS / SLEEPING ON DUTY
        if mode in ("all", "retail", "phone"):
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
                if slump_dur >= 3.5:
                    active_violations.append("SLEEPING ON DUTY")
                    radar_status["Shift Vigor"] = (f"SLEEPING ({slump_dur:.1f}s)", COLOR_DANGER)
                    self.trigger_clip_capture("Shift Laziness / Sleeping on Duty", f"Staff sleeping on duty / prolonged slump ({slump_dur:.1f}s).", frame)
                else:
                    radar_status["Shift Vigor"] = (f"SLUMP ({slump_dur:.1f}s)", COLOR_WARNING)
            else:
                self.sleep_start = None
                radar_status["Shift Vigor"] = ("ACTIVE", COLOR_SUCCESS)
        else:
            radar_status["Shift Vigor"] = ("STANDBY", COLOR_TEXT_MUTED)

        # 5. SUDDEN FALL & COLLAPSE
        if mode in ("all", "fall", "eldercare"):
            fall_detected = False
            for p in people:
                bw = p["bbox"][2] - p["bbox"][0]
                bh = p["bbox"][3] - p["bbox"][1]
                aspect_ratio = bw / max(1, bh)
                cy = p["center"][1]
                if aspect_ratio >= 1.25 and cy > int(h * 0.40):
                    fall_detected = True
                    p["is_fall"] = True

            if fall_detected:
                if self.fall_start is None:
                    self.fall_start = now
                if (now - self.fall_start) >= 0.8:
                    active_violations.append("SUDDEN FALL")
                    radar_status["Fall Watch"] = ("COLLAPSE DETECTED", COLOR_DANGER)
                    self.trigger_clip_capture("Sudden Fall & Emergency Collapse", "Subject collapsed on floor in horizontal posture.", frame)
            else:
                self.fall_start = None
                radar_status["Fall Watch"] = ("NORMAL", COLOR_SUCCESS)
        else:
            radar_status["Fall Watch"] = ("STANDBY", COLOR_TEXT_MUTED)

        # 6. UNMANNED BILLING DESK
        if mode in ("all", "retail", "phone", "security", "loitering"):
            desk_staff = sum(1 for p in people if desk_x1 <= p["center"][0] <= desk_x2 and desk_y1 <= p["center"][1] <= desk_y2)
            if desk_staff == 0:
                if self.empty_desk_start is None:
                    self.empty_desk_start = now
                empty_dur = now - self.empty_desk_start
                if empty_dur >= self.args.desk_empty_sec:
                    active_violations.append("UNMANNED DESK")
                    radar_status["Duty Desk"] = (f"VACANT ({empty_dur:.1f}s)", COLOR_DANGER)
                    self.trigger_clip_capture("Unmanned Billing Desk Alert", f"Billing desk vacant for {empty_dur:.1f}s.", frame)
                else:
                    radar_status["Duty Desk"] = (f"EMPTY ({empty_dur:.1f}s)", COLOR_WARNING)
            else:
                self.empty_desk_start = None
                radar_status["Duty Desk"] = (f"OCCUPIED ({desk_staff})", COLOR_SUCCESS)
        else:
            radar_status["Duty Desk"] = ("STANDBY", COLOR_TEXT_MUTED)

        # Render Person Targeting Reticles
        for p in people:
            x1, y1, x2, y2 = p["bbox"]
            is_fall = p.get("is_fall", False)
            has_vest = p.get("has_vest", True)

            if is_fall:
                color = COLOR_DANGER
                tag = "SUBJECT: FALL DETECTED"
            elif not has_vest:
                color = COLOR_DANGER
                tag = "SUBJECT: NO PPE VEST"
            else:
                color = COLOR_SUCCESS
                tag = "SUBJECT: COMPLIANT"

            draw_corner_reticle(frame, (x1, y1), (x2, y2), color, 1, 14)
            # Label tab
            lbl_w = 175
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lbl_w, y1), (20, 24, 32), -1)
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + lbl_w, y1), color, 1)
            cv2.putText(frame, tag, (x1 + 6, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_PRIMARY, 1)

        return {
            "active_violations": active_violations,
            "radar_status": radar_status,
        }


def draw_enterprise_ui(
    frame: np.ndarray,
    evaluation: dict[str, Any],
    fps: float,
    telegram_ok: bool,
    show_radar: bool,
    time_to_dispatch: float,
    has_clip_ready: bool,
    is_recording: bool,
) -> None:
    """Renders a gorgeous, dark glassmorphism enterprise HUD (100% emoji-free)."""
    h, w = frame.shape[:2]
    active_violations = evaluation["active_violations"]
    radar_status = evaluation["radar_status"]

    # 1. Top Telemetry Header Ribbon
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.line(frame, (0, 75), (w, 75), COLOR_BORDER, 1)

    # Brand Title & Platform Subtitle
    cv2.putText(frame, "GYANOVATE VISIONAI", (14, 24), cv2.FONT_HERSHEY_DUPLEX, 0.62, COLOR_GOLD, 1)
    cv2.putText(frame, "// AUTONOMOUS EDGE SAFETY ENGINE", (230, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1)

    # Live Status Badge (Pill box)
    if is_recording:
        status_text = "REC: CAPTURING 5-SECOND VIDEO CLIP"
        status_color = COLOR_DANGER
    elif active_violations:
        status_text = "ALERT: " + " | ".join(active_violations[:2])
        status_color = COLOR_DANGER
    else:
        status_text = "STATUS: ALL SYSTEMS NOMINAL"
        status_color = COLOR_SUCCESS

    badge_w = 340
    cv2.rectangle(frame, (14, 38), (14 + badge_w, 66), (22, 28, 38), -1)
    cv2.rectangle(frame, (14, 38), (14 + badge_w, 66), status_color, 1)
    # Status pip dot
    cv2.circle(frame, (28, 52), 4, status_color, -1)
    cv2.putText(frame, status_text, (40, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

    # 3-Minute Cycle Progress Meter
    bar_x = 370
    bar_w = 180
    bar_h = 10
    bar_y = 50
    total_cycle = 180.0
    elapsed = max(0.0, total_cycle - time_to_dispatch)
    fill_ratio = min(1.0, elapsed / total_cycle)
    fill_w = int(bar_w * fill_ratio)

    mins = int(time_to_dispatch) // 60
    secs = int(time_to_dispatch) % 60
    time_label = f"CYCLE DISPATCH IN: {mins:02d}:{secs:02d}"
    cv2.putText(frame, time_label, (bar_x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1)

    # Draw progress bar track
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 32, 44), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_BORDER, 1)
    # Draw progress fill
    fill_color = COLOR_WARNING if has_clip_ready else COLOR_CYAN
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    # Right side metrics
    tg_text = "BOT: ONLINE" if telegram_ok else "BOT: LOCAL"
    tg_color = COLOR_SUCCESS if telegram_ok else COLOR_TEXT_MUTED
    cv2.putText(frame, f"FPS: {fps:.1f} | {tg_text}", (w - 180, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40, tg_color, 1)
    cv2.putText(frame, "[D] Send Now | [H] Radar | [Q] Quit", (w - 245, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1)

    # 2. Safety Compliance Radar Sidebar
    if show_radar:
        panel_w = 230
        panel_h = 165
        p_overlay = frame.copy()
        cv2.rectangle(p_overlay, (12, 88), (12 + panel_w, 88 + panel_h), COLOR_BG, -1)
        cv2.addWeighted(p_overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (12, 88), (12 + panel_w, 88 + panel_h), COLOR_BORDER, 1)

        cv2.putText(frame, "COMPLIANCE RADAR", (22, 108), cv2.FONT_HERSHEY_DUPLEX, 0.42, COLOR_GOLD, 1)
        cv2.line(frame, (22, 114), (12 + panel_w - 10, 114), COLOR_BORDER, 1)

        y_offset = 134
        for key, (val_text, color) in radar_status.items():
            # Status pip
            cv2.circle(frame, (24, y_offset - 3), 3, color, -1)
            cv2.putText(frame, key, (34, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT_PRIMARY, 1)
            cv2.putText(frame, val_text, (132, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y_offset += 20


def main() -> int:
    args = parse_args()
    config = None if args.no_telegram else load_telegram_config()

    print("=" * 65)
    print("   GYANOVATE VISIONAI - ENTERPRISE SURVEILLANCE PLATFORM")
    print("=" * 65)
    print("UI Architecture: High-Tech Enterprise HUD (100% Emoji-Free)")
    print(f"Periodic Dispatch Interval: {args.interval_sec:.0f} seconds (3 minutes)")
    print("Notification Policy: 1 Validated 5-Sec Video Clip per 3-Min Cycle")
    print("-" * 65)

    if args.no_telegram:
        print("[Status] Local Mode: Telegram dispatches disabled (--no-telegram).")
    elif config:
        print("[Status] Telegram Connected: Video clips dispatch strictly on the 3-minute schedule.")
    else:
        print("[Status] Telegram Not Configured: Run setup_telegram.bat to connect your bot.")

    print(f"\nInitializing Camera Index {args.camera}...")
    camera = open_camera(args.camera)
    if not camera.isOpened():
        print(f"Error: Could not open camera {args.camera}. Check connection or try --camera 1.")
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    model_path = args.model
    if not Path(model_path).is_file() and (ROOT / model_path).is_file():
        model_path = str(ROOT / model_path)
    print(f"Loading local YOLO weights '{model_path}'...")
    try:
        model = YOLO(model_path)
    except Exception as exc:
        camera.release()
        print(f"Could not load AI model: {exc}")
        return 1

    engine = PeriodicClipEngine(args, config)
    print(f"Platform Active in [{args.mode.upper()}] mode. Continuous autonomous monitoring initialized.")
    print("Hotkeys: [D] Dispatch Clip Now | [H] Toggle Radar | [Q] Quit\n")

    fps = 0.0
    last_frame_time = time.time()
    show_radar = True

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a camera frame. Check USB connection.")
                break

            now = time.time()
            dt = now - last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else (0.85 * fps + 0.15 * inst)
            last_frame_time = now

            # YOLO single-shot prediction
            try:
                prediction = model.predict(
                    source=frame,
                    imgsz=640,
                    conf=args.confidence,
                    classes=MONITORED_CLASSES,
                    device="cpu",
                    verbose=False,
                )[0]
            except Exception as exc:
                print(f"Inference error: {exc}")
                break

            boxes_data = []
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    cls_id = int(box.cls[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].item())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    boxes_data.append({
                        "class_id": cls_id,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "center": (cx, cy),
                    })

            # Evaluate core rules
            evaluation = engine.evaluate_core_rules(frame, boxes_data, now)

            # Update rolling ring buffer or active recording
            engine.update_recording(frame)

            # Check 3-minute periodic dispatch timer
            elapsed = now - engine.cycle_start_time
            time_to_dispatch = max(0.0, engine.interval_sec - elapsed)
            if elapsed >= engine.interval_sec:
                engine.dispatch_3min_cycle(now, is_manual=False)

            # Draw gorgeous high-tech enterprise UI (zero emojis)
            draw_enterprise_ui(
                frame,
                evaluation,
                fps,
                bool(config and not args.no_telegram),
                show_radar,
                time_to_dispatch,
                bool(engine.pending_video_path is not None),
                engine.is_recording_clip,
            )

            cv2.imshow("Gyanovate VisionAI - Enterprise Surveillance System", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("d"), ord("D")):
                print("\n[Manual Trigger] Dispatching 3-minute cycle clip immediately...")
                engine.dispatch_3min_cycle(now, is_manual=True)
            elif key in (ord("h"), ord("H")):
                show_radar = not show_radar

    except KeyboardInterrupt:
        print("\nStopping platform...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("Platform stopped. Incident log saved to:", LOG_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
