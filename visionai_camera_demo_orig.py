"""Gyanovate VisionAI - USB camera + AI + Telegram text alert demo.

Default rule: if a YOLO-detected person enters the red rectangle (demo restricted
zone), send one Telegram text alert. Re-arm after the zone has been clear for
3 seconds. Camera frames are processed locally and are not uploaded or saved.

Run: python visionai_camera_demo.py --camera 0
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests

from vision_detection import prime_windows_platform_cache

prime_windows_platform_cache()
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "telegram_config.json"
LOG_FILE = ROOT / "incident_log.csv"
CLEAR_SECONDS_TO_REARM = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB camera AI person detection with Telegram alerts")
    parser.add_argument("--camera", type=int, default=0, help="Camera index: try 0, then 1 or 2")
    parser.add_argument("--confidence", type=float, default=0.40, help="Detection confidence from 0 to 1")
    parser.add_argument("--model", default=MODEL_NAME, help="Ultralytics model weights name/path")
    parser.add_argument("--no-telegram", action="store_true", help="Run detection without phone notifications")
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


def send_telegram_alert(config: dict, text: str) -> tuple[bool, str]:
    url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": config["chat_id"], "text": text}, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if payload.get("ok"):
            return True, "Telegram message sent"
        return False, payload.get("description", "Telegram API error")
    except requests.RequestException as exc:
        return False, str(exc)


def append_log(timestamp: str, persons: int) -> None:
    exists = LOG_FILE.exists()
    try:
        with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp", "event", "persons_in_restricted_zone"])
            writer.writerow([timestamp, "PERSON_IN_RESTRICTED_ZONE", persons])
    except OSError as exc:
        print(f"Could not write incident log: {exc}")


def main() -> int:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        print("Error: confidence must be between 0 and 1")
        return 2

    config = None if args.no_telegram else load_telegram_config()
    if args.no_telegram:
        print("Phone alerts disabled (--no-telegram).")
    elif config:
        print("Telegram configuration found. Phone alerts are enabled.")
    else:
        print("Telegram is not configured. AI detection will run, but no phone alert can be sent.")
        print("Run setup_telegram.bat once, then start this demo again.")

    print(f"\nOpening USB camera index {args.camera}...")
    camera = open_camera(args.camera)
    if not camera.isOpened():
        print("Could not open camera. Try --camera 1 or --camera 2, close Camera app, and check Windows camera permissions.")
        camera.release()
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Loading AI model {args.model}; first run may download model weights...")
    try:
        model = YOLO(args.model)
    except Exception as exc:
        camera.release()
        print(f"Could not load AI model: {exc}")
        return 1

    print("AI camera running. Put a person inside the red zone to test the alert.")
    print("Press Q in the video window to quit. The video is processed locally.")

    previous_alert = False
    clear_since = None
    fps = 0.0
    last_frame_time = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a camera frame. Check the USB connection.")
                break
            height, width = frame.shape[:2]
            zx1, zy1 = int(width * 0.35), int(height * 0.20)
            zx2, zy2 = int(width * 0.65), int(height * 0.90)

            try:
                prediction = model.predict(source=frame, imgsz=640, conf=args.confidence,
                                            classes=[0], device="cpu", verbose=False)[0]
            except Exception as exc:
                print(f"AI inference error: {exc}")
                break

            person_count = 0
            in_zone = 0
            if prediction.boxes is not None:
                for box in prediction.boxes:
                    if int(box.cls[0].item()) != 0:  # COCO class 0 = person
                        continue
                    person_count += 1
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0].item())
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    inside = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                    if inside:
                        in_zone += 1
                    color = (0, 0, 255) if inside else (0, 200, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"PERSON {confidence:.2f}", (x1, max(22, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (zx1, max(22, zy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            now = time.time()
            dt = now - last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else 0.85 * fps + 0.15 * inst
            last_frame_time = now

            status = "NORMAL: AREA CLEAR"
            status_color = (0, 180, 0)
            if in_zone > 0:
                clear_since = None
                status = "ALERT: PERSON IN RESTRICTED ZONE"
                status_color = (0, 0, 255)
                if not previous_alert:
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_log(stamp, in_zone)
                    alert_text = (
                        "🚨 GYANOVATE VISIONAI ALERT\n"
                        "Event: Person detected in the restricted zone.\n"
                        f"Camera: USB Camera {args.camera}\n"
                        f"Time: {stamp}\n"
                        f"People detected in zone: {in_zone}"
                    )
                    if config:
                        sent, detail = send_telegram_alert(config, alert_text)
                        print(f"[{stamp}] {detail}")
                        if not sent:
                            print(f"Telegram send failed: {detail}")
                    else:
                        print(f"[{stamp}] ALERT triggered locally; phone notification not configured.")
                previous_alert = True
            else:
                if previous_alert:
                    if clear_since is None:
                        clear_since = now
                    elif now - clear_since >= CLEAR_SECONDS_TO_REARM:
                        previous_alert = False
                        clear_since = None
                        print("INFO: Zone clear; alert is re-armed for the next entry.")

            cv2.rectangle(frame, (0, 0), (width, 82), (25, 25, 25), -1)
            cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
            phone_state = "Telegram: ON" if config else "Telegram: NOT CONFIGURED"
            cv2.putText(frame, f"People: {person_count} | In zone: {in_zone} | FPS: {fps:.1f} | {phone_state}",
                        (12, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Gyanovate VisionAI - USB Camera (Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        print("Stopping demo...")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print("Demo stopped. Incident log:", LOG_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
