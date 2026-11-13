"""VisionAI Sentinel - Two-Way Telegram AI Assistant & Listener.

Provides bidirectional AI communication between the user's phone and the VisionAI Command Center:
- AI Object Search: Ask "where is the bottle?", "find pen", "camera", "bottel", etc.
  -> Replies with real-time location, quadrant coordinates, and confidence.
  -> Sends high-resolution cropped photo snapshot directly into Telegram chat.
  -> Sends the 10-second MP4 video clip directly into Telegram chat so it can be played immediately on mobile!
  -> Automatically syncs with the web dashboard to lock AI reticles onto the searched target in real-time.
- Emergency Siren Control: Type "alarm" or /alarm to sound the high-decibel Tornado air-raid wail siren on site.
- Silence Siren: Type "stop" or /stop to silence the alarm.
- Two-Way Intercom: Type /speak <message> to announce warnings over the site speaker.
- Real-Time Telemetry: Type "status" or "link" for live camera health, FPS, and web command center URL.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import re
import socket
import threading
import time
from datetime import datetime
from typing import Any, Callable

import requests

import speaker_alarm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
INCIDENTS_DIR = ROOT / "incidents"
SNAPSHOT_DIR = INCIDENTS_DIR / "snapshots"


def get_local_ip() -> str:
    """Detect LAN IP address so links sent to Telegram can be accessed on phone/Wi-Fi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class TelegramTwoWayListener:
    """Polls Telegram getUpdates in background and executes two-way AI queries and actions."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        on_event_callback: Callable[[str, str], None] | None = None,
        object_tracker: Any | None = None,
        status_callback: Callable[[], dict[str, Any]] | None = None,
        set_target_callback: Callable[[str], None] | None = None,
        camera_control_callback: Callable[[str, int | None], tuple[bool, str]] | None = None,
        port: int = 8080,
    ) -> None:
        self.config = config
        self.on_event = on_event_callback
        self.tracker = object_tracker
        self.status_cb = status_callback
        self.set_target_cb = set_target_callback
        self.camera_control_cb = camera_control_callback
        self.port = port
        self.is_running = False
        self.thread: threading.Thread | None = None
        self.last_update_id = 0
        self.last_object_query: dict[str, str] = {}
        self._pending_clip_deliveries: set[tuple[str, str]] = set()
        self._pending_clip_deliveries_lock = threading.Lock()

    def start(self) -> None:
        if not self.config or not self.config.get("bot_token"):
            return
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print("[Telegram Two-Way AI] Listener active. Ready for bidirectional AI queries and commands.")

    def stop(self) -> None:
        self.is_running = False

    def _reply_telegram(self, text: str) -> None:
        """Backward-compatible shortcut for sending text message."""
        if not self.config:
            return
        chat_id = self.config.get("chat_id")
        if chat_id:
            self._send_text(chat_id, text)

    def send_automatic_alert(self, text: str) -> bool:
        """Send an automated detection alert to the configured Telegram chat."""
        chat_id = str((self.config or {}).get("chat_id", "")).strip()
        if not chat_id or not (self.config or {}).get("bot_token"):
            return False
        return self._send_text(chat_id, text)

    def send_violation_clip(self, video_path: Path | str, caption: str = "") -> bool:
        """Send an annotated 10-second violation clip to the configured Telegram chat."""
        chat_id = str((self.config or {}).get("chat_id", "")).strip()
        if not chat_id or not (self.config or {}).get("bot_token"):
            return False
        return self._send_video(chat_id, video_path, caption)

    def _send_text(self, chat_id: str | int, text: str) -> bool:
        """Send formatted text message to Telegram chat."""
        if not self.config:
            return False
        token = self.config.get("bot_token")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "text": text},
                timeout=12,
            )
            return r.status_code == 200
        except Exception as exc:
            logger.debug("[Telegram] sendMessage exception: %s", type(exc).__name__)
            return False

    def _send_photo(self, chat_id: str | int, photo_path: Path | str, caption: str = "") -> bool:
        """Send snapshot photo image to Telegram chat."""
        if not self.config:
            return False
        p = Path(photo_path)
        if not p.is_file():
            return False
        token = self.config.get("bot_token")
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with p.open("rb") as f:
                r = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption[:1024]},
                    files={"photo": (p.name, f, "image/jpeg")},
                    timeout=20,
                )
            return r.status_code == 200
        except Exception as exc:
            logger.debug("[Telegram] sendPhoto exception: %s", type(exc).__name__)
            return False

    def _send_video(self, chat_id: str | int, video_path: Path | str, caption: str = "") -> bool:
        """Send 10-second MP4 video clip directly into Telegram chat for inline mobile playback."""
        if not self.config:
            return False
        p = Path(video_path)
        if not p.is_file():
            return False
        token = self.config.get("bot_token")
        url = f"https://api.telegram.org/bot{token}/sendVideo"
        try:
            with p.open("rb") as f:
                r = requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "caption": caption[:1024],
                        "supports_streaming": "true",
                    },
                    files={"video": (p.name, f, "video/mp4")},
                    timeout=45,
                )
            return r.status_code == 200 and r.json().get("ok") is True
        except Exception as exc:
            logger.debug("[Telegram] sendVideo exception: %s", type(exc).__name__)
            return False

    def _send_latest_safety_report(self, chat_id: str | int) -> bool:
        """Send the latest saved safety incident details and its own incident clip."""
        incident_log = ROOT / "incident_log.csv"
        try:
            with incident_log.open("r", newline="", encoding="utf-8-sig") as file:
                rows = list(csv.reader(file))
        except OSError as exc:
            logger.warning("[Telegram] Could not read safety incident log: %s", exc)
            self._send_text(chat_id, "The saved safety report is unavailable because the incident log could not be read.")
            return False

        latest: dict[str, str] | None = None
        latest_timestamp = ""
        for row in rows:
            if len(row) < 4 or row[0].strip().lower() == "timestamp":
                continue
            timestamp, event, details, raw_clip = (field.strip() for field in row[:4])
            clip_name = Path(raw_clip).name
            clip_path = INCIDENTS_DIR / clip_name
            if (
                not timestamp
                or not event
                or not raw_clip
                or raw_clip != clip_name
                or not clip_name.lower().endswith(".mp4")
                or not clip_path.is_file()
            ):
                continue
            if timestamp >= latest_timestamp:
                latest = {
                    "timestamp": timestamp,
                    "event": event,
                    "details": details,
                    "clip_name": clip_name,
                }
                latest_timestamp = timestamp

        if latest is None:
            self._send_text(chat_id, "No saved safety incident report and video clip are available yet.")
            return False

        caption = (
            "LATEST SAVED SAFETY INCIDENT REPORT\n"
            "--------------------------------------------------\n"
            f"Primary Violation: {latest['event']}\n"
            f"Details: {latest['details']}\n"
            f"Timestamp: {latest['timestamp']}\n\n"
            "Attached: Saved incident video clip."
        )
        self._send_text(chat_id, caption)
        if not self._send_video(
            chat_id,
            INCIDENTS_DIR / latest["clip_name"],
            caption=f"Safety incident: {latest['event']} ({latest['timestamp']})",
        ):
            self._send_text(chat_id, "The safety report was found, but Telegram could not send its video. Try again.")
            return False
        return True

    def _poll_loop(self) -> None:
        token = self.config.get("bot_token")
        auth_chat_id = str(self.config.get("chat_id", "")).strip()

        # Discard backlog on startup so old messages don't fire alarms immediately
        url_updates = f"https://api.telegram.org/bot{token}/getUpdates"
        try:
            init_resp = requests.get(url_updates, params={"offset": -1, "timeout": 2}, timeout=4)
            if init_resp.status_code == 200:
                results = init_resp.json().get("result", [])
                if results:
                    self.last_update_id = results[-1].get("update_id", 0)
            elif init_resp.status_code == 401:
                print("[Telegram Warning] Bot token is invalid or expired. Polling will keep retrying.", flush=True)
        except Exception:
            pass

        consecutive_auth_errors = 0
        while self.is_running:
            try:
                params = {"offset": self.last_update_id + 1, "timeout": 8}
                resp = requests.get(url_updates, params=params, timeout=12)
                if resp.status_code == 401:
                    consecutive_auth_errors += 1
                    if consecutive_auth_errors >= 2:
                        print("[Telegram Warning] Bot token unauthorized (401). Pausing Telegram polling.", flush=True)
                        time.sleep(30.0)
                    else:
                        time.sleep(5.0)
                    continue
                if resp.status_code != 200:
                    time.sleep(3.0)
                    continue
                consecutive_auth_errors = 0

                updates = resp.json().get("result", [])
                for upd in updates:
                    self.last_update_id = upd.get("update_id", self.last_update_id)
                    msg = upd.get("message") or upd.get("edited_message")
                    if not msg:
                        continue

                    chat = msg.get("chat", {})
                    sender_chat_id = str(chat.get("id", "")).strip()
                    if sender_chat_id != auth_chat_id:
                        # Ignore unauthorized senders
                        continue

                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue

                    print(f"\n[CLIENT TELEGRAM MESSAGE RECEIVED] \"{text}\"")
                    self._process_incoming_command(sender_chat_id, text)

            except Exception:
                time.sleep(2.0)

    def _process_incoming_command(self, chat_id: str, text: str) -> None:
        """Route incoming user query or command to the appropriate VisionAI subsystem."""
        low_text = text.lower().strip()
        local_ip = get_local_ip()
        dashboard_url = f"http://{local_ip}:{self.port}"

        emergency_match = re.match(
            r"^(?:/emergency(?:@\w+)?|emergency)\b(?:[\s,:-]+(.*))?$",
            text,
            re.IGNORECASE,
        )
        emergency_message: str | None = None
        if emergency_match:
            emergency_message = emergency_match.group(1) or ""
        elif low_text in ("help", "help me", "sos"):
            emergency_message = ""
        else:
            shorthand_match = re.match(r"^(?:help(?:\s+me)?|sos)\s*[:,-]\s*(.+)$", text, re.IGNORECASE)
            if shorthand_match:
                emergency_message = shorthand_match.group(1)
        if emergency_message is not None:
            emergency_text = emergency_message.strip()[:400]
            if not emergency_text:
                emergency_text = "Emergency assistance requested. Please respond immediately."
            speaker_alarm.trigger_two_way_alert(f"Emergency message from Telegram: {emergency_text}")
            self._send_text(
                chat_id,
                "EMERGENCY ALERT SENT\n"
                "The message was sent to the VisionAI dashboard and announced through the site speaker.\n"
                "If you need immediate emergency services, contact your local responders directly.",
            )
            if self.on_event:
                self.on_event("TELEGRAM EMERGENCY", f"Emergency reported via Telegram: {emergency_text}")
            return

        # ---------------------------------------------------------------------
        # 1. Emergency Siren Alarm Trigger (Authentic Tornado air-raid wail)
        # ---------------------------------------------------------------------
        if low_text in ("/alarm", "/siren", "alarm", "siren", "wail", "tornado", "sound alarm", "sound siren"):
            speaker_alarm.play_siren_alarm(duration=4.0)
            reply = (
                "VISIONAI SENTINEL - EMERGENCY SIREN ACTIVATED\n"
                "--------------------------------------------------\n"
                "Authentic Tornado air-raid wail siren sounded on site.\n"
                "Type \"stop\" or /stop to silence."
            )
            self._send_text(chat_id, reply)
            if self.on_event:
                self.on_event("EMERGENCY SIREN (TELEGRAM)", "Client triggered Tornado air-raid siren via Telegram.")
            return

        # ---------------------------------------------------------------------
        # 2. Stop / Silence Siren
        # ---------------------------------------------------------------------
        if low_text in ("/stop", "stop", "silence", "quiet", "off"):
            speaker_alarm.stop_alarm()
            reply = (
                "VISIONAI SENTINEL - SIREN SILENCED\n"
                "--------------------------------------------------\n"
                "Emergency siren stopped."
            )
            self._send_text(chat_id, reply)
            if self.on_event:
                self.on_event("EMERGENCY SIREN SILENCED", "Client silenced alarm via Telegram.")
            return

        # ---------------------------------------------------------------------
        # 3. Two-Way Voice Intercom (/speak <text> or say <text>)
        # ---------------------------------------------------------------------
        if low_text.startswith("/speak") or low_text.startswith("say "):
            speech_text = text[6:].strip() if low_text.startswith("/speak") else text[4:].strip()
            if speech_text:
                speaker_alarm.trigger_two_way_alert(speech_text)
                reply = f"[USB SPEAKER VOICE] Broadcasted warning on site: \"{speech_text}\""
                self._send_text(chat_id, reply)
                if self.on_event:
                    self.on_event("REMOTE VOICE BROADCAST", f"Client broadcasted: \"{speech_text}\"")
                return

        # ---------------------------------------------------------------------
        # 4. System Status & Live Web Command Center Link
        # ---------------------------------------------------------------------
        if low_text in ("/report", "report", "/incident", "incident"):
            self._send_latest_safety_report(chat_id)
            return

        camera_command = re.fullmatch(
            r"/camera(?:@\w+)?(?:\s+(start|stop|status)(?:\s+(\d+))?)?",
            text.strip(),
            re.IGNORECASE,
        )
        if camera_command or low_text in ("camera", "camera status", "camera start", "camera stop"):
            action = (
                (camera_command.group(1) or "status").lower()
                if camera_command
                else (low_text.split()[-1] if low_text != "camera" else "status")
            )
            if not action or action == "status":
                status_data = self.status_cb() if self.status_cb else {}
                running = bool(status_data.get("running"))
                has_frame = bool(status_data.get("has_frame"))
                return self._send_text(
                    chat_id,
                    f"Shared CCMS camera: {'LIVE' if running and has_frame else 'CONNECTING' if running else 'STANDBY'}\n"
                    f"Camera index: {status_data.get('camera', 'Not available')}\n"
                    f"FPS: {float(status_data.get('fps', 0.0)):.1f}\n"
                    "This camera is used by the CCMS dashboard.",
                )
            if not self.camera_control_cb:
                return self._send_text(chat_id, "Camera controls are unavailable in this dashboard session.")
            camera_index = int(camera_command.group(2)) if camera_command and camera_command.group(2) else None
            if camera_index is not None and camera_index > 9:
                return self._send_text(chat_id, "Camera index must be between 0 and 9.")
            try:
                ok, message = self.camera_control_cb(action, camera_index)
            except Exception as exc:
                logger.exception("[Telegram] Camera control failed")
                return self._send_text(chat_id, f"Camera {action} failed: {exc}")
            self._send_text(chat_id, message)
            if ok and self.on_event:
                self.on_event(
                    "TELEGRAM CAMERA CONTROL",
                    f"Authorized Telegram user requested camera {action}."
                    + (f" Camera index: {camera_index}." if camera_index is not None else ""),
                )
            return ok

        if low_text in ("/status", "status", "live", "system", "health", "info", "dashboard", "link", "web", "url"):
            status_data: dict[str, Any] = {}
            if self.status_cb:
                try:
                    status_data = self.status_cb()
                except Exception:
                    pass

            is_running = status_data.get("running", True)
            fps = status_data.get("fps", 24.0)
            violations = status_data.get("violations", [])
            v_desc = ", ".join(violations) if violations else "None (All Zones Clear)"

            obj_count = 0
            if self.tracker:
                try:
                    obj_count = len(self.tracker.get_all_tracked())
                except Exception:
                    pass

            reply = (
                "VISIONAI SENTINEL - SYSTEM TELEMETRY\n"
                "--------------------------------------------------\n"
                f"Surveillance Camera: {'ONLINE (Active)' if is_running else 'STANDBY'}\n"
                f"Processing Speed: {fps:.1f} FPS\n"
                "Batch Cycle: 3-Minute Periodic Audit Active\n"
                f"Active Violations: {v_desc}\n"
                f"Tracked Catalog: {obj_count} Monitored Objects\n\n"
                "Web Command Center:\n"
                f"{dashboard_url}/\n"
                f"(Local access: http://localhost:{self.port}/)\n\n"
                "Two-Way AI Search:\n"
                "Type any object name (e.g. \"bottle\", \"phone\", \"bag\") to retrieve real-time location and video clips."
            )
            self._send_text(chat_id, reply)
            return

        # ---------------------------------------------------------------------
        # 5. Help / Menu
        # ---------------------------------------------------------------------
        if low_text in ("/start", "/help", "help", "commands", "menu"):
            help_text = (
                "VISIONAI SENTINEL - TWO-WAY AI ASSISTANT\n"
                "--------------------------------------------------\n"
                "1. AI OBJECT SEARCH & FOOTAGE RETRIEVER:\n"
                "* Search any object: \"cell phone\", \"bottle\", \"laptop\", \"bag\"\n"
                "-> Returns the latest location, snapshot photo, and 10s video clip.\n\n"
                "2. EMERGENCY ALERT:\n"
                "* Send /emergency <message>, \"help\", or \"SOS\" to show the alert on the dashboard and announce it on the site speaker.\n\n"
                "3. SAFETY INCIDENT REPORTS:\n"
                "* Send /report to receive the latest saved safety incident details and its video clip.\n\n"
                "4. SHARED CAMERA CONTROLS:\n"
                "* /camera start [index] -> Start the shared CCMS USB camera.\n"
                "* /camera stop -> Stop the shared CCMS USB camera.\n"
                "* /camera status -> Check the shared camera feed.\n"
                "5. EMERGENCY SIREN & VOICE CONTROLS:\n"
                "* \"alarm\" or /alarm -> Sound Emergency Siren on site\n"
                "* \"stop\" or /stop -> Silence Siren\n"
                "* /speak <text> -> Broadcast voice message through on-site speaker\n"
                "* \"status\" -> Live camera health, FPS, and web dashboard link\n"
                f"* \"dashboard\" -> Open web command center: {dashboard_url}/"
            )
            self._send_text(chat_id, help_text)
            return

        # ---------------------------------------------------------------------
        # 6. AI Object Search & Live Detection Video Clip Retriever
        # ---------------------------------------------------------------------
        play_match = re.match(r"^(?:/play(?:@\w+)?|play|watch|view|send|show(?:\s+me)?)\b", text, re.IGNORECASE)
        is_play_request = play_match is not None
        if is_play_request:
            query_term = text[play_match.end():].strip(" \t:,-")
            query_term = re.sub(r"^(?:me\s+)?the\s+", "", query_term, flags=re.IGNORECASE)
            query_term = re.sub(
                r"\s+(?:(?:last\s+location)\s+)?(?:clip|video|footage)$",
                "",
                query_term,
                flags=re.IGNORECASE,
            ).strip(" \t?.,!")
            if not query_term:
                query_term = self.last_object_query.get(chat_id, "")
            if not query_term and self.status_cb:
                try:
                    query_term = str(self.status_cb().get("active_target", "")).strip()
                except Exception as exc:
                    logger.warning("[Telegram] Could not read dashboard object target: %s", exc)
            if not query_term:
                self._send_text(
                    chat_id,
                    "Which object's clip should I play? Send /play bottle, or search for an object first and then reply \"Play\".",
                )
                return
        else:
            query_term = text.strip()

        # Check if tracker is present
        if self.tracker:
            results = self.tracker.search(query_term)
            if results:
                top = results[0]
                label = top.get("label", "object")
                display_name = top.get("display_name", label.title())
                last_seen = top.get("last_seen_time", "Recent")
                location = top.get("location_desc", "Within Field of View")
                conf = top.get("confidence", "92%")
                clip_name = top.get("video_clip", "")
                snap_name = top.get("snapshot_file", "")
                self.last_object_query[chat_id] = label

                # Notify dashboard to focus targeting reticle on this searched object in real-time
                if self.set_target_cb:
                    try:
                        self.set_target_cb(label)
                    except Exception:
                        pass

                safe_clip_name = Path(clip_name).name if clip_name else ""
                verified_clip = bool(
                    safe_clip_name
                    and self.tracker._is_object_detection_clip(safe_clip_name, label)
                    and (INCIDENTS_DIR / safe_clip_name).is_file()
                )

                # Send Cropped Snapshot Photo
                photo_sent = False
                if snap_name:
                    snap_path = SNAPSHOT_DIR / snap_name
                    if snap_path.is_file():
                        photo_sent = self._send_photo(
                            chat_id,
                            snap_path,
                            caption=f"Snapshot: {display_name} ({location}, last seen {last_seen})",
                        )

                # Send 10-second target-specific clip
                video_sent = False
                if verified_clip:
                    clip_path = INCIDENTS_DIR / safe_clip_name
                    if clip_path.is_file():
                        video_sent = self._send_video(
                            chat_id,
                            clip_path,
                            caption=f"10-Second Captured Clip: {display_name}",
                        )

                if is_play_request:
                    if not video_sent:
                        if top.get("clip_recording"):
                            self._send_text(
                                chat_id,
                                f"I'm recording a 10s clip of {display_name} now. It will be delivered as soon as ready.",
                            )
                            self._request_object_clip_delivery(chat_id, label, display_name)
                        else:
                            self._send_text(chat_id, f"No verified 10s {display_name} clip available yet.")
                    return

                # Regular Search: return concise summary along with photo & clip
                report_text = (
                    "VISIONAI - OBJECT FOUND\n"
                    "------------------------------------\n"
                    f"Object: {display_name}\n"
                    f"Last Seen: {last_seen}\n"
                    f"Location: {location}\n"
                    f"Confidence: {conf}\n"
                )
                self._send_text(chat_id, report_text)

                if not verified_clip and top.get("clip_recording"):
                    self._send_text(chat_id, f"Recording a 10s {display_name} clip; it will arrive shortly.")
                    self._request_object_clip_delivery(chat_id, label, display_name)

                if self.on_event:
                    self.on_event(
                        "TELEGRAM OBJECT RETRIEVAL",
                        f"Client searched '{query_term}'. Found '{display_name}' ({location}).",
                    )
                return

        # ---------------------------------------------------------------------
        # Fallback: Object Not Found in Active Field of View
        # ---------------------------------------------------------------------
        available_items_str = ""
        if self.tracker:
            try:
                all_items = self.tracker.get_all_tracked()
                names = [it.get("display_name", it.get("label", "")).split("/")[0].strip() for it in all_items[:8]]
                if names:
                    available_items_str = "\n".join(f"* {n}" for n in names)
            except Exception:
                pass

        if not available_items_str:
            available_items_str = "* Bottle\n* Backpack / Book Bag\n* Cell Phone\n* Pen\n* Pencil\n* Camera\n* Keys\n* Laptop"

        not_found_reply = (
            "VISIONAI SENTINEL - OBJECT NOT CURRENTLY IN FOV\n"
            "--------------------------------------------------\n"
            f"No surveillance record currently matched \"{query_term}\".\n\n"
            "Currently Monitored Catalog:\n"
            f"{available_items_str}\n\n"
            "Tip: Search for any item above (e.g. \"cell phone\", \"bottle\", \"bag\"), or type \"status\" for camera stream link."
        )
        self._send_text(chat_id, not_found_reply)

    def _request_object_clip_delivery(self, chat_id: str, label: str, display_name: str) -> None:
        key = (str(chat_id), label)
        with self._pending_clip_deliveries_lock:
            if key in self._pending_clip_deliveries:
                return
            self._pending_clip_deliveries.add(key)
        threading.Thread(
            target=self._send_object_clip_delivery_and_clear,
            args=(key, label, display_name),
            daemon=True,
        ).start()

    def _send_object_clip_delivery_and_clear(
        self,
        key: tuple[str, str],
        label: str,
        display_name: str,
    ) -> None:
        try:
            self._send_object_clip_when_ready(key[0], label, display_name)
        finally:
            with self._pending_clip_deliveries_lock:
                self._pending_clip_deliveries.discard(key)

    def _send_object_clip_when_ready(self, chat_id: str, label: str, display_name: str) -> None:
        """Send a target's clip after its in-progress seven-second capture is encoded."""
        deadline = time.monotonic() + 45.0
        while time.monotonic() < deadline:
            time.sleep(1.0)
            try:
                matches = self.tracker.search(label) if self.tracker else []
                item = next((entry for entry in matches if entry.get("label") == label), None)
                clip_name = item.get("video_clip", "") if item else ""
                if not clip_name:
                    continue
                safe_name = Path(clip_name).name
                clip_path = INCIDENTS_DIR / safe_name
                if not self.tracker._is_object_detection_clip(safe_name, label) or not clip_path.is_file():
                    continue
                if self._send_video(
                    chat_id,
                    clip_path,
                    caption=f"Verified detection clip: {display_name} ({item.get('location_desc', 'last detected location')})",
                ):
                    self._send_latest_safety_report(chat_id)
                    return
                self._send_text(chat_id, f"The matching {display_name} clip was recorded, but Telegram could not send it. Try /play {label} again.")
                return
            except Exception as exc:
                logger.warning("[Telegram] Could not retrieve %s clip: %s", label, exc)
                break

        self._send_text(
            chat_id,
            f"I couldn't finish a verified {display_name} clip. Make sure the camera is running and the object remains visible, then send /play {label} again.",
        )
