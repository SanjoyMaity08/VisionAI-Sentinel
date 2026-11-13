from __future__ import annotations

import ast
import collections
from datetime import datetime
import math
import os
from pathlib import Path
import queue
import re
import tempfile
import threading
import time
from typing import Any, Callable
import unittest
from unittest.mock import Mock, patch
import json

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def isolated_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (target.id.startswith(("CLASS_", "COLOR_")) or target.id in names):
                    nodes.append(node)
                    break
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, "exec"), namespace)


class AlertPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.incidents = Path(self.temp.name)
        self.listener = Mock()
        self.listener.send_violation_clip.return_value = True
        self.tasks = []
        self.env = dict(
            collections=collections, deque=collections.deque, datetime=datetime, math=math, Path=Path,
            queue=queue, re=re, os=os, threading=threading, time=time, Any=Any,
            Callable=Callable, json=json, np=np, cv2=cv2,
            INCIDENTS_DIR=self.incidents, ROOT=self.incidents,
            append_incident_log=Mock(), GLOBAL_OBJECT_TRACKER=Mock(),
            handle_two_way_event=Mock(), speaker_alarm=Mock(),
            draw_corner_reticle=Mock(), _uploaded_video_listener=lambda: self.listener,
            save_browser_h264_clip=Mock(),
        )
        isolated_definitions("app.py", {
            "ViolationDeliveryQueue", "PeriodicClipEngine", "CRITICAL_AUTO_DISPATCH_TITLES",
            "INFORMATIONAL_CLIP_TITLES", "EMERGENCY_ALERT_COOLDOWN_SEC",
            "dispatch_camera_emergency", "dispatch_camera_violation", "handle_uploaded_video_event",
        }, self.env)
        isolated_definitions("emergency_action_detector.py", {"EmergencyActionDetector"}, self.env)
        self.delivery = self.env["ViolationDeliveryQueue"](self.incidents / "outbox.json")
        self.delivery.next_batch = 180.0
        self.env["GLOBAL_DELIVERY"] = self.delivery
        self.engine = self.env["PeriodicClipEngine"](emergency_callback=Mock(), violation_callback=Mock())
        self.pose = self.env["EmergencyActionDetector"]()
        tasks = self.tasks

        class DeferredThread:
            def __init__(self, target, args=(), **kwargs):
                self.target, self.args = target, args
            def start(self):
                tasks.append((self.target, self.args))

        self.thread_patch = patch.object(threading, "Thread", DeferredThread)
        self.thread_patch.start()
        self.addCleanup(self.thread_patch.stop)

    def clip(self, name):
        path = self.incidents / (name + ".mp4")
        path.write_bytes(b"mock clip")
        return path

    def run_tasks(self):
        while self.tasks:
            target, args = self.tasks.pop(0)
            target(*args)

    @staticmethod
    def person(x=500, y=250, class_id=0):
        return {"bbox": [x - 40, y - 80, x + 40, y + 80], "center": (x, y), "class_id": class_id}

    def evaluate(self, people, now, events=None):
        return self.engine.evaluate_core_rules(np.zeros((600, 1000, 3), dtype=np.uint8), [dict(p) for p in people], now, events, "ready")

    def test_no_send_before_three_minutes(self):
        self.delivery.enqueue(self.clip("one"), "Event one")
        for now in (0, 10, 100, 179.999):
            self.assertFalse(self.delivery.dispatch(now))
        self.assertFalse(self.tasks)
        self.listener.send_violation_clip.assert_not_called()

    def test_batch_all_once_and_quiet_empty_cycle(self):
        for name in ("one", "two", "three"):
            self.delivery.enqueue(self.clip(name), name)
        self.assertTrue(self.delivery.dispatch(180))
        self.assertFalse(self.delivery.dispatch(180))
        self.run_tasks()
        self.assertEqual(self.listener.send_violation_clip.call_count, 3)
        self.assertEqual(self.delivery.status()["pending_count"], 0)
        self.assertFalse(self.delivery.dispatch(360))
        self.assertFalse(self.delivery.enqueue(self.incidents / "one.mp4", "duplicate"))
        self.assertEqual(self.listener.send_violation_clip.call_count, 3)

    def test_failure_retries_only_next_cycle(self):
        self.delivery.enqueue(self.clip("one"), "caption")
        self.listener.send_violation_clip.return_value = False
        self.delivery.dispatch(180)
        self.run_tasks()
        self.assertEqual(self.delivery.status()["pending_count"], 1)
        self.assertFalse(self.delivery.dispatch(181))
        self.listener.send_violation_clip.return_value = True
        self.delivery.dispatch(360)
        self.run_tasks()
        self.assertEqual(self.listener.send_violation_clip.call_count, 2)
        self.assertEqual(self.delivery.status()["pending_count"], 0)

    def test_new_clip_during_send_waits_next_batch(self):
        self.delivery.enqueue(self.clip("first"), "first")
        second = self.clip("second")
        def send(path, caption):
            self.delivery.enqueue(second, "second")
            return True
        self.listener.send_violation_clip.side_effect = send
        self.delivery.dispatch(180)
        self.run_tasks()
        self.assertEqual(self.listener.send_violation_clip.call_count, 1)
        self.assertEqual(self.delivery.status()["pending_count"], 1)
        self.listener.send_violation_clip.side_effect = None
        self.delivery.dispatch(360)
        self.run_tasks()
        self.assertEqual(self.listener.send_violation_clip.call_count, 2)

    def test_restart_preserves_pending_and_acknowledged(self):
        first, second = self.clip("first"), self.clip("second")
        self.delivery.enqueue(first, "first")
        self.delivery.dispatch(180)
        self.run_tasks()
        self.delivery.enqueue(second, "second")
        restored = self.env["ViolationDeliveryQueue"](self.delivery.path)
        self.assertEqual(restored.status()["pending_count"], 1)
        self.assertFalse(restored.enqueue(first, "duplicate"))

    def test_unconfigured_delivery_stays_pending(self):
        self.env["_uploaded_video_listener"] = lambda: None
        self.delivery.enqueue(self.clip("one"), "one")
        self.delivery.dispatch(180)
        self.run_tasks()
        self.assertEqual(self.delivery.status()["pending_count"], 1)
        self.listener.send_violation_clip.assert_not_called()

    def test_uploads_enqueue_without_camera_or_immediate_send(self):
        for kind in ("helmetless", "red_light", "vehicle_speed"):
            event = {"type": kind, "overspeed": True, "clip_file": self.clip(kind).name}
            self.env["handle_uploaded_video_event"]("sample.mp4", event)
            self.env["handle_uploaded_video_event"]("sample.mp4", event)
            self.assertEqual(event["delivery_status"], "queued")
            self.assertFalse(event["clip_sent"])
        self.assertFalse(self.tasks)
        self.assertEqual(self.delivery.status()["pending_count"], 3)
        self.delivery.dispatch(180)
        self.run_tasks()
        self.assertEqual(self.listener.send_violation_clip.call_count, 3)
        self.assertEqual(event["delivery_status"], "sent")

    def test_upload_without_clip_stays_local(self):
        event = {"type": "helmetless"}
        self.env["handle_uploaded_video_event"]("sample.mp4", event)
        self.assertEqual(event["delivery_status"], "no_clip")
        self.assertEqual(self.delivery.status()["pending_count"], 0)

    def test_emergency_is_local_only(self):
        self.env["dispatch_camera_emergency"]("fall", "possible fall")
        self.env["speaker_alarm"].trigger_two_way_alert.assert_called_once()
        self.env["handle_two_way_event"].assert_called_once()
        self.assertFalse(self.tasks)
        self.listener.send_violation_clip.assert_not_called()

    def test_manual_batch_is_explicit_and_deduplicated(self):
        self.delivery.enqueue(self.clip("one"), "one")
        self.assertTrue(self.delivery.dispatch(20, force=True))
        self.assertFalse(self.delivery.dispatch(20, force=True))
        self.run_tasks()
        self.assertFalse(self.delivery.dispatch(21, force=True))
        self.assertEqual(self.listener.send_violation_clip.call_count, 1)

    def test_perimeter_confirms_at_twelve_seconds(self):
        person = self.person()
        for now in range(12):
            self.evaluate([person], float(now))
            self.assertFalse(self.engine._active_captures)
        self.evaluate([person], 12.0)
        self.assertIn("Machine Danger Perimeter Breach", self.engine._active_captures)
        self.engine.violation_callback.assert_called_once()

    def test_missing_evidence_resets_confirmation(self):
        person = self.person()
        for now in range(10):
            self.evaluate([person], float(now))
        self.evaluate([], 10.0)
        self.evaluate([person], 11.0)
        self.evaluate([person], 12.0)
        self.assertFalse(self.engine._active_captures)

    def test_stalled_camera_does_not_count_elapsed_gap(self):
        for now in range(10):
            self.evaluate([self.person()], float(now))
        self.evaluate([self.person()], 20.0)
        self.assertFalse(self.engine._active_captures)

    def test_changed_subject_does_not_inherit_progress(self):
        for now in range(10):
            self.evaluate([self.person()], float(now))
        self.evaluate([self.person(x=800)], 10.0)
        self.evaluate([self.person(x=800)], 12.0)
        self.assertFalse(self.engine._active_captures)

    def test_continuous_episode_does_not_repeat(self):
        for now in range(150):
            self.evaluate([self.person()], float(now))
        self.engine.violation_callback.assert_called_once()

    def test_colour_and_bbox_shape_not_violation_proof(self):
        for now in range(20):
            result = self.evaluate([self.person(x=200)], float(now))
        self.assertFalse(self.engine._active_captures)
        self.assertEqual(result["radar_status"]["PPE Safety Vest"][1], "warning")

    def test_unconfirmed_and_stale_pose_events_ignored(self):
        for event in ({"type": "fall", "evidence_seconds": .8, "captured_at": 12},
                      {"type": "fall", "evidence_seconds": 12, "captured_at": 1}):
            self.evaluate([], 12, [event])
        self.assertFalse(self.engine._active_captures)
        self.engine.emergency_callback.assert_not_called()

    def test_confirmed_pose_event_captures_and_warns(self):
        event = {"type": "fall", "evidence_seconds": 12, "captured_at": 12, "details": "possible fall"}
        self.evaluate([], 12, [event])
        self.assertIn("Sudden Fall and Emergency Collapse", self.engine._active_captures)
        self.engine.emergency_callback.assert_called_once()
        self.listener.send_violation_clip.assert_not_called()

    def test_encoder_enqueues_only_actual_violation(self):
        def save(path, frames, fps):
            path.write_bytes(b"mock clip")
            return True
        self.env["save_browser_h264_clip"].side_effect = save
        for title in ("Machine Danger Perimeter Breach", "VIP Welcoming Alert"):
            self.engine._encode_incident_clip([np.zeros((2, 2, 3), np.uint8)], {"title": title, "timestamp": "2026-11-13 12:00:00"})
        self.assertEqual(self.delivery.status()["pending_count"], 1)
        self.listener.send_violation_clip.assert_not_called()

    def test_evidence_buffer_includes_onset(self):
        frame = np.zeros((2, 2, 3), np.uint8)
        for now in range(12):
            self.engine.update_recording(frame, float(now))
        self.engine.trigger_clip_capture("test", "details", frame, now=12)
        self.engine.update_recording(frame, 12)
        for now in (13, 14, 15):
            self.engine.update_recording(frame, now)
        self.assertEqual(len(self.tasks), 1)
        _, (_, meta, fps) = self.tasks[0]
        self.assertEqual(meta["duration_seconds"], "15.00")
        self.assertGreater(fps, 0)

    @staticmethod
    def pose_sample(horizontal=False, center=(300, 300)):
        points = np.zeros((17, 2), dtype=float)
        points[5], points[6] = (280, 250), (300, 250)
        points[11], points[12] = ((350, 260), (370, 260)) if horizontal else ((280, 330), (300, 330))
        return {"center": center, "bbox": (220, 220, 400, 350) if horizontal else (240, 180, 350, 420), "keypoints": points, "scores": np.ones(17)}

    def test_pose_fall_twelve_seconds_once(self):
        self.pose._analyze_poses([self.pose_sample()], 1000, 600, 0)
        for now in range(1, 13):
            self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, now)
        self.assertFalse(self.pose.drain_events())
        self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, 13)
        events = self.pose.drain_events()
        self.assertEqual([event["type"] for event in events], ["fall"])
        self.assertEqual(events[0]["evidence_seconds"], 12)
        self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, 14)
        self.assertFalse(self.pose.drain_events())

    def test_initial_horizontal_not_fall_or_sleep(self):
        for now in range(20):
            self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, now)
        self.assertFalse(self.pose.drain_events())

    def test_pose_missing_or_stale_evidence_resets(self):
        self.pose._analyze_poses([self.pose_sample()], 1000, 600, 0)
        for now in range(1, 10):
            self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, now)
        self.pose._analyze_poses([], 1000, 600, 10)
        self.pose._analyze_poses([self.pose_sample(True)], 1000, 600, 20)
        self.assertFalse(self.pose.drain_events())
        self.assertFalse(self.pose.status()["candidates"])

    def test_raised_arm_same_pair_requires_twelve_seconds(self):
        actor, target = self.pose_sample(), self.pose_sample(center=(500, 300))
        with patch.object(self.pose, "_raised_hand_threat", side_effect=lambda pair, *args, **kwargs: pair[0]["center"] == actor["center"]), patch.object(self.pose, "_arm_contact_candidate", return_value=False):
            for now in range(12):
                self.pose._analyze_poses([actor, target], 1000, 600, now)
            self.assertFalse(self.pose.drain_events())
            self.pose._analyze_poses([actor, target], 1000, 600, 12)
            self.assertEqual(len(self.pose.drain_events()), 1)

    def test_brief_raised_arm_does_not_confirm(self):
        actor, target = self.pose_sample(), self.pose_sample(center=(500, 300))
        with patch.object(self.pose, "_raised_hand_threat", return_value=True), patch.object(self.pose, "_arm_contact_candidate", return_value=False):
            for now in range(5):
                self.pose._analyze_poses([actor, target], 1000, 600, now)
        self.pose._analyze_poses([], 1000, 600, 5)
        self.assertFalse(self.pose.drain_events())
        self.assertFalse(self.pose.status()["candidates"])

    def test_queue_and_desk_confirm_after_twelve_seconds(self):
        people = [self.person(x=500, y=430), self.person(x=650, y=430)]
        for now in range(12):
            self.evaluate(people, now)
        self.assertFalse(self.engine._active_captures)
        self.evaluate(people, 12)
        self.assertIn("Queue Overflow Alert", self.engine._active_captures)
        self.assertIn("Unmanned Billing Desk Alert", self.engine._active_captures)

    def test_phone_proximity_outside_desk_not_violation(self):
        person = self.person(x=800)
        phone = self.person(x=800, class_id=67)
        for now in range(14):
            self.evaluate([person, phone], now)
        self.assertNotIn("Sustained Phone Presence at Desk", self.engine._active_captures)

    def test_phone_at_desk_requires_twelve_seconds(self):
        person = self.person(x=200)
        phone = {"class_id": 67, "center": (200, 250), "bbox": [185, 230, 215, 270]}
        for now in range(12):
            self.evaluate([person, phone], now)
        self.assertFalse(self.engine._active_captures)
        self.evaluate([person, phone], 12)
        self.assertIn("Sustained Phone Presence at Desk", self.engine._active_captures)

    def test_floor_object_requires_twelve_seconds(self):
        bottle = {"class_id": 39, "center": (500, 550), "bbox": [480, 500, 520, 590]}
        for now in range(12):
            self.evaluate([bottle], now)
        self.assertFalse(self.engine._active_captures)
        self.evaluate([bottle], 12)
        self.assertIn("Unattended Floor Object Alert", self.engine._active_captures)

    def test_edge_dwell_requires_fifteen_seconds(self):
        person = self.person(x=50)
        for now in range(15):
            self.evaluate([person], now)
        self.assertFalse(self.engine._active_captures)
        self.evaluate([person], 15)
        self.assertIn("Perimeter Dwell Alert", self.engine._active_captures)

    def test_telegram_success_requires_api_acknowledgement(self):
        listener_tree = ast.parse((ROOT / "telegram_listener.py").read_text(encoding="utf-8-sig"))
        cls = next(node for node in listener_tree.body if isinstance(node, ast.ClassDef) and node.name == "TelegramTwoWayListener")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_send_video")
        cls.body = [method]
        namespace = {"Path": Path, "requests": Mock(), "logger": Mock()}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "telegram_listener.py", "exec"), namespace)
        instance = namespace["TelegramTwoWayListener"]()
        instance.config = {"bot_token": "dummy-test-token"}
        response = namespace["requests"].post.return_value
        response.status_code = 200
        response.json.return_value = {"ok": False}
        self.assertFalse(instance._send_video("dummy-chat", self.clip("test"), "test"))
        response.json.return_value = {"ok": True}
        self.assertTrue(instance._send_video("dummy-chat", self.incidents / "test.mp4", "test"))

    def test_session_reset_clears_all_evidence(self):
        self.evaluate([self.person()], 0)
        self.engine.reset_emergency_state()
        self.pose.reset()
        self.assertFalse(self.engine._rule_evidence)
        self.assertFalse(self.pose.status()["candidates"])


if __name__ == "__main__":
    unittest.main()
