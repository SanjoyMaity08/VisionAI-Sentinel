"""Gyanovate VisionAI - USB Speaker / Headphone Audio Alarm & Voice Announcer.

Outputs real-time security alerts through the connected USB speaker,
external sound card, or headphones using:
1. High-intensity dual-tone emergency siren synthesis (via Windows winsound)
2. Windows SAPI Voice Speech Synthesizer (for spoken voice warnings)
3. Fallback tone beeps and browser-synchronized Web Audio alerts
"""
from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
import wave

import numpy as np

_alarm_active = False
_alarm_lock = threading.Lock()
_last_siren_time: float = 0.0
SIREN_COOLDOWN_SEC: float = 60.0  # Increased cooldown to strictly prevent continuous siren screeching
_SIREN_CACHE: dict[tuple[float, int], bytes] = {}
_SIREN_CACHE_LOCK = threading.Lock()
_mute_until: float = 0.0


def mute_for(seconds: float) -> float:
    """Silence the local siren and voice announcer for the given number of seconds."""
    global _mute_until
    seconds = max(0.0, float(seconds))
    _mute_until = time.monotonic() + seconds
    stop_alarm()
    return seconds


def clear_mute() -> None:
    global _mute_until
    _mute_until = 0.0


def mute_remaining() -> float:
    return max(0.0, _mute_until - time.monotonic())


def is_muted() -> bool:
    return mute_remaining() > 0.0


def generate_siren_wav(duration: float = 3.5, sample_rate: int = 22050) -> bytes:
    """Generate an authentic mechanical tornado air-raid wail siren in standard 16-bit PCM WAV format.

    Features:
    - Continuous phase integration (no clicks or phase jumps)
    - Slow ominous frequency sweep between 400 Hz and 840 Hz (2.4s cycle)
    - Realistic acoustic horn harmonics (fundamental + 2nd & 3rd harmonics + acoustic saturation)
    - Smooth attack and release envelope
    """
    key = (round(max(0.0, float(duration)), 2), int(sample_rate))
    with _SIREN_CACHE_LOCK:
        cached = _SIREN_CACHE.get(key)
        if cached is not None:
            return cached

    duration = key[0]
    sample_rate = key[1]
    num_samples = int(sample_rate * duration)
    timeline = np.arange(num_samples, dtype=np.float64) / sample_rate

    # Attack and release envelope to prevent abrupt pop
    envelope = np.minimum(1.0, timeline / 0.25) * np.minimum(1.0, (duration - timeline) / 0.35)

    # Continuous phase integral: phi(t) = 2*pi*f_mid*t - (f_amp/cycle_freq)*sin(2*pi*cycle_freq*t)
    phase = (
        2.0 * np.pi * 620.0 * timeline
        - (220.0 / 0.42) * np.sin(2.0 * np.pi * 0.42 * timeline)
    )

    # Authentic siren horn timbre: rich fundamental + rotor harmonics
    raw = 0.62 * np.sin(phase) + 0.26 * np.sin(2.0 * phase) + 0.12 * np.sin(3.0 * phase)

    # Acoustic horn saturation
    samples = np.clip(28000.0 * envelope * np.tanh(raw * 1.45), -32768, 32767).astype("<i2")

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())

    wav_bytes = wav_io.getvalue()
    with _SIREN_CACHE_LOCK:
        _SIREN_CACHE[key] = wav_bytes
    return wav_bytes


def play_siren_alarm(duration: float = 2.5) -> bool:
    """Sound the emergency siren on the connected USB speaker in a background thread.

    Enforces a cooldown of SIREN_COOLDOWN_SEC seconds between consecutive triggers
    to prevent continuous siren loops on repeated violation detections.
    Returns False when the siren was suppressed by mute or cooldown.
    """
    global _alarm_active, _last_siren_time

    if is_muted():
        print(f"[USB SPEAKER] Siren suppressed — muted for another {mute_remaining():.0f}s.")
        return False

    import time as _time
    now = _time.monotonic()
    with _alarm_lock:
        if now - _last_siren_time < SIREN_COOLDOWN_SEC and _last_siren_time > 0:
            print(f"[USB SPEAKER] Siren suppressed — cooldown active ({SIREN_COOLDOWN_SEC:.0f}s between triggers).")
            return False
        _last_siren_time = now

    def _worker():
        global _alarm_active
        with _alarm_lock:
            _alarm_active = True

        print(f"[USB SPEAKER] Sounding emergency security siren ({duration:.1f}s)...")
        wav_bytes = generate_siren_wav(duration=duration)

        # Method 1: Winsound on Windows
        if sys.platform.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
            except Exception as e:
                # Method 2: PowerShell Media.SoundPlayer fallback
                try:
                    import tempfile
                    from pathlib import Path
                    tmp_wav = Path(tempfile.gettempdir()) / "visionai_siren.wav"
                    tmp_wav.write_bytes(wav_bytes)
                    cmd = f"$p = New-Object System.Media.SoundPlayer '{tmp_wav}'; $p.PlaySync()"
                    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], timeout=int(duration) + 3, capture_output=True)
                except Exception as e2:
                    print(f"[USB Speaker Warning] Audio output error: {e2}")

        with _alarm_lock:
            _alarm_active = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


def speak_voice_announcement(text: str) -> None:
    """Announce a spoken security warning aloud through the USB speaker using Windows SAPI."""
    if is_muted():
        print(f"[USB SPEAKER VOICE] Announcement suppressed — muted for another {mute_remaining():.0f}s.")
        return

    def _worker():
        clean_text = " ".join("".join(char if char.isprintable() else " " for char in text).split())[:300]
        if not clean_text:
            return
        print(f"[USB SPEAKER VOICE] Announcing: \"{clean_text}\"")

        if sys.platform.startswith("win"):
            try:
                # Use Windows SAPI SpVoice COM object via PowerShell
                safe_text = clean_text.replace("'", "''")
                cmd = f"$s = New-Object -ComObject SAPI.SpVoice; $s.Rate = 0; $s.Speak('{safe_text}')"
                subprocess.run(["powershell", "-NoProfile", "-Command", cmd], timeout=15, capture_output=True)
            except Exception as exc:
                print(f"[Voice Announce Error] {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def trigger_two_way_alert(custom_text: str | None = None, siren_duration: float = 1.5) -> None:
    """Sound a short siren burst followed by the spoken security announcement."""
    if is_muted():
        print(f"[USB SPEAKER] Two-way alert suppressed — muted for another {mute_remaining():.0f}s.")
        return

    def _sequence():
        wav_bytes = generate_siren_wav(duration=max(0.0, siren_duration))
        if sys.platform.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(wav_bytes, winsound.SND_MEMORY)
            except Exception:
                pass
        time.sleep(0.2)

        # 2. Speak announcement
        message = custom_text if custom_text else "Warning: Security violation detected. Please comply with safety protocols."
        speak_voice_announcement(message)

    threading.Thread(target=_sequence, daemon=True).start()


def stop_alarm() -> None:
    """Silence any active alarm immediately."""
    global _alarm_active
    with _alarm_lock:
        _alarm_active = False
    if sys.platform.startswith("win"):
        try:
            import winsound
            winsound.PlaySound(None, 0)
        except Exception:
            pass


def is_alarm_active() -> bool:
    return _alarm_active
