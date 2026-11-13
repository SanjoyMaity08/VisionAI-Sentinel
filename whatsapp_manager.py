"""Gyanovate VisionAI - Live WhatsApp Direct Alert Dispatcher.

Supports:
1. CallMeBot WhatsApp Gateway (instant, free, personal number)
2. Twilio WhatsApp API (enterprise sandbox & production)
3. Generic WhatsApp Webhook
"""
from __future__ import annotations

import json
from pathlib import Path
import urllib.parse
import requests

ROOT = Path(__file__).resolve().parent
WHATSAPP_CONFIG_FILE = ROOT / "whatsapp_config.json"


def load_whatsapp_config() -> dict | None:
    if not WHATSAPP_CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(WHATSAPP_CONFIG_FILE.read_text(encoding="utf-8"))
        if data.get("phone") and (data.get("api_key") or data.get("auth_token")):
            return data
    except Exception:
        return None
    return None


def save_whatsapp_config(config: dict) -> None:
    WHATSAPP_CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def send_whatsapp_alert(config: dict, message: str) -> tuple[bool, str]:
    """Send live direct alert message to the client's WhatsApp."""
    if not config or not config.get("enabled", True):
        return False, "WhatsApp alert dispatch is disabled in config"

    provider = config.get("provider", "callmebot").lower()
    phone = str(config.get("phone", "")).strip()

    if not phone:
        return False, "No recipient WhatsApp phone number configured"

    # 1. CallMeBot Provider (Free instant direct WhatsApp)
    if provider == "callmebot":
        api_key = config.get("api_key", "").strip()
        if not api_key:
            return False, "CallMeBot API key missing"
        # Ensure phone format (digits only or with +)
        clean_phone = phone if phone.startswith("+") else f"+{phone}"
        encoded_text = urllib.parse.quote(message)
        url = f"https://api.callmebot.com/whatsapp.php?phone={urllib.parse.quote(clean_phone)}&text={encoded_text}&apikey={api_key}"
        try:
            resp = requests.get(url, timeout=18)
            if resp.status_code == 200:
                return True, "WhatsApp alert delivered via CallMeBot"
            return False, f"CallMeBot returned status {resp.status_code}: {resp.text[:100]}"
        except Exception as exc:
            return False, f"WhatsApp network error: {exc}"

    # 2. Twilio WhatsApp Provider
    elif provider == "twilio":
        sid = config.get("account_sid", "").strip()
        token = config.get("auth_token", "").strip()
        from_wa = config.get("from_whatsapp", "+14155238886").strip()
        if not sid or not token:
            return False, "Twilio account_sid or auth_token missing"
        to_wa = phone if phone.startswith("whatsapp:") else f"whatsapp:{phone}"
        from_num = from_wa if from_wa.startswith("whatsapp:") else f"whatsapp:{from_wa}"
        url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
        try:
            resp = requests.post(
                url,
                auth=(sid, token),
                data={"From": from_num, "To": to_wa, "Body": message},
                timeout=18,
            )
            payload = resp.json()
            if resp.status_code in (200, 201) and not payload.get("error_code"):
                return True, "WhatsApp alert delivered via Twilio"
            return False, payload.get("message", f"Twilio error {resp.status_code}")
        except Exception as exc:
            return False, f"Twilio network error: {exc}"

    return False, f"Unknown WhatsApp provider: {provider}"

