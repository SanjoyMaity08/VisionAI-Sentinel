"""Gyanovate VisionAI - Interactive WhatsApp Alert Activation Helper.

Helps you connect WhatsApp alerts using CallMeBot (instant & free).
1. Add phone number with country code (e.g. +91 9876543210).
2. Follow the 1-step activation to get your free CallMeBot API key.
3. Sends a live test alert to verify delivery.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import urllib.parse
import requests

CONFIG_FILE = Path(__file__).with_name("whatsapp_config.json")


def main() -> int:
    print("=" * 65)
    print("   VISIONAI SENTINEL - WHATSAPP ALERT ACTIVATION HELPER")
    print("=" * 65)
    print("\nHow to get your FREE CallMeBot WhatsApp API Key (takes 30 seconds):")
    print("1. Add +34 644 44 49 46 (or +34 644 10 55 84) to your phone contacts as 'CallMeBot'.")
    print("2. Send this exact WhatsApp message to that number:")
    print("      I allow callmebot to send me messages")
    print("3. You will immediately receive a reply with your API Key (e.g. 123456).\n")
    print("-" * 65)

    phone = input("Enter your WhatsApp Phone Number with country code (e.g. +919876543210): ").strip()
    if not phone:
        print("Error: Phone number is required.")
        return 1
    if not phone.startswith("+"):
        phone = "+" + phone

    api_key = input("Enter your CallMeBot API Key: ").strip()
    if not api_key:
        print("Error: API Key is required.")
        return 1

    print(f"\nSending test alert to {phone}...")
    test_msg = "VISIONAI SENTINEL - LIVE WHATSAPP ACTIVATION SUCCESSFUL.\nYour surveillance system is now connected to WhatsApp."
    url = f"https://api.callmebot.com/whatsapp.php?phone={urllib.parse.quote(phone)}&text={urllib.parse.quote(test_msg)}&apikey={api_key}"

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            config = {
                "enabled": True,
                "provider": "callmebot",
                "phone": phone,
                "api_key": api_key,
                "account_sid": "",
                "auth_token": "",
                "from_whatsapp": "+14155238886",
            }
            CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
            print("\n[SUCCESS] Test WhatsApp message delivered successfully!")
            print(f"Configuration saved to: {CONFIG_FILE.name}")
            return 0
        else:
            print(f"\n[FAILED] Server returned status {resp.status_code}: {resp.text}")
            print("Please make sure you sent 'I allow callmebot to send me messages' first.")
            return 1
    except Exception as exc:
        print(f"\n[NETWORK ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

