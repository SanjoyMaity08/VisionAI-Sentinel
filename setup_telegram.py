"""One-time Telegram setup for the USB camera demo.

Create a bot with @BotFather, open the bot on your phone and press Start,
then run this helper. It discovers your private chat ID and sends a test message.
"""
from __future__ import annotations
import getpass
import json
from pathlib import Path
import sys
import requests

CONFIG_FILE = Path(__file__).with_name("telegram_config.json")
API = "https://api.telegram.org/bot{token}/{method}"


def api_call(token: str, method: str, **kwargs):
    response = requests.post(API.format(token=token, method=method), data=kwargs, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise RuntimeError(payload.get("description", "Telegram API returned an error"))
    return payload["result"]


def main() -> int:
    print("=== VISIONAI SENTINEL: TELEGRAM SETUP ===")
    print("Create a bot using @BotFather, then open your new bot on your phone and press Start.")
    token = getpass.getpass("Paste your bot token (input is hidden): ").strip()
    if not token:
        print("No token entered.")
        return 1
    try:
        bot = api_call(token, "getMe")
        print(f"Connected to bot: @{bot.get('username', 'unknown')}")
        updates = api_call(token, "getUpdates", timeout=10, limit=100)
        chats = {}
        for update in updates:
            message = update.get("message") or update.get("edited_message")
            if not message:
                continue
            chat = message.get("chat", {})
            sender = message.get("from", {})
            if chat.get("type") == "private" and not sender.get("is_bot", False):
                chats[str(chat.get("id"))] = chat.get("first_name") or chat.get("username") or "Your phone"
        if not chats:
            print("I could not find a private chat yet.")
            print("On your phone: open your bot, press Start, and send /start.")
            print("Then run setup_telegram.bat again.")
            return 2
        if len(chats) > 1:
            print("More than one private chat was found. Choose your chat:")
            chat_ids = list(chats)
            for i, chat_id in enumerate(chat_ids, 1):
                print(f"{i}. {chats[chat_id]} (chat ID {chat_id})")
            choice = input(f"Choose 1-{len(chat_ids)}: ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(chat_ids)):
                print("Invalid selection.")
                return 1
            chat_id = chat_ids[int(choice)-1]
        else:
            chat_id = next(iter(chats))
        CONFIG_FILE.write_text(json.dumps({"bot_token": token, "chat_id": chat_id}, indent=2), encoding="utf-8")
        api_call(token, "sendMessage", chat_id=chat_id,
                 text="VisionAI Sentinel connected. Your phone can receive camera event alerts.")
        print("\nSuccess! A test message has been sent to your Telegram.")
        print(f"Configuration saved to: {CONFIG_FILE.name}")
        print("Keep this file private. It contains your bot token; do not share it or upload it.")
        return 0
    except requests.RequestException as exc:
        print(f"Network error: {exc}")
        print("Check your internet connection and try again.")
        return 1
    except Exception as exc:
        print(f"Setup failed: {exc}")
        print("Check the bot token, and make sure you pressed Start in the bot chat.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
