GYANOVATE VISIONAI – USB CAMERA + AI + PHONE TEXT ALERT
=====================================================

WHAT THIS DEMO DOES
-------------------
1. Captures live video from a USB camera connected to your Windows PC.
2. Runs a pretrained YOLO AI model locally on the PC to detect people.
3. Uses a simple demo rule: send an alert when a detected person's bounding-box
   centre enters the red rectangle (the simulated restricted zone).
4. Sends a text message to your phone through a Telegram bot.
5. Writes each new zone-entry event to incident_log.csv.

IMPORTANT: This is a proof-of-concept, not the complete Gyanovate product.
The built-in YOLO model detects people, not helmets or every custom event.
Video is processed locally and is not uploaded by this demo. Telegram receives
only the text alert. Internet is needed for Telegram alerts and first-time
package/model downloads. A phone notification needs Telegram installed.

REQUIREMENTS
------------
- Windows 10 or 11 PC
- External USB webcam/camera
- Python 3.10–3.14 (64-bit), with "Add Python to PATH" selected
- Internet for installation, model download, and Telegram messages
- Telegram account on your phone

STEP A – INSTALL
----------------
1. Extract this ZIP into a normal folder, for example Desktop\VisionAI_Demo.
2. Connect the USB camera. Close the Windows Camera app and other apps using it.
3. Double-click install_windows.bat and wait for installation to complete.

STEP B – CONNECT TELEGRAM TO YOUR PHONE (ONE TIME)
-------------------------------------------------
1. On your phone, open Telegram and search for the official bot account
   @BotFather.
2. Open BotFather and send: /newbot
3. Follow its instructions to choose a bot name and username.
4. BotFather gives you a BOT TOKEN. Keep it private. Never send it to anyone.
5. In Telegram, open your newly-created bot using its username and press Start.
   Send /start in that chat.
6. On your PC, double-click setup_telegram.bat.
7. Paste the bot token when prompted (the input is hidden). The helper finds
   your private chat ID, saves telegram_config.json, and sends a test message.
8. Confirm the test message arrives on your phone.

SECURITY NOTE: telegram_config.json contains the bot token. Keep it private.
Do not share the file, screenshots of it, or your bot token. If it is exposed,
revoke/regenerate the token with @BotFather.

STEP C – RUN LIVE AI CAMERA
---------------------------
1. Double-click run_camera_0.bat.
2. If the wrong camera opens or no camera opens, close the program and try
   run_camera_1.bat. Camera index 2 can be tested from Command Prompt with:
      python visionai_camera_demo.py --camera 2
3. Wait for the model to load. First run downloads yolo26n.pt.
4. Stand in front of the camera and move into the red rectangle.
5. On the first detected entry, you should receive one Telegram text alert.
6. Move out of the rectangle. After it has been clear for 3 seconds, the rule
   is re-armed and can send another alert on the next entry.
7. Press Q in the camera window to stop.

THE DEMO RULE
-------------
IF AI detects a person AND the centre of that person's detection box is inside
RESTRICTED ZONE, THEN send a Telegram text alert.

The rectangle is a simple example zone. It can be customised later. Detection
may be affected by lighting, camera angle, distance, occlusion, and PC speed.
Do not use this demo as the only safety/security measure in a real workplace.

TROUBLESHOOTING
---------------
- Camera not opening: try camera index 1 or 2; close other camera applications;
  check Windows Settings > Privacy & security > Camera permissions.
- No Telegram message: ensure you pressed Start and sent /start to your bot;
  run setup_telegram.bat again and check the test message; check PC internet.
- AI model download fails: check internet and rerun the demo.
- Slow detection: close other programs; this starter version uses CPU inference.
- Want local detection only: run
      python visionai_camera_demo.py --camera 0 --no-telegram

CURRENT CONDITION / LIMITATION
------------------------------
The default trigger is PERSON ENTERS RESTRICTED ZONE. This is intentionally a
simple, generic condition for the first end-to-end test. Helmet missing,
smoking, fall detection, or other events need a suitable trained/validated AI
model and a matching rule. A generic person detector cannot reliably infer
those conditions.
