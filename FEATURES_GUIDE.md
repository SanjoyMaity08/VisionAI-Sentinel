# Gyanovate VisionAI - CCMS Enterprise Command Center

All CCMS controls, live surveillance feeds, two-way AI communication, and footage reviews are available in one enterprise application.

The command center uses a responsive, Zoho-inspired light enterprise interface with a white navigation bar, blue action controls, clear status panels, and a dark high-contrast video viewport. The dashboard includes surveillance, object search, safety radar, and incident review.

This project contains the CCMS dashboard only. Camera startup is explicit; opening or refreshing the dashboard does not start surveillance.

---

## Access & Operation

1. **Launch the Application:**  
   Double-click **`start_app.bat`** (or `run_web_dashboard.bat`).  
   The CCMS-only application opens its dashboard in your browser, normally at  
   `http://127.0.0.1:8080` (or `http://<your-ip>:8080` across your local network). If another application is already using that port, it selects the next available port and prints the actual address in the console.

2. **Control Directly from the UI Screen:**
   * **Select Camera:** Choose **Camera 0 (Primary / Built-in)** or **Camera 1 (External USB Camera)** from the dropdown.
   * **Start Camera:** Click the blue **START CAMERA** button.
   * **Stop Camera:** Click the **STOP CAMERA** button at any time to release the camera hardware cleanly.
   * **Automatic violation alerts:** Each newly active CCMS violation is announced through the speaker and sent to the configured Telegram chat. Fall and possible-aggression emergency alerts use the critical siren-and-voice alert path. Continuous detections are not repeated every video frame.
   * **Cycle countdown:** The 3-minute countdown starts only after the camera is explicitly started, pauses and resets to `03:00` when stopped, and remains idle on dashboard load. Opening or refreshing the dashboard does not start the camera.
   * **EMERGENCY SIREN:** Click the dedicated red **EMERGENCY SIREN** button to trigger the authentic high-decibel Tornado air-raid wail siren through on-site speakers.
   * **Camera Emergency Alerts (Fall / Possible Aggression):**
     - With the camera running, the app performs a low-rate pose scan in a background worker. On first use, Ultralytics may download `yolo26n-pose.pt`; pose inference runs on CPU and can lower overall camera performance.
     - A possible fall alert requires a tracked standing-to-horizontal body transition; the existing persistent floor-level person-box check remains as a secondary fall signal. An alert activates a six-second local siren and sends an immediate text alert to the Telegram chat configured in `telegram_config.json`.
     - A possible aggression alert requires a rapid, extended arm movement ending in a nearby person's upper-body region. Ordinary close proximity or fast movement alone does not trigger this emergency alert.
     - These are AI-generated possible-event alerts, not confirmed violence, injury, or medical diagnoses. The system cannot determine a person's age or intent. Verify the scene immediately; detection and Telegram delivery are not guaranteed.
     - Alerts are rate-limited to one per event type per 60 seconds to avoid repeated sirens/messages during a continuous event. A missing or unavailable pose model is shown in the Conduct Watch radar rather than treated as a successful scan.
   * **AI Object Search & Last Known Location Video Retriever:**
     - Search objects that the installed detection models can identify (e.g. `bottle`, `camera`, `keys`, `bag`, `phone`, `laptop`).
     - Pen and pencil detections require a fine-grained model that recognizes those classes; shape-based guesses are disabled to prevent false alerts.
     - The camera requests 1280×720 and runs overlapping detail scans every three frames to improve detection of small objects; actual camera resolution depends on device support.
     - A supplemental YOLO-World detector checks an editable vocabulary of common items (including keyboard, computer mouse, pen, and accessories) using asynchronous overlapping scans. Install dependencies from `requirements.txt`; first initialization downloads the YOLO-World model and roughly 338 MB of CLIP text-encoder weights.
     - Open-vocabulary detection can reduce FPS on CPU-only systems and cannot guarantee every item; small objects, occlusion, lighting, focus, and camera resolution still affect accuracy.
     - Search results show the detector's model score, not a guarantee of correctness. A score of 100% cannot be guaranteed; uncertain detections are suppressed instead of presented as facts.
     - Primary detections require at least a 70% model score and stable boxes across frames; trains require 80%, and baseball bats require 88% to reduce false matches. The ambiguous primary `TV` class is suppressed. Open-vocabulary detections require at least 75% and four matching scans (computer monitors require at least 55% with the same four-scan confirmation). Old ambiguous TV and low-score object records/clips are quarantined from results without deleting footage.
     - Computer searches include separately named desktop, computer tower/case, and computer monitor labels where the model can confidently distinguish them. `CPU` is accepted as a search alias for a tower/case.
     - Built-in typo tolerance and fuzzy distance normalization automatically corrects misspelled searches like `bottel` to `bottle`.
     - When a verified object-specific recording exists, **PLAY LAST LOCATION CLIP** opens it in the embedded player; **OPEN VIDEO DIRECTLY** opens a new tab. If no clip is available, the play button is disabled and the popup explains that no recording exists instead of showing an empty player.
     - Live surveillance stream locks onto the active target in real-time with gold targeting reticles.
   * **Uploaded Video Analysis:**
     - Open **Video Analysis** in the CCMS sidebar, choose an MP4, MOV, MKV, AVI, WEBM, or M4V file up to 200 MB, preview it, and select **ANALYZE VIDEO**.
     - YOLO11 and ByteTrack track cars and motorcycles. A separate YOLO11 license-plate detector draws plate boxes, and EasyOCR provides an approximate plate reading. Plate text is fallible and must be checked against the annotated snapshot.
     - For speed estimates, pause or preview the video, draw Gate A and Gate B across the same vehicle path, and enter the real-world distance between the lines. The default alert limit is 60 km/h. The estimate assumes a fixed camera and accurate calibration; vehicles that do not cross both gates have no speed estimate and cannot trigger an overspeed alert.
     - Results include clip timestamps, annotated snapshots, plate OCR where readable, and calibrated speed estimates. A previously detected person absent from the frame for at least 10 seconds is also reported; person-absence events remain dashboard-only.
     - Stop live camera surveillance before starting an uploaded-video analysis. Only one uploaded clip is analyzed at a time. The server removes the temporary upload after analysis.
     - Only a measured speed above the configured limit sends a Telegram message and triggers a short speaker siren followed by a spoken alert. The message includes the uploaded filename, clip timestamp, estimated speed, limit, and OCR text (or “Unreadable”). Historical-video alerts are not live observations. Below-limit vehicles remain in the dashboard results without an alert. Existing live-camera alert behavior is unchanged.
     - This analysis does not issue official citations. Speed and plate readings can be wrong or missed; inspect the original footage before taking action. The plate model weights are attributed in `weights/LICENSE-PLATE-MODEL.txt`; its repository is MIT-licensed, but the upstream README references a training dataset whose licensing has not been independently verified.
     - The Ultralytics YOLO11 vehicle weights and EasyOCR recognition weights may need to download the first time analysis runs. Initial setup therefore requires an internet connection.
   * **Embedded 10-Second Video Player:**
     - Instant HTML5 video streaming with Range request support (HTTP 206 Partial Content).
     - Auto-fallback for browser autoplay restrictions.
     - Direct clip link for standalone review.

---

## Bidirectional Two-Way AI Communication via Telegram

The VisionAI Telegram Bot features full two-way AI communication:

1. **AI Object Search & Last Known Location Retrieval:**
   * Send the name or a question about any tracked object, for example:
     - `where is the bottle?` or `find bottle`
     - `bottel` (typo-tolerant)
     - `where is my bottle?` or `search camera`
     - `mouse`, `keyboard`, `keys`, `bag`, `laptop`, `phone`
   * **What the AI Bot does in response:**
     - Queries the active detection catalog and spatial quadrant mapping.
     - Replies with exact coordinates, quadrant location, confidence, and timestamp.
     - **Dispatches the high-resolution cropped snapshot photo** directly into your Telegram chat.
     - Sends a verified object-specific MP4 clip inline in Telegram. If its clip is still recording, the bot sends it when ready; the object must be detected by the running camera. It never substitutes footage of a different object.
     - Automatically updates the web dashboard to lock AI reticles onto the searched target in real-time.
      - Includes the latest saved safety incident report and its separate clip when one is available. Send `/report` to retrieve that report by itself.

2. **Emergency Message to Dashboard and Site Speaker:**
   * Send `/emergency <message>` (for example, `/emergency I fell and need assistance`), `SOS`, `help`, or `HELP: <message>`.
   * The alert message appears in the dashboard's emergency banner and is announced through the site speaker.
   * For immediate danger, contact local emergency responders directly; this feature is not a replacement for emergency services.

3. **Remote Emergency Siren Trigger:**
   * Send `alarm`, `siren`, or `/alarm` to sound the Tornado air-raid wail siren on site.
   * Send `stop`, `silence`, or `/stop` to silence the siren.

4. **Remote Voice Intercom:**
   * Send `/speak <your message>` to announce a voice warning through on-site USB speakers.

5. **Live System Telemetry & Dashboard Links:**
   * Send `status` or `dashboard` to receive real-time camera health, FPS, monitored zones, and direct clickable web dashboard URLs.

---

## Design Standards

* **100% Emoji-Free:** Dashboard controls use SVG icons and readable system typography.
* **Zoho-Inspired Enterprise Format:** Light gray workspace, white panels, blue primary actions (`#0F6FDE`), clear safety-status colors, and Gyanovate Corp Pvt Ltd branding. The camera and clip playback areas remain dark for video contrast.
* **CCMS Operations Dashboard:** The CCMS overview presents a persistent navigation rail, live camera view, operational KPIs, AI object results, incident clips, emergency controls, safety monitors, and the audit log in a responsive control-room layout. KPI figures are populated from the running camera, tracked-object catalog, current active violations, saved clips, and dashboard process uptime.
* **Strict 3-Minute Batch Schedule:** Automated violation reports hold 1 validated 10-second video clip every 3 minutes.
* **Dedicated CCMS AI Object Search:** Select **AI Object Search** from the CCMS navigation to open a full-width search view with the object catalog, quick filters, active target status, locations, model scores, snapshots, and available clips. Search results and playback use the same detection and clip workflow as the CCMS overview.
