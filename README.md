# VisionAI Sentinel

A local Windows camera-monitoring dashboard with multi-object detection, evidence-based safety screening, searchable incident clips, uploaded-video analysis, and quiet Telegram delivery.

**Suggested GitHub repository name:** `visionai-sentinel`

**GitHub About description:**

> AI-assisted camera monitoring dashboard with multi-object detection, 12-second event confirmation, incident playback, uploaded-video screening, and Telegram violation-clip batches every three minutes.

## Status and intended use

This project is a working local pilot, not a certified safety system or a validated commercial surveillance appliance. AI outputs require human review. Do not rely on it as the sole emergency response mechanism, for automatic disciplinary decisions, or for legal traffic enforcement.

The current alert policy applies to the dashboard launched through `app.py`, `start_app.bat`, or `run_web_dashboard.bat`. Older standalone camera-demo scripts in an existing working folder have separate behavior and are not included in the clean dashboard source package.

## Features

- Live USB-camera preview with individual coloured boxes for supported simultaneous object classes.
- Primary YOLO detection, optional YOLO-World search, and conservative matching against a privately supplied labelled photo library.
- Object snapshots, object search, and playback links only when an associated clip is available.
- Local safety radar, visible warning messages, spoken warnings, and siren controls.
- Continuous-evidence confirmation for eligible live-camera events instead of immediate single-frame violations.
- Independent incident recordings for different event titles; encoding runs outside the camera inference loop.
- Browser incident playback and searchable local audit records.
- Uploaded-video screening for potential helmet issues, plate OCR, red-light crossings, and estimated vehicle speed.
- A dashboard assistant for supported local commands and optional Telegram command interaction.
- Optional WhatsApp integration; credential values remain masked in its status response.

## Alert delivery policy

```text
Camera observations
    -> local CHECKING status while evidence accumulates
    -> confirmed screening condition + local warning/siren
    -> save the incident clip
    -> persistent pending outbox
    -> next three-minute Telegram batch: each pending clip + its caption
```

- Automatic camera alerts do not send immediate Telegram text messages or immediate clips, including possible falls and raised-arm interactions.
- Every three minutes, the application attempts delivery of all pending violation clips, each with its own event caption. This is a batch of individual videos, not one combined video.
- The scheduler runs while the application is open, including when the camera is stopped. Uploaded-video violation clips use the same queue.
- Empty batches send nothing. A successfully acknowledged clip is removed from the pending queue and is not automatically resent.
- Failures remain pending for a later three-minute cycle. The outbox and acknowledged filenames are stored locally in `telegram_outbox.json` and survive application restarts. A restarted application begins a new three-minute countdown.
- Clips finished during an active batch wait for a later batch. Sending many videos can take longer than three minutes; batches do not overlap.
- A network timeout can occur after Telegram accepts a video but before acknowledgement reaches the application. A retry can then duplicate that video; Telegram does not provide an exactly-once guarantee here.
- **Send Pending Batch Now** is an explicit manual override. Interactive bot replies and manually requested actions are separate from automatic violation alerts.
- Informational arrivals, vehicle-bay turnaround, normal speed readings, and plate-only readings do not create automatic Telegram violation messages.

Warnings remain local without Telegram configuration; completed incident clips remain available on disk.

## Live-camera confirmation

| Screening condition | Evidence window | Meaning and limitation |
| --- | --- | --- |
| Demo perimeter occupancy | 12 seconds | A matched person remains in the demo zone; actual hazards require site configuration. |
| Unoccupied desk with queue presence | 12 seconds | No person is seen in the demo desk zone while a matched person remains in the queue zone. |
| Phone presence at desk | 12 seconds | A detected phone remains inside a person's torso region at the demo desk; this does not prove browsing or addiction. |
| Queue congestion | 12 seconds | At least two people remain in the demo queue corridor. |
| Unattended floor object | 12 seconds | A detected object remains near floor level without a nearby detected person; this does not identify a spill or rubbish. |
| Possible fall | 12 seconds | Visible upright-to-horizontal pose transition followed by sustained matched-person horizontal evidence. |
| Raised-arm interaction | 12 seconds | Sustained geometry involving the same locally tracked actor and target; aggressive intent is not determined. |
| Frame-edge dwell | 15 seconds | A matched person remains near the frame edge; nighttime or suspicious intent is not inferred. |

Missing or negative evidence, changed spatial subjects, out-of-order timestamps, and observation gaps over 2.5 seconds reset confirmation. Matching is geometric, not biometric identity recognition; crossing or overlapping people can still cause tracking errors.

Colour-based vest checks, bounding-box posture checks, rapid nearby movement, and inactivity remain local advisory checks rather than proof of violations. A horizontal pose does not diagnose sleeping; the pose detector does not issue a second sleeping alert for a fall episode.

Continuous eligible episodes are latched to reduce repeat alerts. New episodes also respect the existing per-title cooldown. Ordinary object boxes do not wait 12 seconds: event confirmation and object recognition are different operations.

Live incident capture retains up to 12 seconds of preceding evidence plus 3 seconds of follow-up. Playback uses actual timestamps; startup, stopped-camera, or sparse recordings may be shorter. Frames are not repeated to invent a duration. Uploaded-video events retain their separate video-analysis evidence logic; they are not forced to stay true for 12 seconds, which would invalidate brief crossings.

## Requirements

- Windows 64-bit, a working camera, and permission for Python to access it.
- Python 3.10 or newer, subject to availability of compatible PyTorch/OpenCV dependencies.
- A speaker for audible warnings; a modern browser for the dashboard.
- Internet access for initial dependency/model installation and optional Telegram/WhatsApp delivery.
- Sufficient RAM, disk space, and CPU/GPU resources for the selected models. FPS is hardware- and scene-dependent.

The existing installation was tested with Python 3.14.5, Ultralytics 8.4.208, OpenCV 5.0.0.93, PyTorch 2.15.0, torchvision 0.29.2, NumPy 2.5.4, requests 2.34.2, and EasyOCR 1.7.2. This is a tested environment snapshot, not a dependency lockfile or a clean-install certification. CLIP/PyTorch emit TorchScript compatibility warnings on Python 3.14; validate your chosen Python and package versions before deployment.

## Quick start

Open a terminal in the extracted project folder:

```bat
python -m pip install -r requirements.txt
python download_models.py
python app.py
```

Alternatively, use `install_windows.bat`, then `start_app.bat`. The dashboard opens at `http://localhost:8080` by default. Select the correct camera in the dashboard and start it; allow model warm-up before judging reference-object readiness.

For an isolated environment, create and activate a virtual environment before installing and launching from the same terminal. The existing batch launchers use Python from PATH; they do not automatically activate a virtual environment.

Optional settings in a Windows command prompt:

```bat
set VISIONAI_PORT=8080
set VISIONAI_NO_BROWSER=1
python app.py
```

The server tries another port if the requested one is occupied; check the startup output for the actual address. Use `create_desktop_shortcut.bat` only after installation and checking the launcher uses your intended Python environment.

## Models and private references

Model binaries and private images are deliberately excluded from this repository. After cloning, fetch every weight file from its official source with:

```bat
python download_models.py
```

| Local path | Size (approx.) | Purpose |
| --- | --- | --- |
| `yolo26n.pt` | 5 MB | Primary object detection |
| `yolo11n.pt` | 5 MB | Uploaded-video detection |
| `yolo26n-pose.pt` | 8 MB | Optional possible-fall and raised-arm pose screening |
| `yolov8s-worldv2.pt` | 25 MB | Optional open-vocabulary object detection and helmet screening |
| `weights/clip/ViT-B-32.pt` | 338 MB | Optional private photo-reference matching |
| `weights/license_plate_yolo11n.pt` | 5 MB | Uploaded-video plate detection (required by that feature) |

Ultralytics models are obtained through the Ultralytics library's official asset download; the CLIP weights through the `clip` package; the plate detector from its pinned public source. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and licensing. The script skips files that already exist; use `python download_models.py --force` to re-download.

Some Ultralytics model paths may also trigger automatic downloads when missing at startup. Missing optional models or references reduce capabilities; inspect readiness indicators rather than assuming every detector is active.

Private reference metadata belongs in `reference_objects/references.json`. Its entries contain `file`, `label`, and `display_name`, referring to authorised images inside `reference_objects/`. Use accurate, reviewed labels and test against visually similar objects. The supplied user's reference library is not distributed with the repository.

## Uploading to GitHub

The repository ships without model binaries, so no individual file approaches GitHub's upload limits. The largest tracked source file is the dashboard module (`app.py`).

- Use the Git command line (`git add`, `git commit`, `git push`) or upload the extracted clean archive contents. The included `.gitignore` excludes `weights/`, `*.pt`, footage, snapshots, references, credentials, and runtime files.
- **Do not drag the live working folder into GitHub's web uploader.** The web uploader ignores `.gitignore`, warns about files over 25 MB, and rejects files over 100 MB — the local CLIP weights are about 338 MB and will always be refused. Upload only the repository files.
- If a file is ever reported as too large, check `git status` output for untracked weights, videos, or archives before pushing. `git lfs` is not used by this project.

## Telegram and WhatsApp setup

```bat
python setup_telegram.py
```

Supply your own bot token and destination through the local setup flow. Start the bot conversation first. Never commit or paste the resulting configuration into an issue, screenshot, log, or README.

Optional WhatsApp setup:

```bat
python setup_whatsapp.py
```

External providers may require separate account configuration. Automatic Telegram batching does not enable automatic WhatsApp delivery.

## Source layout

```text
app.py                         Dashboard, HTTP routes, rule engine, recordings, delivery outbox
vision_detection.py            Primary detection and stabilisation
micro_object_ai.py             Optional open-vocabulary and private-reference matching
emergency_action_detector.py   Pose evidence tracking and confirmation
object_tracker.py              Object snapshots, search index, associated recordings
uploaded_video_analysis.py     Historical-video screening
telegram_listener.py           Telegram commands and acknowledged video delivery
whatsapp_manager.py            Optional WhatsApp integration
speaker_alarm.py               Local siren and speech controls
setup_telegram.py              Local Telegram setup
setup_whatsapp.py              Local WhatsApp setup
requirements.txt               Dependency requirements
download_models.py             Fetches all model weights from official sources
test_alert_policy.py            Isolated alert-policy regression tests
THIRD_PARTY_NOTICES.md         Model attribution and licensing notes
```

The clean source archive includes current dashboard modules, setup/launch scripts, branding assets, this README, the model downloader, third-party notices, tests, and ignore rules. It does not contain live configuration, incident footage, object indexes, private references, model binaries, or the private Bengali notes.

## Tests and verification

```bat
python -B -m unittest discover -s . -p "test_alert_policy.py"
```

The 32 tests isolate current definitions without importing the running application, loading model weights, reading live credentials, sounding speakers, or contacting Telegram. They cover batch timing, deduplication, retry timing, restart persistence, uploaded-event queues, API acknowledgements, event thresholds, evidence gaps, subject changes, conservative pose confirmation, and real-duration capture metadata.

Browser checks additionally covered batch status/countdown, warning wording, queued-to-sent uploaded-event presentation, delivery-status polling, navigation, and console errors. A native 15-second H.264 encode/decode check passed on the local Windows installation.

These checks do not establish physical-camera accuracy for every scenario, a live Telegram delivery guarantee, or commercial acceptance. Perform supervised on-site tests with representative positives and negatives before deployment.

## Privacy and deployment checklist

- Obtain the required consent and legal authority for camera monitoring and external video transmission.
- Keep tokens, WhatsApp credentials, footage, snapshots, reference photos, the outbox, and private notes outside public repositories. The included `.gitignore` excludes known local private/runtime files.
- Ignore rules do not remove files already committed to Git history. If credentials were previously uploaded, revoke/rotate them and address repository history separately.
- The built-in HTTP server binds to all interfaces and does not provide a production authentication/TLS/access-control layer. **Do not expose it directly to the internet or an untrusted network.** Use a controlled workstation or an appropriately secured gateway.
- Establish retention/deletion, backup, operator access, and disk-capacity policies. Recordings and the delivery journal are not automatically retention-managed.
- Calibrate site zones, camera placement, lighting, occlusion tolerance, and acceptable alert thresholds. Current zone geometry is a demo configuration, not a discovered map of machinery or workstations.
- Measure precision, recall, missed incidents, false alerts per hour, latency, and FPS with labelled site recordings. A 12-second delay can reduce transient false positives but can miss brief incidents or delay assistance.
- Uploaded traffic speed, stop-line, helmet, and OCR results are uncalibrated screening estimates, not certified enforcement measurements.
- Review project and dependency licensing before sale or redistribution, including Ultralytics' AGPL/commercial terms and rights to model weights, logos, and reference images. No project-wide commercial licensing grant is added by this README.
