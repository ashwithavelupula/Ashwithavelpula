# Ashwithavelpula
NEXUS OFFLINE ASSISTANT - CLIENT SETUP GUIDE
============================================

WHAT THIS IS
------------
A private AI assistant that runs 100% on your own computer:
chat, object detection in photos, face detection and live facial
emotion detection, image tools,
calculator/converter/notes/document Q&A, and voice mode.
No account needed. Your data stays on your machine.

WHAT YOU NEED
-------------
- Windows 10 or 11 (64-bit)
- About 3 GB free disk space
- Internet connection DURING SETUP ONLY
  (to download Python libraries + the AI and detection models once).
  After setup, everything except the optional LCD cloud
  sync works without internet.

HOW TO START (ONE CLICK)
------------------------
1. Copy the whole "NexusOffline" folder anywhere you like,
   e.g.  C:\NexusOffline   or   Desktop\NexusOffline
2. Right-click setup_and_run.bat, choose "Run as administrator",
   then double-click it. (Admin rights let it install the
   database service automatically.)
3. Wait. First run takes a while:
     - libraries:   ~5-10 minutes
     - database:    ~2 minutes (automatic)
     - AI model:    ~670 MB download (resumes if it breaks)
     - mobile detector: ~35 MB download
4. Your browser opens automatically at:
     http://127.0.0.1:5000
5. Log in with any name (for example: guest) and use it.

DAY-TO-DAY USE
--------------
- Just double-click  run.bat  (fast start, no reinstalling).
- Keep the black window OPEN while using the app.
- Press CTRL+C in that window to stop the app.

FEATURES INCLUDED
-----------------
Chat, Objects, Faces, live facial emotions (seven categories,
smoothed per face in the browser), Register Face, Image Tools,
Docs Q&A, Notes Vault, Smart Tools, Voice, History,
LCD cloud sync toggle (see below).

LIVE CAMERA EMOTION DETECTION
-----------------------------
The live camera panel reuses its existing camera stream. Face boxes
and seven-way expression scores are computed in the browser with
the locally bundled face-api.js runtime and model files in
static\vendor and static\models\face-api. No camera images are sent
to a third-party service. The existing object-analysis request still
goes to this computer's local Flask server. Emotion inference works
offline after setup and can be switched off without stopping the
camera or object analysis. The face-api.js MIT license is included
in static\vendor\face-api.LICENSE.

LIVE MOBILE PHONE DETECTION
---------------------------
Live object detection combines the existing Pascal VOC model with
the OpenCV Zoo YOLOX COCO model for mobile phones. Both models run
locally through OpenCV; camera frames are not sent to third parties.
The YOLOX model and its Apache-2.0 license are included in models.
Small, distant, obscured, or unusual phone views can still be missed;
the live panel shows detection confidence and can speak "Mobile detected."

MYSQL DATABASE (AUTOMATIC)
--------------------------
Handled by setup_and_run.bat, no manual work needed:
- If this machine already has MySQL/MariaDB, the setup
  reuses it. It will ask once for the root password, then
  creates the "chatbot_db" database and all tables in it.
- If nothing is installed, the setup installs a portable
  database service itself (named NexusDB) with a random
  password saved locally in the ".env" file.
  (Run the setup as administrator for this step.)
Without any database the app still runs in offline mode
(history is kept in the left sidebar instead).

OPTIONAL: LCD / FIREBASE SYNC
-----------------------------
Typed chat questions and full answers are permanently written
to "nexus_emotion_history" with Firebase push IDs as keys.
They include "type": "nexus_history" and "source": "nexus_history".
The separate "nexus_live" node contains only the latest
short answer for an LCD and is deleted after 20 seconds.
Manual LCD resend updates only "nexus_live".
Voice-mode turns are tagged with "mode": "voice" in
"nexus_emotion_history" as records with "type": "voice". The temporary "nexus_live"
payload includes the voice mode and prefixes the LCD lines with
"VOICE:" so the display can identify voice replies.

Firebase sync needs internet and a service account with
Realtime Database access. Set FIREBASE_URL and
FIREBASE_CREDENTIALS in ".env", then restart run.bat.
The RTDB status is available at:
  http://127.0.0.1:5000/api/rtdb-status
It reports the history node and the last archive result.
Voice-history write status is also available there.

TROUBLESHOOTING
---------------
- "Python not found" and auto-install fails:
  install Python 3.11 from python.org, TICK "Add python.exe
  to PATH", then run setup_and_run.bat again.
- Browser says "can't reach page":
  the black setup window must stay open; wait 60 seconds
  on first start (the AI model loads slowly), then refresh.
- "Port 5000 in use":
  close the other program using it, or ask support.
- Antivirus blocks something:
  allow Python through when asked; all processing is local.
- Model download interrupted:
  just run setup_and_run.bat again, it resumes.
- Microphone button does nothing:
  use Chrome or Edge and allow microphone access.
  (Browser voice transcription itself needs internet.)

SUPPORT
-------
Send back: a screenshot + the text from the black window.
