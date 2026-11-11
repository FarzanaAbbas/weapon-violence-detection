# Weapon Identification and Violence Detection

A web app that watches a live camera or an uploaded video/photo and raises an alert when it sees a **weapon** (knife, pistol, rifle, grenade) or **violence** (fighting, a **neck hold** or a **kick**).

When something is detected, the app plays a siren, shows a popup with the captured frame, saves the incident and can send an email with the image.

## What it uses

| Part | Model |
| --- | --- |
| Weapon detection | YOLOv8 trained on 5 classes: Grenade, Knife, Missile, Pistol, Rifle (`models/best.pt`) |
| Violence detection | ResNet50 + LSTM over 20 frames (`models/violence_lstm_model.h5`) |
| Neck hold and kick | YOLOv8 pose model (`models/yolov8n-pose.pt`) with rules in `pose_actions.py` |

Built with Python, Flask, OpenCV, TensorFlow, Ultralytics YOLOv8 and SQLite.

## Features

- **Live Monitor** – uses the webcam and shows boxes, alert status and the scene (isolated / moderate / crowded) in real time.
- **Upload & Analyze** – checks a video or photo. The video pauses at each threat with a siren and popup; **Resume video** continues it.
- **Proper labels** – alerts name the weapon: *KNIFE DETECTED*, *PISTOL DETECTED*, *RIFLE DETECTED*.
- **Crowded scenes** – counts people and searches harder for small weapons when the scene is crowded.
- **Incident log** – every alert is saved with its frame. Mark it as *Verified* or *False positive*, and download all incidents as CSV.
- **Email alerts** – sends the alert image by email (optional).
- **Phone viewer** – watch the live feed on a phone on the same Wi-Fi by scanning the QR code.

## Setup

1. Install **Python 3.9 or later**.
2. Clone the project and install the libraries:

```
git clone https://github.com/<your-username>/weapon-violence-detection.git
cd weapon-violence-detection
pip install -r requirements.txt
```

3. Run the app:

```
python app.py
```

4. Open **http://127.0.0.1:5000** in Chrome or Edge.

## Email alerts (optional)

1. Copy `data/email_config.example.json` to `data/email_config.json`.
2. Fill in the sender Gmail address, a Gmail **app password** (Google Account → Security → App passwords) and the receiver address.
3. Test it by opening **http://127.0.0.1:5000/api/test_email**.

`data/email_config.json` is ignored by git, so the password never goes to GitHub.

## Project structure

```
app.py                 Flask server, weapon + violence detection, alerts, incidents, email
pose_actions.py        Neck hold, kick and scene (isolated / crowded) rules
models/                Trained models (weapon, violence, pose)
templates/             Web pages: home, live monitor, upload, phone viewer, alert popup
static/style.css       Styling
data/                  Created when the app runs: incident database, saved frames, email settings
uploads/               Created when the app runs: uploaded and annotated videos
```

## Notes

- Runs on a normal laptop CPU. A GPU or a laptop on charger in "Best performance" mode gives more frames per second and faster alerts.
- Detection quality depends on the camera angle, lighting and how clearly the weapon is visible.
