from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import cv2
import numpy as np
import base64
import os
import time
import subprocess
import sqlite3
import threading
import csv
import io
import socket
import secrets
import json
import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import imageio_ffmpeg
from pose_actions import PoseActionDetector
import pose_actions

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = 'models'

IMG_SIZE = 128

# Number of frames required by the LSTM
SEQUENCE_LENGTH = 20

# ------------------------------------------------------------
# LIVE FEED SPEED
#
# Why live violence never fired before: the LSTM looks at 20 frames
# and was trained on normal-speed video (20 frames ~ 1-3 seconds of
# action). The old live loop captured every 250 ms AND only kept every
# 2nd frame, so its 20-frame window covered ~10+ seconds -- a punch was
# 1 frame out of 20 and the model scored it as nothing (that's where
# the old "punching only reaches ~0.012" came from).
#
# The live loop now keeps EVERY frame it receives and the browser
# sends the next frame as soon as the previous answer arrives, so the
# window covers ~2-3 seconds at the ~8-10 fps a laptop CPU manages.
# To afford that, YOLO runs on every LIVE_YOLO_EVERY_N-th frame only
# (its boxes are reused in between) and both TensorFlow models run as
# compiled graphs instead of model.predict() -- see the timings in
# LOAD MODELS below.
# ------------------------------------------------------------

# Run weapon detection on every Nth live frame. 2 keeps the weapon
# alert fast (3 confirmations = ~0.6-0.8 s) while halving YOLO cost.
LIVE_YOLO_EVERY_N = 2

# Width of the frame the browser sends. The video you see on screen
# is the raw webcam at full quality -- this is only what gets analysed.
# (Set in live.html, SEND_WIDTH; noted here so both stay in sync.)
LIVE_SEND_WIDTH = 480

# YOLO input size for the LIVE feed. 640 (the default) is overkill
# for a 480px webcam frame and is one of the two heaviest parts of
# each tick along with ResNet50. Dropping to 320 roughly halves YOLO's
# cost with only a small accuracy trade-off. Raise back to 640/480 if
# you have a GPU and want max accuracy instead of speed.
# Preserve more detail for small handheld objects such as knives.  This costs
# some FPS but is materially more reliable than the 320px live inference size.
LIVE_YOLO_IMGSZ = 480

# Weapon model input size for the live feed (the pose model keeps
# LIVE_YOLO_IMGSZ). 640, the size uploads are checked at: knives in the
# live feed scored too low at 480. About 25 ms more per weapon check on CPU.
LIVE_WEAPON_IMGSZ = 640

# SLOW LAPTOP (live). When frames arrive more than this many seconds
# apart (the laptop analyses fewer than ~3 frames a second), the weapon
# and pose models both run on EVERY frame instead of taking turns, and
# the extra-large CROWDED sizes are not used. Otherwise at ~1 frame/s
# each model looked only every 2 s: too few looks for the knife vote,
# the neck hold (1 s) or the kick (2 looks) to ever confirm.
LIVE_SLOW_FRAME_SECONDS = 0.35

# YOLO weapon confidence.
#
# IMPORTANT NOTE ON FALSE POSITIVES ("sharp object = weapon"):
# has_weapon_class() already only fires on classes whose name matches
# WEAPON_KEYWORDS below, so this is not a "any object triggers weapon"
# bug in the code. What's happening is that best.pt itself was trained
# on a dataset that doesn't contain enough hard-negative examples of
# other elongated / reflective / pointed objects (scissors, pens,
# remotes, cutlery, tools), so the model itself generalises "knife" to
# those objects. Raising the confidence threshold and requiring a
# longer consecutive-hit streak (below) reduces the problem but cannot
# fully fix it -- the durable fix is retraining/fine-tuning best.pt
# with hard negatives. See log_incident()/the /api/incidents/<id>/verify
# route: every alert an operator marks "false positive" gets its source
# frame saved to uploads/in5cident_frames/, which builds exactly that
# hard-negative dataset over time for a future retraining pass.
CONFIDENCE_THRESHOLD = 0.45

# Weapon must be detected this many consecutive
# detection checks before alerting
# A short confirmation removes one-frame noise without making the operator
# hold a real weapon in front of the camera for several seconds.
CONSECUTIVE_HITS_REQUIRED = 3

# ------------------------------------------------------------
# VIOLENCE ALERT CONDITIONS
#
# Measured on the clips in uploads/ with the real ResNet50 + LSTM:
#
#   non-violent (drill, guns, knife cutting, shadowboxing, kata,
#   soldiers, nature)  -> score ~0.0003, highest single window 0.16
#   real fights (boxing / MMA sparring, street fight, assault)
#                      -> score 0.9-1.0 for seconds at a time
#
# So this model is decisive: it is either near 0 or near 1. The old
# threshold of 0.001-0.003 sat right in the noise floor, which is why
# a slap or a touch could trip it. 0.5 sits in the empty gap between
# the two groups.
#
# A slap can still produce a brief spike, so the alert also needs the
# score to be high for VIOLENCE_MIN_SECONDS out of the last
# VIOLENCE_WINDOW_SECONDS (see ViolenceGate). "Out of the last N
# seconds" rather than "continuously" matters: real fights have short
# dips between exchanges and must not reset the count.
#
# Result on the test clips: all 4 clearly violent fights alert, all
# 11 non-violent clips stay clear. Two clips don't alert: the judo
# throw (the model scores it 0 -- sport throws aren't in its training
# data) and the assault clip, whose violence only starts in the last
# 0.6 s of footage.
#
# Tuning:
#   - Slap still alerts     -> raise VIOLENCE_MIN_SECONDS (e.g. 1.5)
#   - Missing real fights   -> lower VIOLENCE_MIN_SECONDS (e.g. 0.7)
#                              or VIOLENCE_PROB_THRESHOLD (e.g. 0.3)
# The [VIOLENCE GATE] debug line prints the score and the seconds of
# high score in the window on every check.
# ------------------------------------------------------------

VIOLENCE_PROB_THRESHOLD = 0.5

# Look-back window, in seconds (wall clock for live, video time for
# uploads).
VIOLENCE_WINDOW_SECONDS = 2.0

# How much of that window must be above the threshold before alerting.
VIOLENCE_MIN_SECONDS = 1.0

# ------------------------------------------------------------
# POSE-BASED VIOLENCE: NECK HOLD + LEG KICK
#
# A YOLOv8-pose model finds people and their joints; pose_actions.py
# raises "neck hold" when one person's hand is on another person's neck
# and "kick" when a leg swings out fast. Both count as VIOLENCE and are
# added on top of the LSTM score -- the LSTM path is unchanged.
# All the thresholds (hand-to-neck distance, kick speed, how long it must
# last) are at the top of pose_actions.py.
#
# If the pose weights can't be loaded the app still runs, just without
# these two cues.
# ------------------------------------------------------------

POSE_MODEL_FILE = 'yolov8n-pose.pt'

# Minimum person confidence from the pose model. Was 0.5: fighters who
# overlap or blur come out at 0.3-0.5 (the kicker in the kickboxing clip
# was 0.29-0.51 during the kick), so they were dropped before the kick
# check ever saw them. Duplicate detections of one person are merged in
# pose_actions.dedupe_people(), so the lower bar doesn't double-count.
POSE_PERSON_CONF = 0.30

# Run pose on every Nth live frame, on the frames weapon YOLO skips, so
# the two models share the load instead of piling onto the same tick.
LIVE_POSE_EVERY_N = 2

# Keep a confirmed pose alert on screen this long (live), like the LSTM
# alert above.
POSE_ALERT_HOLD_SECONDS = 4

# ------------------------------------------------------------
# SCENE: ISOLATED / MODERATE / CROWDED
#
# The pose model already finds every person in view, so the app counts
# them and labels the scene. The thresholds are in pose_actions.py
# (ISOLATED_MAX_PEOPLE = 2, CROWDED_MIN_PEOPLE = 6). The label is shown
# on the live / phone / upload pages, written on every alert banner and
# saved with every incident (and in the CSV report).
#
# In a CROWDED scene people and weapons are smaller in the frame, so
# detection switches to higher resolution there:
#   live    the browser is asked for bigger frames (LIVE_CROWD_SEND_WIDTH)
#           and weapon + pose YOLO run at LIVE_CROWD_YOLO_IMGSZ;
#   upload  weapon YOLO also runs on overlapping tiles of the full-
#           resolution frame, and pose runs at CROWD_POSE_IMGSZ.
# ISOLATED / MODERATE scenes keep the normal, faster settings. Pose goes
# up to the bigger size once CROWD_HINT_PEOPLE people are in view, so the
# small, distant people of a forming crowd are counted and the switch to
# CROWDED isn't missed.
#
# Everything here needs the pose model; without it the scene shows as
# UNKNOWN and detection runs with the normal settings.
# ------------------------------------------------------------

LIVE_CROWD_SEND_WIDTH = 800

# YOLO input size for CROWDED live frames -- the frame's own width, so
# nothing is shrunk away. Measured on a crowd clip with small weapons
# (800 px frames, 41 checks): imgsz 640 found a weapon in 2, 800 in 25,
# 2x2 tiles in 31 but at 266 ms per frame (too slow for live; uploads
# use the tiles). No false weapons at any size on a weapon-free crowd.
LIVE_CROWD_YOLO_IMGSZ = 800

# Tiles per side (2 -> 2 x 2) and how much neighbouring tiles overlap,
# so a weapon on a tile edge is still whole in one of them.
CROWD_TILE_GRID = 2
CROWD_TILE_OVERLAP = 0.20

# Two weapon boxes of the same class overlapping more than this (IoU)
# after tiling are the same weapon.
CROWD_TILE_NMS_IOU = 0.50

CROWD_POSE_IMGSZ = 960

# People in view at which pose already switches to the bigger input size
# (two below CROWDED_MIN_PEOPLE in pose_actions.py).
CROWD_HINT_PEOPLE = pose_actions.CROWDED_MIN_PEOPLE - 2

# Process every third uploaded frame.  The cached feature pipeline below keeps
# scans responsive while retaining enough coverage for sustained events.
FRAME_SKIP = 3

# ------------------------------------------------------------
# UPLOAD SPEED
#
# Why big uploads were slow: the annotated copy was written at the
# source resolution. On the 4K judo clip (431 frames) that meant ~25 s
# writing 4K frames plus 64 s (your log) for the 4K -> browser MP4
# conversion -- more than the detection itself. Detection only ever
# looks at a 640 px copy, so the saved video is now capped at
# OUTPUT_MAX_WIDTH: the same clip writes in ~2.5 s and converts in ~5 s.
#
# High frame-rate clips (60 / 120 fps) were also analysed 20-40 times
# per second of footage with a fixed FRAME_SKIP. Uploads now analyse
# about UPLOAD_CHECKS_PER_SECOND frames per second of footage, never
# more often than every FRAME_SKIP-th frame. 25-30 fps clips are
# unchanged (every 3rd frame); 60 fps -> every 6th, 120 fps -> every 12th.
# ------------------------------------------------------------

OUTPUT_MAX_WIDTH = 1280

UPLOAD_CHECKS_PER_SECOND = 10

# x264 speed preset for the browser MP4 (ffmpeg's default is 'medium').
FFMPEG_PRESET = 'veryfast'

# ------------------------------------------------------------
# PHONE VIEW (watch the laptop's live monitor from a phone)
#
# The laptop runs the camera + detection on /live as usual. A phone
# opens /watch and sees the same feed with boxes, alerts, the alarm
# and the incident list -- it does NOT use the phone's camera.
#
# Anyone other than the laptop itself needs an access key, because
# the server is now reachable from the network. The key is generated
# once and saved in data/viewer_key.txt; the /live page shows the
# full phone link (with key) and a QR code to scan.
# ------------------------------------------------------------

# Listen on all network interfaces so a phone on the same Wi-Fi can
# connect. Set to '127.0.0.1' to make the app laptop-only again.
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5000

VIEWER_KEY_PATH = os.path.join('data', 'viewer_key.txt')

# The laptop only prepares the annotated phone frame while a phone
# has asked for one within this many seconds -- no phone, no extra work.
VIEWER_ACTIVE_SECONDS = 5

# The phone shows "laptop camera offline" if no new frame for this long.
VIEWER_STALE_SECONDS = 3

VIEWER_JPEG_QUALITY = 70


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models, please wait...")

# ------------------------------------------------------------
# WEAPON MODEL
# ------------------------------------------------------------

weapon_model = YOLO(
    os.path.join(MODELS_DIR, 'best.pt')
)

WEAPON_KEYWORDS = (
    'gun',
    'pistol',
    'rifle',
    'knife',
    'weapon',
    'blade',
    'firearm',
    'grenade'
)

# ------------------------------------------------------------
# CLASSES THAT ARE NEVER A WEAPON
#
# best.pt's classes are Grenade, Knife, Missile, Pistol, Rifle. Of
# those, four now alert: Pistol, Rifle, Knife and Grenade. Only Missile
# is ignored.
#
# GRENADE -- WARNING. It alerts because you asked for it, but it is the
# least trustworthy class in the model. Across every clip and photo in
# this project it has never once fired on an actual grenade (there are
# none in the footage) and it fires readily on ordinary objects:
#
#   chef's-knife scene   Grenade up to 0.87, on 38 frames, median 0.82
#   phone photo          Grenade up to 0.87
#   shadowboxing clip    Grenade up to 0.76
#   knife-on-board clip  Grenade up to 0.73
#
# Its bar is therefore 0.90 -- just above every false one measured, so
# the phone and the chef's-knife scene stay clear. The honest caveat:
# with no real grenade to test against, nobody knows whether a real one
# reaches 0.90 either. If grenades aren't part of what this camera is
# watching for, the safer setting is to put 'grenade' back in
# IGNORED_CLASS_KEYWORDS below and remove its bar.
#
# MISSILE stays ignored: it fired on a scissors photo at 0.75 and a
# held knife at 0.71, and a missile is not a handheld threat a camera
# like this is ever going to see.
#
# 'scissor' is here for a different reason: scissors are an everyday
# object, not a threat. Note the current best.pt has no scissors class,
# so when scissors do raise an alert it's because the model called them
# "Knife" and this filter can't catch that. Fixing that means retraining
# with scissors images as background -- see retrain_with_negatives.py.
# ------------------------------------------------------------

IGNORED_CLASS_KEYWORDS = (
    'scissor',
    'shear',
    'missile'
)

# ------------------------------------------------------------
# EXTRA GUARDRAIL FOR GUN/PISTOL AND KNIFE/BLADE CLASSES
# ("phone shows as pistol", "pen shows as knife")
#
# best.pt is confidently misclassifying small handheld objects as
# weapons -- phones/remotes as pistols, pens as knives. This is a
# known failure mode of small weapon-detection datasets: the model
# has learned "elongated/dark object in a hand" rather than the
# actual weapon shape, because the training data didn't include
# enough negatives of ordinary handheld objects. It is NOT fixable
# in code the way the old "any object = weapon" bug was; the real
# fix is retraining best.pt with hard negatives of pens, phones,
# remotes, wallets, etc. In the meantime, two cheap heuristics cut
# down the obvious cases per weapon type:
#
#   1. Require a higher confidence for gun-like AND knife-like
#      classes than the general weapon bar.
#   2. Shape check:
#        - gun-like: reject boxes that are "portrait" (taller than
#          wide) -- a handgun viewed side-on is wider than tall; a
#          handheld phone is almost always the opposite.
#        - knife-like: reject boxes that are too "needle-thin"
#          relative to their length -- a knife blade has real width
#          to it; a pen or pencil is much thinner relative to its
#          length. This is measured as min(width,height) /
#          max(width,height) of the box, normalized so orientation
#          (held vertically or horizontally) doesn't matter.
#
# Use the "False positive" button in the live UI whenever this fires
# on a non-weapon -- it saves that exact frame to
# uploads/incident_frames/, which is exactly the hard-negative data
# needed for a real retrain. These thresholds are a starting point;
# tune KNIFE_LIKE_MIN_THICKNESS_RATIO up if pens still slip through,
# or down if it starts missing real knives on your camera setup.
# ------------------------------------------------------------

GUN_LIKE_KEYWORDS = ('gun', 'pistol', 'firearm', 'rifle')

KNIFE_LIKE_KEYWORDS = ('knife', 'blade')

# ------------------------------------------------------------
# PER-CLASS CONFIDENCE BARS
#
# One bar for everything doesn't work, because the classes are not
# equally trustworthy. Measured on the clips and images in uploads/
# (best confidence per frame, imgsz=480):
#
#   class    real examples                    false positives
#   Pistol   revolver 0.88 med, photo 0.92    drill 0.77 med / 0.85 max
#   Rifle    rifles 0.64 med, 0.87 max        (none seen)
#   Knife    real knives 0.62-0.89            pen 0.58
#   Grenade  (none in any footage)            chef's knife 0.87, phone 0.87
#
# Pistol is the class that fires on phones, drills and dark handheld
# objects, and those land around 0.77-0.85 -- so its bar is 0.85, high
# enough to sit above that group while the revolver (0.88 median)
# still alerts. Rifle isn't confusable with a handheld object, so it
# keeps a lower bar. Grenade and Missile only ever appeared as false
# positives here, so they need near-certainty before they alert.
#
# Tune with the numbers, not by guessing: point the camera at the
# object and read the [WEAPON DEBUG] confidence lines in the terminal.
# ------------------------------------------------------------

CLASS_CONFIDENCE_THRESHOLDS = {
    'pistol': 0.85,
    'gun': 0.85,
    'firearm': 0.85,
    'rifle': 0.70,
    'knife': 0.75,
    'blade': 0.75,
    # Grenade is ON at your request, with the highest bar of any class.
    # Read the warning in CLASSES THAT ARE NEVER A WEAPON above before
    # lowering it: every Grenade detection in this project's footage so
    # far has been a false one, the worst at 0.87.
    'grenade': 0.90,
}

# Bar for a weapon class not listed above.
DEFAULT_CLASS_CONFIDENCE_THRESHOLD = 0.80

GUN_LIKE_MAX_ASPECT_RATIO = 1.3  # height/width above this looks like a held phone, not a gun silhouette

# height/width BELOW this is a flat, wide object seen side-on -- a
# phone held sideways, a remote, a book edge. Real pistols measured
# 0.42-1.04 (revolver) and 0.64-0.97 (shotgun), so 0.40 rejects the
# flattest shapes without touching them.
GUN_LIKE_MIN_ASPECT_RATIO = 0.40

KNIFE_LIKE_MIN_THICKNESS_RATIO = 0.12  # min(w,h)/max(w,h) below this looks needle-thin, like a pen/pencil

# KNIFE IN VIDEO (upload + live). In moving video the knife model
# usually scores a real knife 0.50-0.80, so it rarely passes the 0.75
# knife bar 3 checks in a row. In video a knife ALSO alerts when 3 of
# the last 5 checks see a knife at 0.50+ whose average is 0.60+.
# Measured: real knives fire (kitchen 8.9 s, robbery 3.3 s, 141853
# 4.7 s); no other test clip does (highest 0.51 / 0.58 / 0.60 single
# spikes). Photos keep the normal 0.75 bar.
KNIFE_VOTE_MIN_CONF = 0.50
KNIFE_VOTE_WINDOW = 5
KNIFE_VOTE_HITS = 3
KNIFE_VOTE_MEAN = 0.60


# KNIFE HELD IN A HAND (live only). A knife held up to a webcam often
# scores only 0.25-0.50. Such a low score counts for the knife vote only
# when the box is at a person's hand (a wrist from the pose model within
# LIVE_HELD_KNIFE_HAND_PAD of the box) and the box is knife-shaped (a box
# covering more than 30% of the frame must be long and thin). The live
# vote then needs 3 of the last 5 checks at 0.25+ with an average of 0.30+.
# Measured on the test videos: all 6 knife videos alert in live (before:
# 2-3), none of the other 20 videos raise a knife alert.
LIVE_HELD_KNIFE_MIN_CONF = 0.25
LIVE_KNIFE_VOTE_MEAN = 0.30
LIVE_HELD_KNIFE_HAND_PAD = 0.5
HELD_KNIFE_BIG_AREA = 0.30
HELD_KNIFE_BIG_MIN_ELONGATION = 2.2


class KnifeVote:
    """Knife vote over the last KNIFE_VOTE_WINDOW weapon checks."""

    def __init__(self, min_conf=KNIFE_VOTE_MIN_CONF, min_mean=KNIFE_VOTE_MEAN):
        self.min_conf = min_conf
        self.min_mean = min_mean
        self.recent = deque(maxlen=KNIFE_VOTE_WINDOW)

    def reset(self):
        self.recent.clear()

    def update(self, knives):
        best = max((k['confidence'] for k in knives), default=0.0)
        self.recent.append(best)
        return self.active()

    def active(self):
        hits = [c for c in self.recent if c >= self.min_conf]
        return (
            len(hits) >= KNIFE_VOTE_HITS
            and sum(hits) / len(hits) >= self.min_mean
        )


def class_confidence_bar(name):
    """Confidence a detection of this class must reach to be believed."""

    for keyword, bar in CLASS_CONFIDENCE_THRESHOLDS.items():
        if keyword in name:
            return bar

    return DEFAULT_CLASS_CONFIDENCE_THRESHOLD

print("[MODEL CLASSES]", weapon_model.names)


# ------------------------------------------------------------
# POSE MODEL (neck hold + kick)
# ------------------------------------------------------------

pose_model = None

try:

    _pose_path = os.path.join(MODELS_DIR, POSE_MODEL_FILE)

    # Use models/yolov8n-pose.pt if it's there; otherwise ultralytics
    # downloads it once on first run.
    pose_model = YOLO(
        _pose_path if os.path.exists(_pose_path) else POSE_MODEL_FILE
    )

    pose_model(np.zeros((360, 480, 3), dtype='uint8'), verbose=False)

    print("[POSE] Pose model loaded -- neck hold + kick detection ON.")

except Exception as e:

    pose_model = None

    print(f"[POSE] Could not load pose model ({e}). Neck hold + kick detection OFF.")


POSE_ACTION_LABELS = {
    'neck_hold': 'NECK HOLD',
    'kick': 'KICK',
}


# ------------------------------------------------------------
# VIOLENCE LSTM MODEL
# ------------------------------------------------------------

violence_model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, 2048)
    ),
    Dropout(0.3),

    LSTM(32),

    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

violence_model.load_weights(
    os.path.join(
        MODELS_DIR,
        'violence_lstm_model.h5'
    )
)


# ------------------------------------------------------------
# RESNET50 FEATURE EXTRACTOR
# ------------------------------------------------------------

feature_extractor = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

feature_extractor.trainable = False


# ------------------------------------------------------------
# FAST INFERENCE (main lag fix)
#
# model.predict() is built for big offline batches: every call spins
# up a data pipeline, and for ONE frame that overhead dominates.
# Measured on a 2-core CPU:
#
#   ResNet50, 1 frame:   predict() ~115 ms  ->  compiled graph ~45 ms
#   LSTM, 1 window:      predict()  ~77 ms  ->  compiled graph  ~5 ms
#
# Same weights, same outputs (verified identical), ~3x less time per
# live frame. Both functions accept any batch size.
# ------------------------------------------------------------

@tf.function(
    input_signature=[tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], tf.float32)],
    reduce_retracing=True
)
def _resnet_features(batch):
    return feature_extractor(batch, training=False)


@tf.function(
    input_signature=[tf.TensorSpec([None, SEQUENCE_LENGTH, 2048], tf.float32)],
    reduce_retracing=True
)
def _lstm_score(windows):
    return violence_model(windows, training=False)


# Warm up once at startup so the first live frame isn't slow while
# TensorFlow and YOLO build their graphs.
_resnet_features(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype='float32'))
_lstm_score(np.zeros((1, SEQUENCE_LENGTH, 2048), dtype='float32'))
weapon_model(np.zeros((360, 480, 3), dtype='uint8'), verbose=False)

print("Models loaded successfully.")


# ============================================================
# GLOBAL LIVE MONITORING VARIABLES
# ============================================================

# ------------------------------------------------------------
# PERFORMANCE FIX: rolling FEATURE buffer instead of raw frames.
#
# The old code kept 20 raw frames and, every single incoming frame,
# re-ran ALL 20 of them through ResNet50 from scratch inside
# check_violence(). ResNet50 is the expensive part of this pipeline,
# so that meant ~20x more ResNet50 forward passes than necessary on
# every live capture tick -- this was the actual cause of the live
# camera being slow. Now we extract a feature vector for each new
# frame ONCE, cache it here, and only run the (cheap) LSTM over the
# cached window. This cuts ResNet50 work by ~20x per tick.
# ------------------------------------------------------------
feature_buffer = []

frame_skip_counter = 0

# Only one live frame is analysed at a time. A Lock (not a plain
# bool) because Flask serves requests on multiple threads -- two
# requests could both read busy=False before either set it.
live_inference_lock = threading.Lock()

live_weapon_streak = 0
live_knife_vote = KnifeVote(LIVE_HELD_KNIFE_MIN_CONF, LIVE_KNIFE_VOTE_MEAN)
# Hands seen on the last pose check (for KNIFE HELD IN A HAND), as a
# share of the frame size.
live_hand_points = []
# Time of the previous live frame and whether frames are slow (above).
live_last_frame_t = None
live_slow = False
live_knife_active = False
# Weapon type shown in the alert ('KNIFE', 'PISTOL', ...).
live_weapon_label = 'WEAPON'
# A possible knife was seen on the last weapon check: check EVERY frame
# (not every LIVE_YOLO_EVERY_N-th) until it is confirmed or gone, so a
# knife shown briefly still gets enough checks for the knife vote.
live_knife_follow = False

# Weapon boxes from the most recent YOLO pass. YOLO runs on every
# LIVE_YOLO_EVERY_N-th frame; the frames in between reuse these so
# the boxes on screen don't flicker.
live_last_weapons = []

# Latest annotated frame + status for phone viewers (/watch).
# `seq` goes up with every new frame so a phone can ask for
# "anything newer than N" and wait for it.
live_view = {
    'seq': 0,
    'jpeg': None,
    'status': {},
    'updated': 0.0,
}
live_view_cond = threading.Condition()
viewer_last_seen = 0.0

# Keep a confirmed live alert visible long enough for an operator to see it.
live_violence_alert_until = 0.0

# Edge-trigger flags so we log ONE incident per alert event, not one
# per frame while the alert condition stays true.
live_weapon_alert_logged = False

live_violence_alert_logged = False

# Pose cues (neck hold / kick). The detector keeps its own tracking state;
# `until` holds each confirmed action on screen, `logged` is the
# once-per-event incident flag, `events` the latest confirmed boxes.
live_pose_detector = PoseActionDetector()

live_pose_alert_until = {}

live_pose_alert_logged = {}

live_pose_events = []

live_pose_reason = ''

# Latest scene for the live feed (see SCENE).
live_scene = {'scene': 'UNKNOWN', 'people': 0}

# Frame size the stored weapon / pose boxes were measured on. The browser
# switches frame width when the scene becomes CROWDED, so boxes kept from
# an earlier frame are rescaled to the current one before drawing.
live_weapon_size = None

live_pose_size = None


# ============================================================
# INCIDENT / ALERT DATABASE
#
# Implements the missing pieces from the report's ER diagram
# (DETECTION -> INCIDENT -> ALERT -> ACTION_TAKEN) as one practical
# table. Every confirmed alert (weapon or violence, on a rising edge
# so we log once per event rather than once per frame) is written
# here so it survives past a single request/response, can be
# reviewed, verified or dismissed by an operator, and exported as a
# report -- all three of which the report promises but the original
# code never built.
# ============================================================

DB_PATH = os.path.join('data', 'incidents.db')
db_lock = threading.Lock()

INCIDENT_FRAMES_DIR = os.path.join('uploads', 'incident_frames')


def init_db():

    os.makedirs('data', exist_ok=True)
    os.makedirs(INCIDENT_FRAMES_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    conn.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            detection_type TEXT,
            confidence REAL,
            timestamp TEXT,
            frame_path TEXT,
            verified INTEGER DEFAULT 0,
            false_positive INTEGER DEFAULT 0,
            notes TEXT
        )
    ''')

    # Older databases don't have the scene column yet.
    columns = [row[1] for row in conn.execute('PRAGMA table_info(incidents)')]

    if 'scene' not in columns:
        conn.execute('ALTER TABLE incidents ADD COLUMN scene TEXT')

    conn.commit()
    conn.close()


init_db()


# ============================================================
# ACCESS KEY FOR NETWORK (PHONE) ACCESS
# ============================================================

def load_viewer_key():
    """Read the saved access key, creating a random one on first run.
    It's saved so the phone link keeps working across restarts; delete
    data/viewer_key.txt to generate a new one (old links stop working)."""

    os.makedirs('data', exist_ok=True)

    try:
        with open(VIEWER_KEY_PATH) as f:
            key = f.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass

    key = secrets.token_urlsafe(8)

    with open(VIEWER_KEY_PATH, 'w') as f:
        f.write(key)

    return key


VIEWER_KEY = load_viewer_key()


def request_is_from_laptop():
    """True only for the laptop's own browser on http://localhost.

    Checks BOTH the connecting address and the Host header: a tunnel
    (cloudflared/ngrok) also connects from 127.0.0.1, but its requests
    carry the tunnel's hostname, so they still need the key."""

    host = (request.host or '').rsplit(':', 1)[0].lower()

    return (
        request.remote_addr in ('127.0.0.1', '::1')
        and host in ('localhost', '127.0.0.1', '[::1]')
    )


@app.before_request
def require_key_for_network_access():

    if request_is_from_laptop():
        return None

    if request.path.startswith('/static/'):
        return None

    supplied = request.args.get('key') or request.cookies.get('sentry_key')

    if supplied and secrets.compare_digest(supplied, VIEWER_KEY):
        return None

    return Response(
        "Access key required. Open the phone link shown on the laptop's "
        "Live Monitor page.",
        status=403,
        mimetype='text/plain'
    )


@app.after_request
def remember_key(response):
    """After a phone opens the link once (with ?key=...), store the key
    in a cookie so the page's own requests don't need it in the URL."""

    supplied = request.args.get('key')

    if supplied and secrets.compare_digest(supplied, VIEWER_KEY):
        response.set_cookie(
            'sentry_key', supplied,
            max_age=60 * 60 * 24 * 30, httponly=True, samesite='Lax'
        )

    return response


def get_lan_ip():
    """This laptop's address on the local network (e.g. 192.168.1.5).
    Connecting a UDP socket sends no packets; it just makes the OS pick
    the network interface it would use."""

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


# ============================================================
# EMAIL ALERTS
#
# Every new incident (weapon / violence, live or upload) is emailed with
# its saved frame attached. Settings live in data/email_config.json,
# created on first run -- fill in the sender account there:
#
#   "username":     the Gmail address that SENDS the alerts
#   "app_password": a Gmail App Password for that account (Google
#                   Account -> Security -> 2-Step Verification -> App
#                   passwords). The normal Gmail password won't work.
#   "to":           who receives the alerts (a list)
#
# Nothing is sent until username and app_password are filled in. Open
# http://localhost:5000/api/test_email on the laptop to send a test.
# Sending runs in the background, so detection never waits for email.
# ============================================================

EMAIL_CONFIG_PATH = os.path.join('data', 'email_config.json')

EMAIL_DEFAULTS = {
    "enabled": True,
    "to": [],          # receiver address(es), set in data/email_config.json
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "",
    "app_password": "",
    # A live alert can fire again every few seconds while a threat stays
    # on camera; at most one email per alert type per source (the live
    # camera, or one uploaded file) in this many seconds.
    "min_seconds_between_emails": 60
}

email_last_sent = {}
email_lock = threading.Lock()


def load_email_config():
    """Read data/email_config.json, creating it with the defaults the
    first time. Re-read on every alert, so edits apply without a restart."""

    os.makedirs('data', exist_ok=True)

    if not os.path.exists(EMAIL_CONFIG_PATH):
        with open(EMAIL_CONFIG_PATH, 'w') as f:
            json.dump(EMAIL_DEFAULTS, f, indent=4)

    try:
        with open(EMAIL_CONFIG_PATH) as f:
            config = json.load(f)
    except Exception as e:
        print(f"[EMAIL] Could not read {EMAIL_CONFIG_PATH}: {e}")
        return None

    merged = dict(EMAIL_DEFAULTS)
    merged.update(config)

    if isinstance(merged.get('to'), str):
        merged['to'] = [merged['to']]

    return merged


def email_ready(config):
    return bool(
        config and config.get('enabled')
        and config.get('username') and config.get('app_password')
        and config.get('to')
    )


def send_email(config, subject, html, text, image_path=None):
    """Send one email (blocking). Returns (ok, error_message)."""

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = f"Sentry Alerts <{config['username']}>"
    msg['To'] = ', '.join(config['to'])
    msg.set_content(text)
    msg.add_alternative(html, subtype='html')

    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as f:
            msg.add_attachment(
                f.read(), maintype='image', subtype='jpeg',
                filename=os.path.basename(image_path)
            )

    try:
        with smtplib.SMTP(config['smtp_host'], int(config['smtp_port']), timeout=20) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(config['username'], config['app_password'].replace(' ', ''))
            server.send_message(msg)
        return True, ''
    except Exception as e:
        return False, str(e)


def notify_incident(incident_id, source, detection_type, confidence, scene, frame_path):
    """Email a new incident in the background (rate-limited per type)."""

    config = load_email_config()

    if not email_ready(config):
        return

    # Rate limit per source + type: 'live' + 'violence: kick', or
    # 'clip.mp4' + 'weapon' -- a new upload or a different alert type
    # always gets its own email.
    key = (source, detection_type)
    now = time.time()

    with email_lock:
        last = email_last_sent.get(key, 0)
        if now - last < float(config.get('min_seconds_between_emails', 60)):
            return
        email_last_sent[key] = now

    label = detection_type.upper()
    where = 'Live camera' if source == 'live' else f'Uploaded file: {source}'
    when = datetime.now().strftime('%d %b %Y, %I:%M:%S %p')
    conf = f"{float(confidence) * 100:.0f}%"

    subject = f"⚠ SENTRY ALERT: {label} detected"
    text = (
        f"SENTRY ALERT\n\n"
        f"Type: {label}\nSource: {where}\nTime: {when}\n"
        f"Confidence: {conf}\nScene: {scene or '-'}\nIncident #{incident_id}\n\n"
        f"The captured frame is attached. Review it in the Sentry dashboard."
    )
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:560px;border:2px solid #ff5a36;border-radius:10px;overflow:hidden">
      <div style="background:#ff5a36;color:#fff;padding:14px 18px;font-size:18px;font-weight:bold">
        &#9888; SENTRY ALERT: {label} detected
      </div>
      <div style="padding:16px 18px;color:#222;font-size:14px;line-height:1.7">
        <b>Source:</b> {where}<br>
        <b>Time:</b> {when}<br>
        <b>Confidence:</b> {conf}<br>
        <b>Scene:</b> {scene or '-'}<br>
        <b>Incident:</b> #{incident_id}<br><br>
        The captured frame is attached. Review and verify it in the Sentry dashboard.
      </div>
    </div>"""

    def worker():
        ok, err = send_email(config, subject, html, text, frame_path)
        if ok:
            print(f"[EMAIL] Alert #{incident_id} ({label}) sent to {', '.join(config['to'])}")
        else:
            print(f"[EMAIL] Could not send alert #{incident_id}: {err}")

    threading.Thread(target=worker, daemon=True).start()


# Create data/email_config.json on startup so it's there to fill in.
_email_config = load_email_config()
print(
    "[EMAIL] Alerts ON -> " + ', '.join(_email_config['to'])
    if email_ready(_email_config)
    else f"[EMAIL] Alerts OFF -- fill in username and app_password in {EMAIL_CONFIG_PATH}"
)


def log_incident(source, detection_type, confidence, frame=None, raw_frame=None, scene=None):
    """
    Insert one incident row (an ALERT tied to a DETECTION, in the
    report's terms). Optionally stores the source frame so an
    operator can visually verify it later, and so that anything later
    marked a false positive becomes a usable hard-negative sample.
    Returns the new incident id.
    """

    frame_path = None

    with db_lock:

        conn = sqlite3.connect(DB_PATH)

        cur = conn.execute(
            'INSERT INTO incidents (source, detection_type, confidence, timestamp, scene) '
            'VALUES (?, ?, ?, ?, ?)',
            (source, detection_type, float(confidence),
             datetime.utcnow().isoformat(), scene)
        )

        incident_id = cur.lastrowid

        if frame is not None:

            frame_path = os.path.join(
                INCIDENT_FRAMES_DIR,
                f'{incident_id}.jpg'
            )

            try:
                cv2.imwrite(frame_path, frame)
            except Exception as e:
                print("[INCIDENT FRAME SAVE ERROR]", str(e))
                frame_path = None

            # Also keep the clean frame, with no boxes or banner drawn
            # on it. Anything marked "false positive" then becomes a
            # usable training image for retrain_with_negatives.py --
            # an annotated frame can't be used for that.
            if raw_frame is not None:
                try:
                    cv2.imwrite(
                        os.path.join(INCIDENT_FRAMES_DIR, f'{incident_id}_raw.jpg'),
                        raw_frame
                    )
                except Exception as e:
                    print("[INCIDENT RAW FRAME SAVE ERROR]", str(e))

            conn.execute(
                'UPDATE incidents SET frame_path = ? WHERE id = ?',
                (frame_path, incident_id)
            )

        conn.commit()
        conn.close()

    # Email the alert (background, rate-limited; see EMAIL ALERTS).
    notify_incident(incident_id, source, detection_type, confidence, scene, frame_path)

    return incident_id


@app.route('/api/test_email')
def test_email():
    """Send a test email with the current settings (laptop only)."""

    if not request_is_from_laptop():
        return jsonify({'error': 'Only available on the laptop'}), 403

    config = load_email_config()

    if not email_ready(config):
        return jsonify({
            'ok': False,
            'error': f'Fill in username and app_password in {EMAIL_CONFIG_PATH} first.'
        })

    ok, err = send_email(
        config,
        'Sentry: test email',
        '<p>This is a test from Sentry. Alert emails will arrive like this, with the captured frame attached.</p>',
        'This is a test from Sentry. Alert emails will arrive like this, with the captured frame attached.'
    )

    return jsonify({'ok': ok, 'to': config['to'], 'error': err})


# ============================================================
# UPLOAD VIDEO PROCESSING STATUS
# ============================================================

processing_status = {
    "current": 0,
    "total": 0,
    "active": False,
    # 'analysing' while frames are scanned, 'encoding' while the
    # browser MP4 is made, 'done' after (shown on the upload page).
    "stage": "done",
    # Alerts found so far in the running upload, so the upload page can
    # sound the siren the moment one is found (not only at the end).
    "alerts": 0,
    "last_alert": "",
    # Processing waits on the frame of each new alert until the upload
    # page presses Resume (see UPLOAD PAUSE ON ALERT).
    "paused": False,
    "alert_incident_id": None
}

# ------------------------------------------------------------
# UPLOAD PAUSE ON ALERT
# When a new alert is found while an upload is being analysed, the
# analysis stops on that frame (siren + popup on the upload page) until
# the page calls /api/upload_resume. If the page stops asking for
# progress (tab closed), it carries on by itself after this many seconds.
# ------------------------------------------------------------
UPLOAD_PAUSE_ON_ALERT = True
UPLOAD_PAUSE_PAGE_GONE_SECONDS = 5.0
upload_last_poll = {"t": 0.0}

# Latest analysed frame of the upload in progress (small JPEG), shown as
# the "live processing preview" on the upload page.
processing_preview = {"jpeg": None}

# Width of that preview image.
PREVIEW_WIDTH = 480


# ============================================================
# VIOLENCE DETECTION
# ============================================================

def check_violence(frames_list):
    """Score a list of raw (128,128,3) RGB frames. Not used by the live
    or upload paths (they cache features per frame) -- kept for
    scripts/tests that want a one-shot score."""

    try:

        frames = preprocess_input(
            np.array(frames_list).astype('float32')
        )

        features = _resnet_features(frames).numpy()

        return check_violence_from_features(features)

    except Exception as e:

        print("[VIOLENCE ERROR]", str(e))

        return False, 0.0


# ============================================================
# SINGLE-FRAME FEATURE EXTRACTION
# ============================================================

def extract_feature(frame_rgb_128):
    """
    Run ResNet50 on exactly ONE already-resized (128,128,3) RGB frame
    and return its 2048-d feature vector. Each frame's features are
    computed once and cached, instead of re-extracting a whole
    20-frame window every tick.
    """

    batch = preprocess_input(
        np.expand_dims(frame_rgb_128.astype('float32'), axis=0)
    )

    return _resnet_features(batch).numpy()[0]


def check_violence_from_features(feature_list):
    """
    Score a (SEQUENCE_LENGTH, 2048) window of cached feature vectors.
    Returns (score_above_threshold, score). The alert decision itself
    is made by ViolenceGate, not here.
    """

    try:

        window = np.expand_dims(
            np.array(feature_list, dtype='float32'),
            axis=0
        )

        prob = float(_lstm_score(window).numpy()[0][0])

        return prob >= VIOLENCE_PROB_THRESHOLD, prob

    except Exception as e:

        print("[VIOLENCE ERROR]", str(e))

        return False, 0.0


# ============================================================
# VIOLENCE ALERT GATE
#
# Shared by the live monitor and the upload scan so the two can never
# disagree about what counts as violence.
# ============================================================

class ViolenceGate:
    """Raises a violence alert only when the model score has been high
    (>= VIOLENCE_PROB_THRESHOLD) for at least VIOLENCE_MIN_SECONDS out
    of the last VIOLENCE_WINDOW_SECONDS.

    - A slap or a touch gives at most a short spike -> not enough
      high-score time -> no alert.
    - A fight keeps the score high, with brief dips between exchanges.
      Because it's "time out of the window" rather than "continuous",
      those dips don't reset it.

    `now` is wall-clock seconds for the live feed and video-time
    seconds (frame_count / fps) for an uploaded file, so the timing
    means the same thing in both.
    """

    def __init__(self,
                 window_seconds=VIOLENCE_WINDOW_SECONDS,
                 min_seconds=VIOLENCE_MIN_SECONDS):

        self.window_seconds = window_seconds
        self.min_seconds = min_seconds
        self.reset()

    def reset(self):

        # (timestamp, score) for each check inside the look-back window
        self.history = []
        self.last_reason = 'building up frame window'
        self.high_seconds = 0.0

    def _seconds_above_threshold(self):
        """Time covered by high-score checks: each check 'owns' the gap
        until the next one, so this is independent of frame rate."""

        total = 0.0

        for (t, score), (t_next, _) in zip(self.history, self.history[1:]):
            if score >= VIOLENCE_PROB_THRESHOLD:
                total += t_next - t

        return total

    def update(self, prob, now):
        """Feed one model score in; returns True while the alert
        condition holds."""

        self.history.append((now, prob))

        cutoff = now - self.window_seconds

        while self.history and self.history[0][0] < cutoff:
            self.history.pop(0)

        self.high_seconds = self._seconds_above_threshold()

        confirmed = self.high_seconds >= self.min_seconds

        self.last_reason = (
            f'score {prob:.3f} | high for {self.high_seconds:.1f}s '
            f'of last {self.window_seconds:.0f}s '
            f'(need {self.min_seconds:.1f}s)'
        )

        print(
            f"[VIOLENCE GATE] {'ALERT' if confirmed else 'clear'} -- "
            f"{self.last_reason}"
        )

        return confirmed


live_violence_gate = ViolenceGate()


# ============================================================
# BASE64 IMAGE DECODER
# ============================================================

def decode_base64_image(data_url):

    header, encoded = data_url.split(
        ',',
        1
    )

    binary = base64.b64decode(
        encoded
    )

    arr = np.frombuffer(
        binary,
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    return frame


# ============================================================
# WEAPON DETECTION
# ============================================================

def validated_weapon_detections(results, weak_knives=None, hands=None):
    """Return only detections that pass the weapon false-positive guards.

    The detector's raw boxes are deliberately not shown to the operator. A
    raw YOLO label is a model hypothesis, not an alert: showing rejected
    phone/pen/drill detections as a red "pistol" or "knife" box is
    misleading even when the alert streak correctly remains clear.
    """

    if results is None or len(results) == 0:
        return []

    if results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    return validate_weapon_boxes(results[0].boxes, weak_knives, hands)


def knife_in_hand(box, hands):
    """Is this knife box at a person's hand? `hands` = {'points': [(x, y)
    as a share of frame width/height], 'size': (w, h) of this frame}."""

    try:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
    except Exception:
        return False

    w, h = hands['size']
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    # A big box (part of a body, a long gun close up) must be long and thin.
    if (bw * bh) / float(w * h) > HELD_KNIFE_BIG_AREA and max(bw, bh) / min(bw, bh) < HELD_KNIFE_BIG_MIN_ELONGATION:
        return False

    m = LIVE_HELD_KNIFE_HAND_PAD * max(bw, bh)

    return any(
        x1 - m <= px * w <= x2 + m and y1 - m <= py * h <= y2 + m
        for px, py in hands['points']
    )


def hand_points(people, frame_size):
    """Wrists (and the hand just past each wrist) of every person, as a
    share of the frame size, for knife_in_hand()."""

    w, h = frame_size
    points = []

    for p in people:
        k = p['kpts']
        for wi, ei in ((9, 7), (10, 8)):        # (wrist, elbow), left and right
            if k[wi][2] < 0.25:
                continue
            wx, wy = float(k[wi][0]), float(k[wi][1])
            points.append((wx / w, wy / h))
            if k[ei][2] >= 0.25:
                ex, ey = float(k[ei][0]), float(k[ei][1])
                points.append(((wx + 0.35 * (wx - ex)) / w, (wy + 0.35 * (wy - ey)) / h))

    return points


def knife_thickness_ok(box):
    try:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        return min(width, height) / max(width, height) >= KNIFE_LIKE_MIN_THICKNESS_RATIO
    except Exception:
        return True


def validate_weapon_boxes(boxes, weak_knives=None, hands=None):
    """The false-positive guards themselves, for any list of boxes that
    have .cls / .conf / .xyxy -- YOLO's own, or the tiled ones from
    detect_weapons_tiled() (wrapped in RawBox).

    weak_knives: pass a list (video only) to collect knife boxes below
    the knife bar but at KNIFE_VOTE_MIN_CONF or more, for KnifeVote.
    hands (live only): also collect knives at LIVE_HELD_KNIFE_MIN_CONF or
    more that are at a person's hand (see KNIFE HELD IN A HAND)."""

    accepted = []

    for box in boxes:

        cls_id = int(box.cls[0])
        name = weapon_model.names[cls_id].lower()
        conf = float(box.conf[0])

        print(f"[WEAPON DEBUG] class={name} conf={conf:.2f}")

        # Scissors (and similar shears) are never a weapon here -- drop
        # them before any other check so they can't be drawn or alerted on.
        if any(k in name for k in IGNORED_CLASS_KEYWORDS):
            print(f"[WEAPON DEBUG] ignored '{name}' (not treated as a weapon class)")
            continue

        is_gun_like = any(k in name for k in GUN_LIKE_KEYWORDS)
        is_knife_like = any(k in name for k in KNIFE_LIKE_KEYWORDS)

        # Every weapon class has its own confidence bar (see
        # CLASS_CONFIDENCE_THRESHOLDS). This applies to ALL of them,
        # including Grenade and Missile, which used to alert at the
        # general 0.45 bar and fired on knives and scissors.
        bar = class_confidence_bar(name)

        if conf < bar:
            if (
                is_knife_like and weak_knives is not None
                and knife_thickness_ok(box)
                and (
                    conf >= KNIFE_VOTE_MIN_CONF
                    or (hands and conf >= LIVE_HELD_KNIFE_MIN_CONF and knife_in_hand(box, hands))
                )
            ):
                weak_knives.append({'name': name, 'confidence': conf, 'xyxy': box.xyxy[0].tolist()})
                print(f"[WEAPON DEBUG] weak '{name}' {conf:.2f} kept for the knife vote")
                continue
            print(f"[WEAPON DEBUG] rejected '{name}' (conf {conf:.2f} below {name} bar {bar})")
            continue

        # The shape checks reject phones and drills mistaken for a PISTOL.
        # A rifle is long and is held flat, upright or at an angle, so it
        # is never shape-checked (no phone/drill was ever read as a rifle).
        if is_gun_like and 'rifle' not in name:

            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = max(x2 - x1, 1.0)
                height = max(y2 - y1, 1.0)
                aspect_ratio = height / width
            except Exception:
                aspect_ratio = 1.0

            if aspect_ratio > GUN_LIKE_MAX_ASPECT_RATIO:
                print(f"[WEAPON DEBUG] rejected '{name}' as likely upright handheld object, e.g. a phone or drill (aspect ratio {aspect_ratio:.2f} > {GUN_LIKE_MAX_ASPECT_RATIO})")
                continue

            if aspect_ratio < GUN_LIKE_MIN_ASPECT_RATIO:
                print(f"[WEAPON DEBUG] rejected '{name}' as likely flat object seen side-on, e.g. a phone (aspect ratio {aspect_ratio:.2f} < {GUN_LIKE_MIN_ASPECT_RATIO})")
                continue

            accepted.append({'name': name, 'confidence': conf, 'xyxy': box.xyxy[0].tolist()})
            continue

        elif is_knife_like:

            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = max(x2 - x1, 1.0)
                height = max(y2 - y1, 1.0)
                thickness_ratio = min(width, height) / max(width, height)
            except Exception:
                thickness_ratio = 1.0

            if thickness_ratio < KNIFE_LIKE_MIN_THICKNESS_RATIO:
                print(f"[WEAPON DEBUG] rejected '{name}' as likely pen/pencil (thickness ratio {thickness_ratio:.2f} < {KNIFE_LIKE_MIN_THICKNESS_RATIO})")
                continue

            accepted.append({'name': name, 'confidence': conf, 'xyxy': box.xyxy[0].tolist()})
            continue

        elif any(k in name for k in WEAPON_KEYWORDS):
            accepted.append({'name': name, 'confidence': conf, 'xyxy': box.xyxy[0].tolist()})

    return accepted


# ============================================================
# CROWDED SCENES: TILED WEAPON DETECTION (uploads)
# ============================================================

class RawBox:
    """Stand-in for an ultralytics box (.cls / .conf / .xyxy), so boxes
    found on tiles go through validate_weapon_boxes() unchanged."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [int(cls_id)]
        self.conf = [float(conf)]
        self.xyxy = [np.array(xyxy, dtype='float64')]


def raw_boxes(results, scale=1.0, offset=(0.0, 0.0)):
    """YOLO result -> RawBox list, shifted by `offset` (pixels in the
    frame YOLO saw) and then multiplied by `scale`."""

    if results is None or len(results) == 0 or results[0].boxes is None:
        return []

    out = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        ox, oy = offset
        out.append(RawBox(
            int(box.cls[0]), float(box.conf[0]),
            [(x1 + ox) * scale, (y1 + oy) * scale,
             (x2 + ox) * scale, (y2 + oy) * scale]
        ))

    return out


def detect_weapons_tiled(frame_full, out_scale):
    """Run weapon YOLO on CROWD_TILE_GRID x CROWD_TILE_GRID overlapping
    tiles of the FULL-resolution frame. In a crowd a knife can be a few
    pixels wide once the frame is shrunk to 640 px; each tile gives it
    several times more. Returns unvalidated RawBoxes in output
    coordinates (full-resolution pixels x out_scale)."""

    H, W = frame_full.shape[:2]
    g = CROWD_TILE_GRID

    tile_w = int(W / (g - (g - 1) * CROWD_TILE_OVERLAP))
    tile_h = int(H / (g - (g - 1) * CROWD_TILE_OVERLAP))

    xs = [min(int(i * tile_w * (1 - CROWD_TILE_OVERLAP)), W - tile_w) for i in range(g)]
    ys = [min(int(i * tile_h * (1 - CROWD_TILE_OVERLAP)), H - tile_h) for i in range(g)]

    boxes = []

    for y0 in ys:
        for x0 in xs:

            tile = frame_full[y0:y0 + tile_h, x0:x0 + tile_w]

            results = weapon_model(
                tile,
                verbose=False,
                conf=CONFIDENCE_THRESHOLD
            )

            boxes += raw_boxes(results, scale=out_scale, offset=(x0, y0))

    return boxes


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_weapon_boxes(boxes):
    """One box per weapon: the same weapon found on the full frame and on
    one or more overlapping tiles keeps only its most confident box."""

    kept = []

    for box in sorted(boxes, key=lambda b: b.conf[0], reverse=True):

        xyxy = box.xyxy[0].tolist()

        if any(
            k.cls[0] == box.cls[0]
            and _iou(xyxy, k.xyxy[0].tolist()) > CROWD_TILE_NMS_IOU
            for k in kept
        ):
            continue

        kept.append(box)

    return kept


def has_weapon_class(results):
    """Compatibility helper for the alert pipeline."""
    return bool(validated_weapon_detections(results))


def annotate_validated_weapons(frame, detections):
    """Draw only accepted weapon detections, never raw model guesses."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection['xyxy']]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{detection['name'].upper()} {detection['confidence']:.0%}"
        label_y = max(y1 - 10, 24)
        cv2.putText(
            annotated, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 2
        )
    return annotated


# ============================================================
# POSE PEOPLE (input for neck hold / kick detection)
# ============================================================

def extract_pose_people(results):
    """Turn one pose-model result into plain dicts:
    {'box': [x1, y1, x2, y2], 'kpts': (17, 3) array of x, y, conf}."""

    if results is None or len(results) == 0:
        return []

    r = results[0]

    if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
        return []

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    kpts = r.keypoints.data.cpu().numpy()

    return [
        {'box': boxes[i].tolist(), 'kpts': kpts[i], 'conf': float(confs[i])}
        for i in range(len(boxes))
    ]


def weapon_label(detections):
    """Weapon type(s) for the alert: 'KNIFE', 'PISTOL', 'KNIFE + PISTOL'."""
    names = []
    for d in detections:
        n = d['name'].upper()
        if n not in names:
            names.append(n)
    return ' + '.join(names) or 'WEAPON'


def pose_alert_text(labels):
    """'NECK HOLD + KICK' style text for the banner."""
    return ' + '.join(labels)


def scene_text(scene, people):
    """'CROWDED - 8 people' / 'ISOLATED - 1 person' / '' when unknown."""

    if scene == 'UNKNOWN':
        return ''

    if people == 0:
        return f"{scene} - no one in view"

    return f"{scene} - {people} {'person' if people == 1 else 'people'}"


def scene_tag(scene, people):
    """Suffix for alert banners: ' [CROWDED - 8 people]'."""

    text = scene_text(scene, people)

    return f' [{text}]' if text else ''


def scale_people(people, factor):
    """Pose people measured on a frame `factor` times smaller/larger than
    the one boxes are drawn on -> drawing-frame coordinates."""

    if factor == 1.0:
        return people

    out = []

    for p in people:
        kpts = p['kpts'].copy()
        kpts[:, :2] *= factor
        out.append(dict(p, box=[v * factor for v in p['box']], kpts=kpts))

    return out


def upload_pose_people(frame_full, draw_width, crowded, frame_small=None):
    """Run the pose model for an uploaded image / video frame and return
    people in the coordinates of the `draw_width`-wide frame the boxes
    are drawn on. In a CROWDED scene it runs on a bigger copy of the
    frame (CROWD_POSE_IMGSZ) so small, distant people are found."""

    H, W = frame_full.shape[:2]

    pose_width = min(W, CROWD_POSE_IMGSZ) if crowded else draw_width

    if not crowded and frame_small is not None and frame_small.shape[1] == pose_width:
        # Reuse the 640 px copy already made for weapon detection
        # instead of shrinking the full (maybe 4K) frame again.
        pose_frame = frame_small
    else:
        pose_frame = cv2.resize(frame_full, (pose_width, int(pose_width * H / W)))

    results = pose_model(
        pose_frame,
        verbose=False,
        conf=POSE_PERSON_CONF,
        imgsz=CROWD_POSE_IMGSZ if crowded else 640
    )

    return scale_people(extract_pose_people(results), draw_width / pose_width)


def upload_weapons(frame_full, frame_small, weapon_results, crowded, weak_knives=None):
    """Validated weapon boxes (in frame_small coordinates) for one
    uploaded frame. CROWDED scenes add tiled detection on the full-
    resolution frame (see detect_weapons_tiled())."""

    if not crowded:
        return validated_weapon_detections(weapon_results, weak_knives)

    boxes = raw_boxes(weapon_results)
    boxes += detect_weapons_tiled(
        frame_full, frame_small.shape[1] / frame_full.shape[1]
    )

    return validate_weapon_boxes(merge_weapon_boxes(boxes), weak_knives)


def rescale_detections(detections, from_size, to_size):
    """Move boxes measured on a frame of `from_size` (w, h) onto a frame
    of `to_size`. The live frame width changes when the scene turns
    CROWDED, and boxes from the previous YOLO pass are reused until the
    next one."""

    if not from_size or from_size == to_size:
        return detections

    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]

    return [
        dict(d, xyxy=[d['xyxy'][0] * sx, d['xyxy'][1] * sy,
                      d['xyxy'][2] * sx, d['xyxy'][3] * sy])
        for d in detections
    ]


# ============================================================
# FRAME ENCODER
# ============================================================

def encode_frame_to_base64(frame):

    _, buffer = cv2.imencode(
        '.jpg',
        frame
    )

    return (
        'data:image/jpeg;base64,'
        +
        base64.b64encode(
            buffer
        ).decode('utf-8')
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route('/')
def home():

    return render_template(
        'home.html',
        active='home'
    )


# ============================================================
# LIVE PAGE
# ============================================================

@app.route('/live')
def live():

    return render_template(
        'live.html',
        active='live'
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route('/upload')
def upload_page():

    return render_template(
        'upload.html',
        active='upload'
    )


# ============================================================
# PROGRESS API
# ============================================================

@app.route('/api/upload_preview')
def upload_preview():
    """Latest analysed frame of the running upload (with its boxes)."""

    jpeg = processing_preview.get("jpeg")

    if jpeg is None:
        return Response(status=204)

    response = Response(jpeg, mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route('/api/progress')
def get_progress():

    upload_last_poll["t"] = time.time()

    return jsonify(
        processing_status
    )


@app.route('/api/upload_resume', methods=['POST'])
def upload_resume():
    """Continue an upload analysis paused on an alert."""

    processing_status["paused"] = False

    return jsonify({'ok': True})


# ============================================================
# RESET LIVE MONITORING
# ============================================================

def reset_live_detection():

    global feature_buffer
    global frame_skip_counter
    global live_weapon_streak
    global live_last_weapons
    global live_violence_alert_until
    global live_weapon_alert_logged
    global live_violence_alert_logged
    global live_pose_events
    global live_pose_reason
    global live_weapon_size
    global live_pose_size
    global live_knife_follow
    global live_hand_points
    global live_last_frame_t


    # Wait for any frame that is still being analysed to finish first.
    # Otherwise that frame lands AFTER the reset: it writes into the
    # fresh buffers and re-publishes "live" to phones after Stop.
    with live_inference_lock:

        feature_buffer = []

        frame_skip_counter = 0

        live_weapon_streak = 0

        live_knife_vote.reset()

        live_knife_follow = False

        live_hand_points = []

        live_last_frame_t = None

        live_last_weapons = []

        live_violence_gate.reset()

        live_violence_alert_until = 0.0

        live_weapon_alert_logged = False

        live_violence_alert_logged = False

        live_pose_detector.reset()

        live_pose_alert_until.clear()

        live_pose_alert_logged.clear()

        live_pose_events = []

        live_pose_reason = ''

        live_scene.update(scene='UNKNOWN', people=0)

        live_weapon_size = None

        live_pose_size = None

        mark_live_offline()

    print(
        "[LIVE] Detection state reset."
    )


@app.route('/api/reset_live_detection', methods=['POST'])
def reset_live_detection_api():

    reset_live_detection()

    return jsonify({'ok': True})


# ============================================================
# INCIDENT / ALERT API
# (operator "verify the detected threat" workflow + reporting,
# both called for in the report's use-case diagram / objectives)
# ============================================================

@app.route('/api/incidents')
def list_incidents():

    limit = request.args.get('limit', 100, type=int)

    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM incidents ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
        conn.close()

    incidents = []

    for r in rows:
        row = dict(r)
        if row.get('frame_path'):
            row['frame_url'] = '/' + row['frame_path'].replace('\\', '/')
        incidents.append(row)

    return jsonify(incidents)


@app.route('/api/incidents/summary')
def incidents_summary():

    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute('SELECT COUNT(*) FROM incidents').fetchone()[0]
        verified = conn.execute(
            'SELECT COUNT(*) FROM incidents WHERE verified = 1'
        ).fetchone()[0]
        false_positive = conn.execute(
            'SELECT COUNT(*) FROM incidents WHERE false_positive = 1'
        ).fetchone()[0]
        conn.close()

    return jsonify({
        'total': total,
        'verified': verified,
        'false_positive': false_positive,
        'pending': total - verified - false_positive
    })


@app.route('/api/incidents/<int:incident_id>/verify', methods=['POST'])
def verify_incident(incident_id):

    data = request.json or {}

    # verified=True  -> operator confirmed it's a real threat
    # false_positive=True -> operator dismissed it (this frame becomes
    # a hard-negative candidate for retraining best.pt)
    verified = 1 if data.get('verified') else 0
    false_positive = 1 if data.get('false_positive') else 0
    notes = data.get('notes', '')

    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'UPDATE incidents SET verified = ?, false_positive = ?, notes = ? WHERE id = ?',
            (verified, false_positive, notes, incident_id)
        )
        conn.commit()
        conn.close()

    return jsonify({'ok': True})


@app.route('/api/report')
def download_report():
    """Objective: 'generate reports based on detected incidents'."""

    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute('SELECT * FROM incidents ORDER BY id').fetchall()
        conn.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'ID', 'Source', 'Type', 'Confidence',
        'Timestamp (UTC)', 'Scene', 'Verified', 'False Positive', 'Notes'
    ])

    for r in rows:
        writer.writerow([
            r['id'], r['source'], r['detection_type'],
            f"{r['confidence']:.3f}", r['timestamp'],
            r['scene'] or '',
            'Yes' if r['verified'] else 'No',
            'Yes' if r['false_positive'] else 'No',
            r['notes'] or ''
        ])

    return Response(
        buf.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=incident_report.csv'}
    )


# ============================================================
# LIVE FRAME DETECTION
# ============================================================

@app.route(
    '/api/detect_frame',
    methods=['POST']
)
def detect_frame():
    """
    Analyse one live webcam frame.

    Returns JSON only -- no annotated image. The browser shows the raw
    webcam <video> directly (smooth, no delay) and draws the weapon
    boxes and alert banner on a canvas on top of it from this JSON.
    The old version sent every frame back as a re-encoded JPEG and
    swapped it into an <img>, so the picture on screen was always a
    frame or two old -- that was the visible "lag".
    """

    global feature_buffer
    global frame_skip_counter
    global live_weapon_streak
    global live_last_weapons
    global live_violence_alert_until
    global live_weapon_alert_logged
    global live_violence_alert_logged
    global live_pose_events
    global live_pose_reason
    global live_weapon_size
    global live_pose_size
    global live_knife_active
    global live_weapon_label
    global live_knife_follow
    global live_hand_points
    global live_last_frame_t
    global live_slow

    data = request.json

    if not data or 'image' not in data:
        return jsonify({'error': 'No image received'}), 400

    # Only one frame is analysed at a time. The browser waits for each
    # answer before sending the next frame, so this only drops frames
    # if a second tab is open on the live page.
    if not live_inference_lock.acquire(blocking=False):
        return jsonify({'skip': True, 'buffer_size': len(feature_buffer)})

    try:

        frame = decode_base64_image(data['image'])

        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        now = time.time()

        h, w = frame.shape[:2]

        frame_skip_counter += 1

        # Boxes kept from earlier passes, moved onto this frame's size
        # (the frame gets wider when the scene turns CROWDED).
        live_last_weapons = rescale_detections(live_last_weapons, live_weapon_size, (w, h))
        live_pose_events = rescale_detections(live_pose_events, live_pose_size, (w, h))
        live_weapon_size = live_pose_size = (w, h)

        # Slow laptop? (see SLOW LAPTOP)
        live_slow = (
            live_last_frame_t is not None
            and now - live_last_frame_t > LIVE_SLOW_FRAME_SECONDS
        )
        live_last_frame_t = now

        # Scene from the last pose pass (see SCENE). The bigger CROWDED
        # sizes are skipped on a slow laptop.
        crowded = live_scene['scene'] == 'CROWDED' and not live_slow


        # ====================================================
        # WEAPON DETECTION (every LIVE_YOLO_EVERY_N-th frame)
        # ====================================================

        if frame_skip_counter % LIVE_YOLO_EVERY_N == 0 or live_knife_follow or live_slow:

            weapon_results = weapon_model(
                frame,
                verbose=False,
                # low enough to see a knife held in a hand (each class
                # still has its own bar in validate_weapon_boxes)
                conf=min(CONFIDENCE_THRESHOLD, LIVE_HELD_KNIFE_MIN_CONF),
                imgsz=LIVE_CROWD_YOLO_IMGSZ if crowded else LIVE_WEAPON_IMGSZ
            )

            weak_knives = []
            live_last_weapons = validated_weapon_detections(
                weapon_results, weak_knives,
                {'points': live_hand_points, 'size': (w, h)}
            )

            if live_last_weapons:
                live_weapon_streak += 1
            else:
                live_weapon_streak = 0

            # Knife vote (see KNIFE IN VIDEO): weak knife boxes are only
            # drawn while the vote holds.
            live_knife_active = live_knife_vote.update(
                weak_knives + [d for d in live_last_weapons if any(k in d['name'] for k in KNIFE_LIKE_KEYWORDS)]
            )
            if live_knife_active:
                live_last_weapons = live_last_weapons + weak_knives

            if live_last_weapons:
                live_weapon_label = weapon_label(live_last_weapons)

            live_knife_follow = bool(weak_knives) or any(
                any(k in d['name'] for k in KNIFE_LIKE_KEYWORDS) for d in live_last_weapons
            )

        weapon_detected = (
            live_weapon_streak >= CONSECUTIVE_HITS_REQUIRED
            or live_knife_active
        )


        # ====================================================
        # POSE: NECK HOLD + KICK (on the frames weapon YOLO skips)
        # ====================================================

        if (
            pose_model is not None
            and ((frame_skip_counter + 1) % LIVE_POSE_EVERY_N == 0 or live_slow)
        ):

            # Bigger input once there are several people, so small,
            # distant people are still found and counted.
            pose_imgsz = (
                LIVE_CROWD_YOLO_IMGSZ
                if crowded or (live_pose_detector.people >= CROWD_HINT_PEOPLE and not live_slow)
                else LIVE_YOLO_IMGSZ
            )

            pose_results = pose_model(
                frame,
                verbose=False,
                conf=POSE_PERSON_CONF,
                imgsz=pose_imgsz
            )

            pose_people = extract_pose_people(pose_results)

            live_hand_points = hand_points(pose_people, (w, h))

            live_pose_events = live_pose_detector.update(
                pose_people,
                now
            )

            live_pose_reason = live_pose_detector.last_reason

            live_scene.update(
                scene=live_pose_detector.scene,
                people=live_pose_detector.scene_people
            )

            for ev in live_pose_events:
                live_pose_alert_until[ev['action']] = (
                    now + POSE_ALERT_HOLD_SECONDS
                )

        pose_active = [
            action for action, until in live_pose_alert_until.items()
            if now < until
        ]

        # Boxes to draw: the weapon boxes plus the people involved in a
        # confirmed neck hold / kick.
        live_draw = live_last_weapons + live_pose_events


        # ====================================================
        # VIOLENCE: one ResNet50 feature per frame, cached; the
        # LSTM scores the last SEQUENCE_LENGTH of them.
        # ====================================================

        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        feature_buffer.append(extract_feature(small_rgb))

        if len(feature_buffer) > SEQUENCE_LENGTH:
            feature_buffer.pop(0)

        violence_prob = 0.0
        violence_reason = (
            f'building up frame window '
            f'({len(feature_buffer)}/{SEQUENCE_LENGTH})'
        )

        if len(feature_buffer) == SEQUENCE_LENGTH:

            _, violence_prob = check_violence_from_features(feature_buffer)

            if live_violence_gate.update(violence_prob, now):
                # Keep the alert up long enough for an operator to see it.
                live_violence_alert_until = now + 4

            violence_reason = live_violence_gate.last_reason

        lstm_violence_detected = now < live_violence_alert_until

        # A weapon on screen is shown as that weapon (KNIFE DETECTED),
        # not as general violence. Neck hold / kick still show.
        if weapon_detected:
            lstm_violence_detected = False

        if pose_model is not None and live_pose_reason:
            violence_reason += ' | ' + live_pose_reason

        violence_detected = lstm_violence_detected or bool(pose_active)


        # ====================================================
        # ALERT TEXT
        # ====================================================

        alert_text = ""

        if weapon_detected:
            alert_text += f"{live_weapon_label} DETECTED! "

        if violence_detected:
            alert_text += "VIOLENCE DETECTED!"

            if pose_active:
                alert_text += (
                    ' ('
                    + pose_alert_text(
                        [POSE_ACTION_LABELS[a] for a in pose_active]
                    )
                    + ')'
                )

        live_scene_text = scene_text(live_scene['scene'], live_scene['people'])

        if alert_text:
            alert_text = alert_text.strip() + scene_tag(
                live_scene['scene'], live_scene['people']
            )


        # ====================================================
        # LOG INCIDENTS (rising edge only -- one row per event,
        # not one per frame while the condition stays true).
        # The saved frame is annotated here, only when logging.
        # ====================================================

        new_incident_id = None

        new_pose_actions = [
            a for a in pose_active if not live_pose_alert_logged.get(a)
        ]

        needs_log = (
            (weapon_detected and not live_weapon_alert_logged) or
            (lstm_violence_detected and not live_violence_alert_logged) or
            bool(new_pose_actions)
        )

        if needs_log:

            incident_frame = annotate_validated_weapons(frame, live_draw)

            cv2.putText(
                incident_frame, alert_text.strip(), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        if weapon_detected and not live_weapon_alert_logged:
            new_incident_id = log_incident(
                'live', f"weapon: {live_weapon_label.lower()}", 1.0, frame=incident_frame, raw_frame=frame,
                scene=live_scene_text
            )
            live_weapon_alert_logged = True
        elif not weapon_detected:
            live_weapon_alert_logged = False

        if lstm_violence_detected and not live_violence_alert_logged:
            new_incident_id = log_incident(
                'live', 'violence', violence_prob, frame=incident_frame, raw_frame=frame,
                scene=live_scene_text
            )
            live_violence_alert_logged = True
        elif not lstm_violence_detected:
            live_violence_alert_logged = False

        # One incident per neck hold / kick event, logged as
        # "violence: neck hold" / "violence: kick".
        for action in POSE_ACTION_LABELS:

            if action in pose_active:

                if not live_pose_alert_logged.get(action):

                    conf = max(
                        (e['confidence'] for e in live_pose_events
                         if e['action'] == action),
                        default=1.0
                    )

                    new_incident_id = log_incident(
                        'live',
                        f"violence: {POSE_ACTION_LABELS[action].lower()}",
                        conf,
                        frame=incident_frame,
                        raw_frame=frame,
                        scene=live_scene_text
                    )

                    live_pose_alert_logged[action] = True

            else:

                live_pose_alert_logged[action] = False


        # ====================================================
        # RETURN RESULT -- boxes as 0..1 fractions of the frame so
        # the browser can draw them at any display size.
        # ====================================================

        boxes = [
            {
                'name': d['name'],
                'confidence': round(d['confidence'], 3),
                'x1': d['xyxy'][0] / w,
                'y1': d['xyxy'][1] / h,
                'x2': d['xyxy'][2] / w,
                'y2': d['xyxy'][3] / h,
            }
            for d in live_draw
        ]

        process_ms = int((time.time() - now) * 1000)

        # Frame width the browser should send next (bigger in a crowd).
        send_width = (
            LIVE_CROWD_SEND_WIDTH
            if crowded
            else LIVE_SEND_WIDTH
        )

        publish_to_viewers(
            frame, live_draw, alert_text, {
                'weapon_detected': bool(weapon_detected),
                'weapon_label': live_weapon_label,
                'violence_detected': bool(violence_detected),
                'violence_reason': violence_reason,
                'alert': alert_text,
                'process_ms': process_ms,
                'new_incident_id': new_incident_id,
                'scene': live_scene['scene'],
                'people': live_scene['people'],
                'scene_text': live_scene_text,
            }
        )

        return jsonify({
            'weapon_detected': bool(weapon_detected),
            'weapon_label': live_weapon_label,
            'violence_detected': bool(violence_detected),
            'violence_prob': float(violence_prob),
            # Why the violence gate did or didn't alert -- shown under
            # the status badges in the live UI.
            'violence_reason': violence_reason,
            'boxes': boxes,
            'alert': alert_text,
            'buffer_size': len(feature_buffer),
            'new_incident_id': new_incident_id,
            'process_ms': process_ms,
            'scene': live_scene['scene'],
            'people': live_scene['people'],
            'scene_text': live_scene_text,
            'send_width': send_width
        })

    except Exception as e:

        print("[LIVE DETECTION ERROR]", str(e))

        return jsonify({'error': str(e)}), 500

    finally:

        live_inference_lock.release()


# ============================================================
# PHONE VIEW
#
# How it fits together:
#   laptop /live  --frames-->  /api/detect_frame  --latest annotated
#   frame + status-->  live_view  <--asks for newer frame--  phone /watch
#
# The phone polls /api/watch/frame?after=<seq>. The server holds that
# request open until a newer frame exists (or 2 s pass), then returns
# the JPEG with the current status in a response header -- one request
# per frame, no wasted polling, and it works through tunnels too.
# ============================================================

def _offline_placeholder():
    """JPEG shown on the phone while the laptop isn't monitoring."""

    img = np.zeros((360, 640, 3), dtype=np.uint8)

    cv2.putText(
        img, 'LAPTOP CAMERA OFFLINE', (120, 170),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (140, 152, 172), 2
    )
    cv2.putText(
        img, 'Start Live Monitoring on the laptop', (130, 215),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 110, 125), 1
    )

    return cv2.imencode('.jpg', img)[1].tobytes()


OFFLINE_JPEG = _offline_placeholder()


def publish_to_viewers(frame, weapons, alert_text, status):
    """Store the latest frame + status for phone viewers. The annotated
    JPEG is only built while a phone is actually watching."""

    phone_watching = time.time() - viewer_last_seen < VIEWER_ACTIVE_SECONDS

    jpeg = None

    if phone_watching:

        annotated = annotate_validated_weapons(frame, weapons)

        if alert_text:
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), (0, 0, 200), -1)
            cv2.putText(
                annotated, alert_text.strip(), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )

        jpeg = cv2.imencode(
            '.jpg', annotated,
            [cv2.IMWRITE_JPEG_QUALITY, VIEWER_JPEG_QUALITY]
        )[1].tobytes()

    with live_view_cond:

        live_view['seq'] += 1
        live_view['status'] = status
        live_view['updated'] = time.time()

        if jpeg is not None:
            live_view['jpeg'] = jpeg

        live_view_cond.notify_all()


def mark_live_offline():
    """Called when the laptop stops monitoring, so phones switch to the
    offline screen straight away instead of freezing on the last frame."""

    with live_view_cond:
        live_view['seq'] += 1
        live_view['jpeg'] = None
        live_view['status'] = {}
        live_view['updated'] = 0.0
        live_view_cond.notify_all()


@app.route('/watch')
def watch():

    return render_template('watch.html', active='watch')


@app.route('/api/watch/frame')
def watch_frame():

    global viewer_last_seen

    viewer_last_seen = time.time()

    after = request.args.get('after', -1, type=int)

    with live_view_cond:

        live_view_cond.wait_for(
            lambda: live_view['seq'] > after,
            timeout=2.0
        )

        seq = live_view['seq']
        jpeg = live_view['jpeg']
        status = dict(live_view['status'])
        age = time.time() - live_view['updated']

    online = jpeg is not None and age < VIEWER_STALE_SECONDS

    status['online'] = online
    status['seq'] = seq

    response = Response(
        jpeg if online else OFFLINE_JPEG,
        mimetype='image/jpeg'
    )

    # Status travels with the image so the phone needs one request per frame.
    response.headers['X-Sentry-Status'] = json.dumps(status)
    response.headers['Cache-Control'] = 'no-store'

    return response


@app.route('/api/viewer_info')
def viewer_info():
    """Phone link for the /live page. Only the laptop itself can ask
    (anyone else would need the key to get here anyway)."""

    if not request_is_from_laptop():
        return jsonify({'error': 'Only available on the laptop'}), 403

    ip = get_lan_ip()

    return jsonify({
        'key': VIEWER_KEY,
        'lan_url': (
            f'http://{ip}:{SERVER_PORT}/watch?key={VIEWER_KEY}' if ip else None
        ),
        'listening_on_network': SERVER_HOST != '127.0.0.1',
    })


# ============================================================
# UPLOAD VIDEO / IMAGE DETECTION
# ============================================================

@app.route(
    '/api/upload_video',
    methods=['POST']
)
def upload_video():

    global processing_status

    t_start = time.time()


    # ========================================================
    # GET FILE
    # ========================================================

    file = request.files['video']


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    input_path = os.path.join(
        'uploads',
        file.filename
    )


    os.makedirs(
        'uploads',
        exist_ok=True
    )


    file.save(
        input_path
    )


    # ========================================================
    # IMAGE EXTENSIONS
    # ========================================================

    image_exts = [
        '.jpg',
        '.jpeg',
        '.png',
        '.webp',
        '.bmp'
    ]


    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    if ext in image_exts:

        frame = cv2.imread(
            input_path
        )


        if frame is None:

            return jsonify({
                'error':
                'Could not read image file. '
                'Try a .jpg or .png instead.'
            }), 400


        height, width = frame.shape[:2]


        detect_frame_small = cv2.resize(
            frame,
            (
                640,
                int(
                    640 * height / width
                )
            )
        )


        # Scene: count the people first, so a CROWDED image gets the
        # tiled weapon search.
        image_scene, image_people = 'UNKNOWN', 0

        # Neck hold / kick in a photo (single-frame rules, see
        # pose_actions.single_frame_actions).
        image_actions = []

        if pose_model is not None:

            image_pose = PoseActionDetector()

            people = upload_pose_people(frame, 640, crowded=False)
            image_pose.update(people, 0.0)

            if image_pose.people >= CROWD_HINT_PEOPLE:
                # Close to a crowd -- look again at higher resolution.
                people = upload_pose_people(frame, 640, crowded=True)
                image_pose.reset()
                image_pose.update(people, 0.0)

            image_scene, image_people = image_pose.scene, image_pose.people

            image_actions = pose_actions.single_frame_actions(people)

        weapon_results = weapon_model(
            detect_frame_small,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD
        )


        valid_weapons = upload_weapons(
            frame, detect_frame_small, weapon_results,
            crowded=image_scene == 'CROWDED'
        )
        weapon_detected = bool(valid_weapons)


        annotated_frame = annotate_validated_weapons(
            detect_frame_small, valid_weapons + image_actions
        )


        annotated_frame = cv2.resize(
            annotated_frame,
            (
                width,
                height
            )
        )


        alerts_log = []


        image_scene_text = scene_text(image_scene, image_people)

        if weapon_detected:

            image_weapon_label = weapon_label(valid_weapons)

            alerts_log.append(
                f"Frame 1: {image_weapon_label} DETECTED!"
                + scene_tag(image_scene, image_people)
            )

            log_incident(
                file.filename, f"weapon: {image_weapon_label.lower()}", 1.0,
                frame=annotated_frame, raw_frame=frame,
                scene=image_scene_text
            )

        if image_actions:

            alerts_log.append(
                "Frame 1: VIOLENCE DETECTED! ("
                + pose_alert_text([POSE_ACTION_LABELS[e['action']] for e in image_actions])
                + ")"
                + scene_tag(image_scene, image_people)
            )

            for e in image_actions:
                log_incident(
                    file.filename,
                    f"violence: {POSE_ACTION_LABELS[e['action']].lower()}",
                    e['confidence'],
                    frame=annotated_frame, raw_frame=frame,
                    scene=image_scene_text
                )


        output_filename = (
            'annotated_'
            +
            os.path.splitext(
                file.filename
            )[0]
            +
            '.jpg'
        )


        output_path = os.path.join(
            'uploads',
            output_filename
        )


        cv2.imwrite(
            output_path,
            annotated_frame
        )


        print(
            f"[TIMING] Image processed in "
            f"{time.time() - t_start:.1f}s"
        )


        return jsonify({

            'output_video':
                '/'
                +
                output_path.replace(
                    '\\',
                    '/'
                ),

            'alerts_log':
                alerts_log,

            'total_frames':
                1,

            'is_image':
                True,

            'scene_summary':
                f"Scene: {image_scene_text}" if image_scene_text else ''

        })


    # ========================================================
    # VIDEO PROCESSING
    # ========================================================

    temp_avi_path = os.path.join(
        'uploads',
        'temp_'
        +
        os.path.splitext(
            file.filename
        )[0]
        +
        '.avi'
    )


    output_path = os.path.join(
        'uploads',
        'annotated_'
        +
        os.path.splitext(
            file.filename
        )[0]
        +
        '.mp4'
    )


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        input_path
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *'XVID'
    )


    # Saved copy is capped at OUTPUT_MAX_WIDTH (see UPLOAD SPEED).
    # Even sizes keep the H.264 encoder happy.
    out_width = min(width, OUTPUT_MAX_WIDTH)
    out_height = int(round(height * out_width / width / 2)) * 2
    out_width = out_width // 2 * 2

    out = cv2.VideoWriter(
        temp_avi_path,
        fourcc,
        fps,
        (
            out_width,
            out_height
        )
    )


    # ========================================================
    # TOTAL FRAME COUNT
    # ========================================================

    total_frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    processing_status["total"] = (
        total_frame_count
    )


    processing_status["current"] = 0

    processing_status["active"] = True

    processing_status["stage"] = "analysing"

    processing_status["alerts"] = 0

    processing_status["last_alert"] = ""

    processing_status["paused"] = False

    processing_status["alert_incident_id"] = None

    upload_last_poll["t"] = time.time()

    processing_preview["jpeg"] = None

    # Where each alert starts in the video (seconds), so the upload page
    # can pause the result at the first one.
    alert_events = []

    # Analyse ~UPLOAD_CHECKS_PER_SECOND frames per second of footage
    # (see UPLOAD SPEED); never more often than every FRAME_SKIP-th.
    frame_skip = max(
        FRAME_SKIP,
        int(round(fps / UPLOAD_CHECKS_PER_SECOND)) if fps and fps > 0 else FRAME_SKIP
    )

    print(f"[UPLOAD] {width}x{height} @ {fps:.1f} fps -> analysing every {frame_skip} frame(s), "
          f"saving at {min(width, OUTPUT_MAX_WIDTH)} px wide")


    # ========================================================
    # LOCAL VIOLENCE FEATURE BUFFER
    # ========================================================

    local_feature_buffer = []


    # Same gate as the live feed. Timing is measured in video time, so
    # "1 s of high score" means 1 s of footage regardless of how fast
    # the scan itself runs.
    upload_violence_gate = ViolenceGate()


    alerts_log = []

    frame_count = 0


    # ========================================================
    # ALERT HOLD
    # ========================================================

    ALERT_HOLD_SECONDS = 2


    alert_hold_frames = (
        int(
            fps * ALERT_HOLD_SECONDS
        )
        if fps > 0
        else 30
    )


    alert_hold_remaining = 0

    last_alert_text = ""


    # ========================================================
    # VIDEO STREAKS
    # ========================================================

    weapon_streak = 0

    upload_knife_vote = KnifeVote()

    # Weapon type shown in the alert ('KNIFE', 'PISTOL', ...).
    upload_weapon_label = 'WEAPON'

    upload_weapon_logged = False

    upload_violence_logged = False

    # Neck hold / kick (own tracking state per upload)
    upload_pose_detector = PoseActionDetector()

    upload_pose_logged = {}

    # Scene for this video: current label, seconds spent in each, and
    # the most people seen at once (see SCENE).
    upload_scene = 'UNKNOWN'

    upload_scene_people = 0

    scene_seconds = {}

    peak_people = 0

    seconds_per_check = (frame_skip / fps) if fps and fps > 0 else frame_skip / 30.0


    # ========================================================
    # VIDEO LOOP
    # ========================================================

    while cap.isOpened():

        ret, frame = cap.read()


        if not ret:

            break


        frame_count += 1


        processing_status["current"] = (
            frame_count
        )


        # ====================================================
        # FRAME SKIP FOR UPLOADED VIDEO
        # ====================================================

        if frame_count % frame_skip != 0:

            frame = cv2.resize(frame, (out_width, out_height))

            if alert_hold_remaining > 0:

                cv2.putText(
                    frame,
                    last_alert_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )


                alert_hold_remaining -= 1


            out.write(
                frame
            )

            continue


        # ====================================================
        # WEAPON DETECTION
        # ====================================================

        detect_frame_small = cv2.resize(
            frame,
            (
                640,
                int(
                    640 * height / width
                )
            )
        )


        weapon_results = weapon_model(
            detect_frame_small,
            verbose=False,
            conf=CONFIDENCE_THRESHOLD
        )


        # ----------------------------------------------------
        # Check any YOLO result
        # ----------------------------------------------------

        any_class_seen = (
            len(
                weapon_results[0].boxes
            ) > 0
        )


        # ----------------------------------------------------
        # Check actual weapon
        # ----------------------------------------------------

        # CROWDED (from the previous check) adds the tiled search.
        weak_knives = []
        valid_weapons = upload_weapons(
            frame, detect_frame_small, weapon_results,
            crowded=upload_scene == 'CROWDED',
            weak_knives=weak_knives
        )
        raw_weapon_seen = bool(valid_weapons)

        # Knife vote (see KNIFE IN VIDEO): weak knife boxes are only
        # drawn while the vote holds.
        knife_vote_active = upload_knife_vote.update(
            weak_knives + [d for d in valid_weapons if any(k in d['name'] for k in KNIFE_LIKE_KEYWORDS)]
        )
        if knife_vote_active:
            valid_weapons = valid_weapons + weak_knives

        if valid_weapons:
            upload_weapon_label = weapon_label(valid_weapons)


        # ====================================================
        # DEBUG DETECTIONS
        # ====================================================

        if any_class_seen:

            for box in weapon_results[0].boxes:

                cls_id = int(
                    box.cls[0]
                )

                conf = float(
                    box.conf[0]
                )


                print(
                    f"[DEBUG] Detected class: "
                    f"{weapon_model.names[cls_id]}, "
                    f"confidence: {conf:.2f}"
                )


        # ====================================================
        # POSE: NECK HOLD + KICK
        # ====================================================

        pose_events = []

        if pose_model is not None:

            pose_events = upload_pose_detector.update(
                upload_pose_people(
                    frame, detect_frame_small.shape[1],
                    crowded=(
                        upload_scene == 'CROWDED'
                        or upload_pose_detector.people >= CROWD_HINT_PEOPLE
                    ),
                    frame_small=detect_frame_small
                ),
                frame_count / fps if fps and fps > 0 else frame_count / 30.0
            )

            upload_scene = upload_pose_detector.scene
            upload_scene_people = upload_pose_detector.scene_people

            scene_seconds[upload_scene] = (
                scene_seconds.get(upload_scene, 0.0) + seconds_per_check
            )
            peak_people = max(peak_people, upload_pose_detector.people)

        pose_active = [e['action'] for e in pose_events]


        # ====================================================
        # ANNOTATE
        # ====================================================

        annotated_frame = annotate_validated_weapons(
            detect_frame_small, valid_weapons + pose_events
        )


        annotated_frame = cv2.resize(
            annotated_frame,
            (
                out_width,
                out_height
            )
        )


        # ====================================================
        # WEAPON STREAK
        # ====================================================

        if raw_weapon_seen:

            weapon_streak += 1

        else:

            weapon_streak = 0


        weapon_detected = (
            weapon_streak
            >= CONSECUTIVE_HITS_REQUIRED
            or knife_vote_active
        )


        # ====================================================
        # VIOLENCE FEATURE BUFFER
        # ====================================================

        small = cv2.resize(
            frame,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )


        small_rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )


        # Run ResNet50 once per frame and reuse these cached features for the
        # LSTM, rather than recalculating features for the entire window.
        local_feature_buffer.append(extract_feature(small_rgb))


        if len(local_feature_buffer) > SEQUENCE_LENGTH:

            local_feature_buffer.pop(0)


        # ====================================================
        # VIOLENCE DEFAULT
        # ====================================================

        violence_detected = False

        violence_prob = 0.0


        # ====================================================
        # RUN VIOLENCE MODEL
        # ====================================================

        if len(local_feature_buffer) == SEQUENCE_LENGTH:

            _, violence_prob = check_violence_from_features(
                local_feature_buffer
            )


            # Video time of this frame, so the gate's duration
            # condition is measured against the footage itself.
            video_time = (
                frame_count / fps
                if fps and fps > 0
                else frame_count / 30.0
            )


            violence_detected = upload_violence_gate.update(
                violence_prob,
                video_time
            )


        # ====================================================
        # ALERT TEXT
        # ====================================================

        alert_text = ""

        # A weapon on screen is shown as that weapon (KNIFE DETECTED),
        # not as general violence. Neck hold / kick still show.
        if weapon_detected:
            violence_detected = False


        if weapon_detected:

            alert_text += (
                f"{upload_weapon_label} DETECTED! "
            )


        if violence_detected:

            alert_text += (
                f"VIOLENCE DETECTED! "
                f"({violence_prob:.2f})"
            )


        if pose_active:

            pose_text = pose_alert_text(
                [POSE_ACTION_LABELS[a] for a in pose_active]
            )

            if violence_detected:
                alert_text += f" + {pose_text}"
            else:
                alert_text += f"VIOLENCE DETECTED! ({pose_text})"


        upload_scene_text = scene_text(upload_scene, upload_scene_people)

        if alert_text:
            alert_text = alert_text.strip() + scene_tag(upload_scene, upload_scene_people)


        # ====================================================
        # RISING-EDGE INCIDENT LOGGING
        # ====================================================

        new_alert_event = False

        pause_on_this_frame = False

        # Incidents logged on this frame -- sent with the alert time so the
        # upload page can show each one's popup when playback reaches it.
        frame_incident_ids = []

        if weapon_detected and not upload_weapon_logged:
            frame_incident_ids.append(log_incident(file.filename, f"weapon: {upload_weapon_label.lower()}", 1.0, frame=annotated_frame, raw_frame=frame,
                         scene=upload_scene_text))
            upload_weapon_logged = True
            new_alert_event = True
        elif not weapon_detected:
            upload_weapon_logged = False

        if violence_detected and not upload_violence_logged:
            frame_incident_ids.append(log_incident(file.filename, 'violence', violence_prob, frame=annotated_frame, raw_frame=frame,
                         scene=upload_scene_text))
            upload_violence_logged = True
            new_alert_event = True
        elif not violence_detected:
            upload_violence_logged = False

        for action in POSE_ACTION_LABELS:

            if action in pose_active:

                if not upload_pose_logged.get(action):

                    conf = max(
                        e['confidence'] for e in pose_events
                        if e['action'] == action
                    )

                    frame_incident_ids.append(log_incident(
                        file.filename,
                        f"violence: {POSE_ACTION_LABELS[action].lower()}",
                        conf,
                        frame=annotated_frame,
                        raw_frame=frame,
                        scene=upload_scene_text
                    ))

                    upload_pose_logged[action] = True
                    new_alert_event = True

            else:

                upload_pose_logged[action] = False


        # ====================================================
        # DRAW ALERT
        # ====================================================

        if alert_text:

            cv2.putText(
                annotated_frame,
                alert_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )


            if new_alert_event:
                alerts_log.append(
                    f"Frame {frame_count}: "
                    f"{alert_text}"
                )
                alert_events.append({
                    'frame': frame_count,
                    'time': round(
                        (frame_count - 1) / fps if fps and fps > 0 else frame_count / 30.0, 2
                    ),
                    'text': alert_text,
                    'incident_ids': frame_incident_ids,
                })
                processing_status["last_alert"] = alert_text
                processing_status["alert_incident_id"] = (
                    frame_incident_ids[0] if frame_incident_ids else None
                )
                processing_status["alerts"] += 1
                pause_on_this_frame = UPLOAD_PAUSE_ON_ALERT


            last_alert_text = (
                alert_text
            )


            alert_hold_remaining = (
                alert_hold_frames
            )


        elif alert_hold_remaining > 0:

            cv2.putText(
                annotated_frame,
                last_alert_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )


            alert_hold_remaining -= 1


        # ====================================================
        # LIVE PROCESSING PREVIEW (upload page)
        # ====================================================

        preview = cv2.resize(
            annotated_frame,
            (PREVIEW_WIDTH, int(PREVIEW_WIDTH * out_height / out_width))
        )
        processing_preview["jpeg"] = cv2.imencode(
            '.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )[1].tobytes()


        # ====================================================
        # PAUSE ON ALERT (until Resume on the upload page)
        # ====================================================

        if pause_on_this_frame:

            pause_on_this_frame = False
            processing_status["paused"] = True

            while (
                processing_status["paused"]
                and time.time() - upload_last_poll["t"] < UPLOAD_PAUSE_PAGE_GONE_SECONDS
            ):
                time.sleep(0.1)

            processing_status["paused"] = False


        # ====================================================
        # WRITE FRAME
        # ====================================================

        out.write(
            annotated_frame
        )


    # ========================================================
    # CLOSE VIDEO
    # ========================================================

    cap.release()

    out.release()


    processing_status["active"] = False

    processing_status["stage"] = "encoding"


    # ========================================================
    # TIMING
    # ========================================================

    print(
        f"[TIMING] Detection loop took "
        f"{time.time() - t_start:.1f}s"
    )


    # ========================================================
    # FFMPEG CONVERSION
    # ========================================================

    t_ffmpeg = time.time()


    ffmpeg_path = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )


    subprocess.run([
        ffmpeg_path,
        '-y',
        '-i',
        temp_avi_path,
        '-vcodec',
        'libx264',
        '-preset',
        FFMPEG_PRESET,
        '-pix_fmt',
        'yuv420p',
        output_path
    ])

    processing_status["stage"] = "done"


    # ========================================================
    # DELETE TEMP FILE
    # ========================================================

    os.remove(
        temp_avi_path
    )


    print(
        f"[TIMING] FFmpeg conversion took "
        f"{time.time() - t_ffmpeg:.1f}s"
    )


    # ========================================================
    # SCENE SUMMARY
    # ========================================================

    scene_summary = ''

    if scene_seconds:
        scene_summary = (
            'Scene: '
            + ' · '.join(
                f'{label} {secs:.1f}s'
                for label, secs in sorted(scene_seconds.items(), key=lambda kv: -kv[1])
            )
            + f" · up to {peak_people} {'person' if peak_people == 1 else 'people'} at once"
        )


    # ========================================================
    # RETURN VIDEO RESULT
    # ========================================================

    return jsonify({

        'output_video':
            '/'
            +
            output_path.replace(
                '\\',
                '/'
            ),

        'alerts_log':
            alerts_log,

        'total_frames':
            frame_count,

        'is_image':
            False,

        'scene_summary':
            scene_summary,

        'alert_events':
            alert_events

    })


# ============================================================
# SERVE UPLOADED FILES
# ============================================================

@app.route(
    '/uploads/<path:filename>'
)
def serve_uploaded(filename):

    return send_from_directory(
        'uploads',
        filename
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == '__main__':

    lan_ip = get_lan_ip()

    print()
    print("=" * 60)
    print(f"  Laptop (camera):  http://localhost:{SERVER_PORT}/live")

    if SERVER_HOST != '127.0.0.1' and lan_ip:
        print(f"  Phone (same Wi-Fi): http://{lan_ip}:{SERVER_PORT}/watch?key={VIEWER_KEY}")

    print("=" * 60)
    print()

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        # One thread per request: the phone's waiting frame request
        # must not block the laptop's detection requests.
        threaded=True
    )