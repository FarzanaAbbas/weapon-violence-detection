"""
Pose-based violence cues -- NECK HOLD and KICK -- plus scene awareness
(how many people are in view: ISOLATED / MODERATE / CROWDED).

Input is YOLOv8-pose output (COCO 17 keypoints) turned into plain dicts by
extract_pose_people() in app.py:

    {'box': [x1, y1, x2, y2], 'kpts': (17, 3) array of x, y, conf, 'conf': float}

Only numpy is needed here, so the rules can be tested without a model.

NECK HOLD  one person's wrist on ANOTHER person's neck, held for
           NECK_MIN_SECONDS out of the last NECK_WINDOW_SECONDS.

KICK       a leg extended toward ANOTHER person -- raised to roughly hip
           height or higher (the shape of a kick at its peak), or swung
           fast. KICK_MIN_EVENTS such frames inside KICK_WINDOW_SECONDS.

           Measured on the kickboxing clip (6296651): the roundhouse is
           visible for only 3-4 sampled frames, right after a camera cut,
           with the leg at 84-119 deg from vertical and the foot on the
           other fighter. A speed-only check can't see that (it needs the
           same leg in two clean frames in a row), so the shape check is
           the main path and speed is the backup for low/body kicks.

Both cues need a SECOND person, so kicking the air (shadowboxing, kata) and
a hand on one's own neck don't alert.

Timing uses the `now` you pass in (wall clock for live, video time for
uploads), same as ViolenceGate in app.py.

Known limits: a long hug can look like a neck hold; a kick straight toward
or away from the camera is foreshortened and can be missed; two people
sitting on the floor with legs stretched toward each other can look like a
kick for a moment (the both-legs check below removes most of that).
"""

import numpy as np

# ---------------------------------------------------------------- debug

# One [POSE DEBUG] line per checked frame. Set False once tuned.
DEBUG = False

# ---------------------------------------------------------------- general

# A keypoint below this confidence is treated as not seen.
KP_CONF = 0.40

# Lower bar for the hip / knee / ankle points used by the kick check:
# fighters overlap and motion-blur, so their leg points come out weaker.
KICK_KP_CONF = 0.25

# Ignore people smaller than this many pixels (box height or width,
# whichever is larger) -- their keypoints are too noisy. Kept low so
# distant people in a crowd still count.
MIN_PERSON_SIZE_PX = 40

# Two detections are the SAME person (the pose model sometimes returns
# a person twice) when their matching keypoints are on average closer
# than this fraction of the person's size.
DUPLICATE_KPT_DIST = 0.15

# Two people can only interact if their sizes are similar. A much smaller
# person is further from the camera, and their hand only LOOKS like it is
# on the nearer person's neck. Ratio of the larger box to the smaller.
MAX_SIZE_RATIO = 2.5

# ---------------------------------------------------------------- neck hold

# How close a wrist must be to the other person's neck, as a fraction of
# their shoulder width. Raise if real holds are missed, lower if hugs fire.
#
# Was 0.60, which reached the shoulder joints (they sit 0.5 shoulder-widths
# from the neck): soldiers in a stack formation, each with a hand on the
# shoulder of the man in front (clips 12767079 / 13208041), read as neck
# holds. The wrist must also be closer to the neck than to either shoulder.
NECK_RADIUS_SCALE = 0.45

# The hand reaches past the wrist keypoint: the point checked is the wrist
# moved this fraction of a forearm further along the forearm (elbow ->
# wrist). On the Gemini choke clip the wrists sat 1.2-2x the radius from
# the neck while the hands were clearly on the throat.
HAND_EXTEND = 0.35

# Two hands on the neck: each may be up to this far (fraction of shoulder
# width) from the throat line -- the line from the base of the neck up to
# the head. A two-handed grip is a strong sign on its own, so the
# "closer to a shoulder" test isn't applied to it (in a 3/4 view the
# victim's shoulder point sits right beside the throat).
NECK_TWO_HAND_SCALE = 0.75
# How long the hand must stay on the neck (out of the last
# NECK_WINDOW_SECONDS). In clip 12767079 a soldier's hand passes over the
# neck of the man in front for ~0.5 s while reaching for his far
# shoulder (that's a single hand -- see below); in the Gemini choke clip
# the first shot of the two-handed choke lasts 1.38 s before the camera
# cuts. 1.0 s leaves margin under that.
NECK_WINDOW_SECONDS = 2.5
NECK_MIN_SECONDS = 1.0

# A single hand on the throat line is weaker evidence: in the same soldier
# clip one man's hand lies across the next man's throat for ~1.8 s while
# gripping his far shoulder. One-handed holds must last this long, out of
# the last NECK_ONE_HAND_WINDOW_SECONDS. (NECK_MIN_SECONDS above applies
# to two-handed grips.)
NECK_ONE_HAND_WINDOW_SECONDS = 4.0
NECK_ONE_HAND_MIN_SECONDS = 2.5

# ---------------------------------------------------------------- kick

# Hip->ankle distance as a fraction of the person's size (the larger side
# of their box). A straight leg is ~0.45; a bent, tucked leg ~0.25.
KICK_MIN_REACH = 0.30

# Hip->knee->ankle angle; 180 is a straight leg. Skipped if the knee
# isn't seen.
KICK_MIN_KNEE_ANGLE_DEG = 120.0

# Angle of the hip->ankle line from straight down (standing is ~0,
# horizontal is 90). At or above this the leg is raised like a kick.
KICK_HIGH_ANGLE_DEG = 70.0

# Backup path for lower (body / leg) kicks: leg at least this far from
# vertical AND moving at least KICK_MIN_SPEED.
KICK_MIN_LEG_ANGLE_DEG = 35.0

# Ankle speed relative to the hip, in person-sizes per second. Walking
# stays around 1; a kick is 2-5.
KICK_MIN_SPEED = 2.0

# Both legs at least this far from vertical (feet apart) = sitting or
# lying, not kicking.
BOTH_LEGS_OUT_ANGLE_DEG = 55.0

# The foot must reach the other person's box, grown on every edge by this
# fraction of that person's size. On the kickboxing clip the foot was
# drawn up to 0.41 short of the other fighter's box (the pose model places
# the ankle short of the contact point), so 0.45 leaves a little room.
KICK_TARGET_PAD = 0.45

# A kick lands from a leg's length away: the kicker's hips must be at
# least this far outside the other person's box (share of that box's
# larger side). Someone lying on the floor under or beside another person
# (pinned victim in 8689848, judo throw in 6253404) has hips 0.36 or
# less from them; real kicks 0.45 (close kick) to 1.33.
KICK_MIN_HIP_GAP = 0.40

# The kicker must be a whole, clearly seen body: at least this many of the
# 17 keypoints at KP_CONF, and a shoulder at KICK_MIN_SHOULDER_CONF. In a
# gun close-up (clip 14443035) the pose model mistakes forearms and hands
# for legs; those phantom "people" had at most 7 confident points and
# shoulders under 0.4, while every real kicker in the test clips had 10+
# points and a shoulder at 0.7+.
KICK_MIN_VISIBLE_KPTS = 9
KICK_MIN_SHOULDER_CONF = 0.5

KICK_WINDOW_SECONDS = 1.5
KICK_MIN_EVENTS = 2

# A kick is quick. A "kick shape" seen in every frame for longer than
# this is someone sitting or lying with a leg out, not a kick.
KICK_MAX_HOLD_SECONDS = 2.5

# ---------------------------------------------------------------- scene

# People in view (smoothed over SCENE_WINDOW_SECONDS):
#   <= ISOLATED_MAX_PEOPLE           -> ISOLATED
#   >= CROWDED_MIN_PEOPLE            -> CROWDED
#   anything between                 -> MODERATE
ISOLATED_MAX_PEOPLE = 2
CROWDED_MIN_PEOPLE = 6
SCENE_WINDOW_SECONDS = 2.0

# ---------------------------------------------------------------- misc

# Samples further apart than this are not treated as continuous.
MAX_SAMPLE_GAP_SECONDS = 0.5

# SLOW COMPUTERS (live). A laptop that analyses only ~1 frame a second
# gives one pose check a second. Each check then owns the time to the
# next one, up to NECK_MAX_SAMPLE_CREDIT (was MAX_SAMPLE_GAP_SECONDS,
# 0.5 s, so a neck hold could never reach 1.0 s), and the kick window
# grows to KICK_WINDOW_GAPS check-gaps. At 8+ checks a second (uploads,
# fast laptops) nothing changes.
NECK_MAX_SAMPLE_CREDIT = 1.2
# ...and a neck hold also needs at least this many checks that saw it, so
# two lucky frames on a slow laptop are not enough (always met at 8+/s).
NECK_MIN_TRUE_CHECKS = 3
# The one-hand hold (2.5 s) needs more looks: with 1-2 looks a second a
# hand resting near a neighbour's neck (soldiers in a line) was enough.
NECK_ONE_HAND_MIN_CHECKS = 6
KICK_WINDOW_GAPS = 2.2
KICK_WINDOW_MAX_SECONDS = 3.0

# Tracks not seen for this long are dropped.
TRACK_TIMEOUT_SECONDS = 1.0

# COCO keypoint indices
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

LEGS = (
    (L_HIP, L_KNEE, L_ANK),
    (R_HIP, R_KNEE, R_ANK),
)


# ============================================================ helpers

def _ok(kpts, idx, conf=KP_CONF):
    return kpts[idx][2] >= conf


def _pt(kpts, idx):
    return kpts[idx][:2].astype('float64')


def _box_height(person):
    x1, y1, x2, y2 = person['box']
    return float(y2 - y1)


def _size(person):
    """Larger side of the box. A person kicking or lying is wider than
    tall, so height alone would understate their size."""
    x1, y1, x2, y2 = person['box']
    return float(max(x2 - x1, y2 - y1, 1.0))


def _union_box(a, b):
    return [
        min(a[0], b[0]), min(a[1], b[1]),
        max(a[2], b[2]), max(a[3], b[3]),
    ]


def _gap_to_box(pt, box):
    """Distance from a point to a box (0 inside), as a share of the box's
    larger side."""
    x1, y1, x2, y2 = box
    dx = max(x1 - pt[0], 0.0, pt[0] - x2)
    dy = max(y1 - pt[1], 0.0, pt[1] - y2)
    return float(np.hypot(dx, dy)) / max(x2 - x1, y2 - y1, 1.0)


def _near_box(pt, box, pad):
    """Point inside the box grown by `pad` x the box's larger side on
    every edge. (A share of the larger side, not of each side: a person
    seen side-on has a narrow box, and a share of its width would leave
    almost no margin.)"""
    x1, y1, x2, y2 = box
    m = pad * max(x2 - x1, y2 - y1)
    return x1 - m <= pt[0] <= x2 + m and y1 - m <= pt[1] <= y2 + m


def _body_scale(person):
    """Pose-independent body size: torso length (shoulder to hip) when
    seen, else shoulder width, else a share of box height. The box itself
    is a poor measure here -- a kicking leg makes the box twice as wide."""

    k = person['kpts']
    sho = [_pt(k, i) for i in (L_SHO, R_SHO) if _ok(k, i, KICK_KP_CONF)]
    hip = [_pt(k, i) for i in (L_HIP, R_HIP) if _ok(k, i, KICK_KP_CONF)]

    if sho and hip:
        return max(float(np.linalg.norm(np.mean(sho, axis=0) - np.mean(hip, axis=0))), 1.0)

    if len(sho) == 2:
        return max(1.4 * float(np.linalg.norm(sho[0] - sho[1])), 1.0)

    return max(0.3 * _box_height(person), 1.0)


def _similar_size(a, b):
    sa, sb = _body_scale(a), _body_scale(b)
    return max(sa, sb) / min(sa, sb) <= MAX_SIZE_RATIO


def is_duplicate(a, b):
    """True when two detections are the same person."""

    ka, kb = a['kpts'], b['kpts']
    both = (ka[:, 2] >= KICK_KP_CONF) & (kb[:, 2] >= KICK_KP_CONF)

    if both.sum() < 3:
        return False

    dist = float(np.linalg.norm(ka[both, :2] - kb[both, :2], axis=1).mean())

    return dist < DUPLICATE_KPT_DIST * min(_size(a), _size(b))


def dedupe_people(persons):
    """Drop small/noisy detections and keep one copy of each person
    (the higher-confidence one)."""

    persons = [p for p in persons if _size(p) >= MIN_PERSON_SIZE_PX]
    persons = sorted(persons, key=lambda p: p.get('conf', 0.0), reverse=True)

    kept = []

    for p in persons:
        if not any(is_duplicate(p, q) for q in kept):
            kept.append(p)

    return kept


def scene_label(people):
    if people <= ISOLATED_MAX_PEOPLE:
        return 'ISOLATED'
    if people >= CROWDED_MIN_PEOPLE:
        return 'CROWDED'
    return 'MODERATE'


# ============================================================ neck hold

def neck_point(person):
    """(neck_xy, radius) of the neck region, or None if shoulders aren't seen."""

    k = person['kpts']

    if not (_ok(k, L_SHO) and _ok(k, R_SHO)):
        return None

    ls, rs = _pt(k, L_SHO), _pt(k, R_SHO)
    sho_mid = (ls + rs) / 2.0
    sho_w = float(np.linalg.norm(ls - rs))

    # Side-on people have a tiny shoulder width; fall back to body height.
    scale = max(sho_w, 0.22 * _box_height(person))

    if _ok(k, NOSE):
        head = _pt(k, NOSE)
    elif _ok(k, L_EAR) and _ok(k, R_EAR):
        head = (_pt(k, L_EAR) + _pt(k, R_EAR)) / 2.0
    else:
        head = sho_mid + np.array([0.0, -0.8 * scale])

    # Neck sits about a third of the way from the shoulders to the head.
    neck = sho_mid + 0.33 * (head - sho_mid)

    return neck, NECK_RADIUS_SCALE * scale


def _hand_point(k, wrist_i, elbow_i):
    """Where the hand is: past the wrist along the forearm."""

    wrist = _pt(k, wrist_i)

    if k[elbow_i][2] >= KICK_KP_CONF:
        return wrist + HAND_EXTEND * (wrist - _pt(k, elbow_i))

    return wrist


def _throat_line(victim):
    """(base_of_neck, head_point, shoulder_scale) or None."""

    k = victim['kpts']

    if not (_ok(k, L_SHO) and _ok(k, R_SHO)):
        return None

    sho_mid = (_pt(k, L_SHO) + _pt(k, R_SHO)) / 2.0
    scale = max(float(np.linalg.norm(_pt(k, L_SHO) - _pt(k, R_SHO))),
                0.22 * _box_height(victim))

    if _ok(k, NOSE):
        head = _pt(k, NOSE)
    elif _ok(k, L_EAR) and _ok(k, R_EAR):
        head = (_pt(k, L_EAR) + _pt(k, R_EAR)) / 2.0
    else:
        head = sho_mid + np.array([0.0, -0.8 * scale])

    # From just above the shoulders to just under the chin.
    return sho_mid + 0.10 * (head - sho_mid), sho_mid + 0.75 * (head - sho_mid), scale


def _dist_to_segment(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-6), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def neck_hold_between(attacker, victim, stats=None):
    """Confidence (0..1) that `attacker` has a hand on `victim`'s neck, else 0."""
    return neck_hold_detail(attacker, victim, stats)[0]


def neck_hold_detail(attacker, victim, stats=None):
    """(confidence, two_hands) -- confidence 0 when there's no hold.
    `stats` (optional dict) collects debug numbers.

    One hand: must be on the throat line (within NECK_RADIUS_SCALE) and
    closer to it than to either shoulder joint -- otherwise it's a hand on
    a shoulder (stack formation).
    Two hands: each within NECK_TWO_HAND_SCALE of the throat line."""

    nk = neck_point(victim)
    line = _throat_line(victim)

    if nk is None or line is None:
        return 0.0, False

    neck, radius = nk
    base, top, scale = line
    k = attacker['kpts']
    vk = victim['kpts']
    shoulders = (_pt(vk, L_SHO), _pt(vk, R_SHO))

    one_hand = []
    two_hand = []

    for w, e in ((L_WRI, L_ELB), (R_WRI, R_ELB)):
        if k[w][2] < KICK_KP_CONF:
            continue

        hand = _hand_point(k, w, e)
        conf = float(k[w][2])

        line_dist = _dist_to_segment(hand, base, top)
        if stats is not None:
            stats['neck_ratio'] = min(stats.get('neck_ratio', 1e9), line_dist / radius)

        # A hand resting on a shoulder is closer to that shoulder joint
        # than to the throat.
        on_shoulder = any(np.linalg.norm(hand - s) < line_dist for s in shoulders)

        if conf >= KP_CONF and line_dist <= radius and not on_shoulder:
            one_hand.append(conf)

        if line_dist <= NECK_TWO_HAND_SCALE * scale:
            two_hand.append(conf)

    if len(two_hand) == 2:
        return float(np.mean(two_hand)), True

    if one_hand:
        return 0.75 * max(one_hand), False

    return 0.0, False


# ============================================================ kick

def _hip_ref(k):
    """Centre of the visible hips, or None."""
    hips = [_pt(k, i) for i in (L_HIP, R_HIP) if _ok(k, i, KICK_KP_CONF)]
    return np.mean(hips, axis=0) if hips else None


def _leg(person, hip_i, knee_i, ank_i, hip_ref):
    """Geometry of one leg, or None if it can't be measured."""

    k = person['kpts']

    if not _ok(k, ank_i, KICK_KP_CONF) or hip_ref is None:
        return None

    hip = _pt(k, hip_i) if _ok(k, hip_i, KICK_KP_CONF) else hip_ref
    ank = _pt(k, ank_i)
    leg = ank - hip

    knee_deg = None

    if _ok(k, knee_i, KICK_KP_CONF):
        knee = _pt(k, knee_i)
        a, b = hip - knee, ank - knee
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-6:
            knee_deg = float(np.degrees(np.arccos(
                np.clip(np.dot(a, b) / denom, -1.0, 1.0)
            )))

    return {
        'ank_i': ank_i,
        'ank': ank,
        'reach': float(np.linalg.norm(leg)) / _size(person),
        # From straight down; an ankle above the hip is > 90.
        'angle': float(np.degrees(np.arctan2(abs(leg[0]), leg[1]))),
        'knee': knee_deg,
    }


def kick_check(person, others, prev, now, stats=None):
    """Is `person` kicking one of `others` right now?

    `prev` is this person's previous track sample
    ({'t': float, 'rel': {ankle_idx: xy relative to hip}}) or None.
    Returns confidence (0..1) or None."""

    k = person['kpts']

    if (
        int((k[:, 2] >= KP_CONF).sum()) < KICK_MIN_VISIBLE_KPTS
        or max(k[L_SHO][2], k[R_SHO][2]) < KICK_MIN_SHOULDER_CONF
    ):
        if stats is not None and 'kick_fail' not in stats:
            stats['kick_fail'] = 'body not clearly seen'
        return None

    hip_ref = _hip_ref(k)

    if hip_ref is None:
        if stats is not None and 'kick_fail' not in stats:
            stats['kick_fail'] = 'no hip seen'
        return None

    legs = [
        leg for leg in (_leg(person, *ids, hip_ref) for ids in LEGS)
        if leg is not None
    ]

    if not legs:
        if stats is not None and 'kick_fail' not in stats:
            stats['kick_fail'] = 'no ankle seen'
        return None

    size = _size(person)

    # Sitting / lying with both legs out: both legs well off vertical with
    # the feet clearly apart. A kicker stands on the other leg, which stays
    # near vertical. (When both legs read as raised with the feet together,
    # the model has put both legs on the one kicking leg -- common when the
    # other leg is hidden -- so that still counts.)
    if len(legs) == 2 and all(l['angle'] >= BOTH_LEGS_OUT_ANGLE_DEG for l in legs):
        if np.linalg.norm(legs[0]['ank'] - legs[1]['ank']) / size > 0.15:
            if stats is not None:
                stats['kick_fail'] = 'both legs out (sitting/lying)'
            return None

    dt = (now - prev['t']) if prev else None

    best = None

    for leg in legs:

        def fail(why):
            if stats is not None:
                stats['kick_fail'] = why

        if stats is not None:
            stats['kick_angle'] = max(stats.get('kick_angle', 0.0), leg['angle'])

        if leg['reach'] < KICK_MIN_REACH:
            fail(f"leg not extended (reach {leg['reach']:.2f})")
            continue

        if leg['knee'] is not None and leg['knee'] < KICK_MIN_KNEE_ANGLE_DEG:
            fail(f"knee bent ({leg['knee']:.0f} deg)")
            continue

        speed = 0.0
        if dt and 0 < dt <= MAX_SAMPLE_GAP_SECONDS and leg['ank_i'] in prev['rel']:
            rel = leg['ank'] - hip_ref
            speed = float(np.linalg.norm(rel - prev['rel'][leg['ank_i']])) / size / dt
            if stats is not None:
                stats['kick_speed'] = max(stats.get('kick_speed', 0.0), speed)

        high = leg['angle'] >= KICK_HIGH_ANGLE_DEG
        fast = speed >= KICK_MIN_SPEED and leg['angle'] >= KICK_MIN_LEG_ANGLE_DEG

        if not (high or fast):
            fail(f"leg down ({leg['angle']:.0f} deg, speed {speed:.1f})")
            continue

        target = next(
            (o for o in others
             if _similar_size(person, o)
             and _gap_to_box(hip_ref, o['box']) >= KICK_MIN_HIP_GAP
             and _near_box(leg['ank'], o['box'], KICK_TARGET_PAD)),
            None
        )

        if target is None:
            fail(f"leg up ({leg['angle']:.0f} deg) but foot not on anyone")
            continue

        conf = 0.6 + 0.2 * high + 0.2 * fast

        if stats is not None:
            stats['kick_fail'] = 'KICK'

        if best is None or conf > best[0]:
            best = (conf, _union_box(person['box'], target['box']))

    return best


# ============================================================ helpers

def _seconds_true(history):
    """Time covered by True checks; each check owns the gap to the next,
    up to MAX_SAMPLE_GAP_SECONDS -- or up to NECK_MAX_SAMPLE_CREDIT when
    the next check is True too. So on a slow laptop (~1 check a second)
    two True checks in a row make a second, but one stray frame does not."""

    total = 0.0

    for (t, flag), (t_next, flag_next) in zip(history, history[1:]):
        if flag:
            total += min(
                t_next - t,
                NECK_MAX_SAMPLE_CREDIT if flag_next else MAX_SAMPLE_GAP_SECONDS
            )

    return total


# ============================================================ detector

class PoseActionDetector:
    """Stateful detector. Call update() with each pose result; it returns
    the list of confirmed actions right now:

        {'action': 'neck_hold' | 'kick', 'name': 'neck hold' | 'kick',
         'confidence': float, 'xyxy': [x1, y1, x2, y2]}

    `name`, `confidence` and `xyxy` match the weapon detections, so the
    existing box drawing code draws them unchanged.

    After each update: `people` is the number of distinct people in this
    frame, `scene` / `scene_people` the smoothed scene (see SCENE)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.neck_history = []     # (t, bool) two-handed grip seen
        self.neck_any_history = [] # (t, bool) any hold (one or two hands)
        self.neck_last = None      # (t, conf, box)
        self.kick_events = []      # (t, conf, box)
        self.kick_shape_since = None
        self.people_history = []   # (t, count)
        self.people = 0
        self.scene_people = 0
        self.scene = 'ISOLATED'
        self.last_reason = 'pose: waiting'
        self.last_update_t = None
        self.sample_gap = 0.0      # smoothed time between checks

    # ---- scene ----------------------------------------------------------

    def _update_scene(self, count, now):
        self.people = count
        self.people_history.append((now, count))
        cutoff = now - SCENE_WINDOW_SECONDS
        self.people_history = [h for h in self.people_history if h[0] >= cutoff]

        # Median, so one missed or doubled detection doesn't flip the label.
        self.scene_people = int(round(float(np.median(
            [c for _, c in self.people_history]
        ))))
        self.scene = scene_label(self.scene_people)

    # ---- tracking -------------------------------------------------------

    def _match_tracks(self, persons, now):
        """Greedy nearest-centre matching of people to existing tracks."""

        self.tracks = [
            t for t in self.tracks if now - t['t'] <= TRACK_TIMEOUT_SECONDS
        ]

        centres = [
            ((p['box'][0] + p['box'][2]) / 2.0, (p['box'][1] + p['box'][3]) / 2.0)
            for p in persons
        ]

        pairs = []

        for pi, c in enumerate(centres):
            limit = 0.9 * _size(persons[pi])
            for ti, track in enumerate(self.tracks):
                tc = track['centre']
                d = float(np.hypot(c[0] - tc[0], c[1] - tc[1]))
                if d <= limit:
                    pairs.append((d, pi, ti))

        pairs.sort()
        assigned, used = {}, set()

        for d, pi, ti in pairs:
            if pi in assigned or ti in used:
                continue
            assigned[pi] = ti
            used.add(ti)

        return assigned, centres

    # ---- main entry -----------------------------------------------------

    def update(self, persons, now):

        stats = {}
        n_raw = len(persons)

        if self.last_update_t is not None and now > self.last_update_t:
            gap = now - self.last_update_t
            self.sample_gap = gap if self.sample_gap == 0 else 0.7 * self.sample_gap + 0.3 * gap
        self.last_update_t = now

        persons = dedupe_people(persons)

        self._update_scene(len(persons), now)

        # ---------------- neck hold
        best_neck = (0.0, None)
        two_hands_seen = False

        for ai, a in enumerate(persons):
            for vi, v in enumerate(persons):
                if ai == vi or not _similar_size(a, v):
                    continue

                conf, two_hands = neck_hold_detail(a, v, stats)
                two_hands_seen = two_hands_seen or (conf > 0 and two_hands)

                if conf > best_neck[0]:
                    best_neck = (conf, _union_box(a['box'], v['box']))

        self.neck_history.append((now, two_hands_seen))
        self.neck_any_history.append((now, best_neck[0] > 0))

        cutoff = now - NECK_WINDOW_SECONDS
        self.neck_history = [h for h in self.neck_history if h[0] >= cutoff]
        cutoff = now - NECK_ONE_HAND_WINDOW_SECONDS
        self.neck_any_history = [h for h in self.neck_any_history if h[0] >= cutoff]

        if best_neck[0] > 0:
            self.neck_last = (now, best_neck[0], best_neck[1])

        neck_secs = _seconds_true(self.neck_history)
        neck_any_secs = _seconds_true(self.neck_any_history)
        neck_two_checks = sum(1 for _, f in self.neck_history if f)
        neck_any_checks = sum(1 for _, f in self.neck_any_history if f)
        neck_confirmed = (
            ((neck_secs >= NECK_MIN_SECONDS and neck_two_checks >= NECK_MIN_TRUE_CHECKS)
             or (neck_any_secs >= NECK_ONE_HAND_MIN_SECONDS and neck_any_checks >= NECK_ONE_HAND_MIN_CHECKS))
            and self.neck_last is not None
            and now - self.neck_last[0] <= NECK_WINDOW_SECONDS
        )

        # ---------------- kick
        assigned, centres = self._match_tracks(persons, now)

        new_tracks = []
        kick_this_frame = None

        for pi, p in enumerate(persons):

            prev = self.tracks[assigned[pi]] if pi in assigned else None
            others = [q for qi, q in enumerate(persons) if qi != pi]

            result = kick_check(p, others, prev, now, stats)

            if result is not None and (
                kick_this_frame is None or result[0] > kick_this_frame[0]
            ):
                kick_this_frame = result

            # Update / create this person's track.
            track = prev if prev is not None else {'id': self.next_id}
            if prev is None:
                self.next_id += 1

            hip_ref = _hip_ref(p['kpts'])
            rel = {}
            if hip_ref is not None:
                for ank_i in (L_ANK, R_ANK):
                    if _ok(p['kpts'], ank_i, KICK_KP_CONF):
                        rel[ank_i] = _pt(p['kpts'], ank_i) - hip_ref

            track['t'] = now
            track['centre'] = centres[pi]
            track['rel'] = rel
            new_tracks.append(track)

        # Tracks not seen this frame stay until they time out.
        kept = {t['id'] for t in new_tracks}
        new_tracks += [t for t in self.tracks if t['id'] not in kept]
        self.tracks = new_tracks

        # A kick shape that never goes away is a pose, not a kick.
        if kick_this_frame is not None:
            if self.kick_shape_since is None:
                self.kick_shape_since = now
        else:
            self.kick_shape_since = None

        held_too_long = (
            self.kick_shape_since is not None
            and now - self.kick_shape_since > KICK_MAX_HOLD_SECONDS
        )

        if kick_this_frame is not None and not held_too_long:
            self.kick_events.append((now, kick_this_frame[0], kick_this_frame[1]))

        kick_window = min(
            max(KICK_WINDOW_SECONDS, KICK_WINDOW_GAPS * self.sample_gap),
            KICK_WINDOW_MAX_SECONDS
        )
        kcut = now - kick_window
        self.kick_events = [e for e in self.kick_events if e[0] >= kcut]

        kick_confirmed = (
            len(self.kick_events) >= KICK_MIN_EVENTS and not held_too_long
        )

        self.last_reason = (
            f'pose: {len(persons)} people ({self.scene}) | neck 2-hand {neck_secs:.1f}s '
            f'(need {NECK_MIN_SECONDS:.1f}s), any {neck_any_secs:.1f}s '
            f'(need {NECK_ONE_HAND_MIN_SECONDS:.1f}s) | kicks {len(self.kick_events)} '
            f'(need {KICK_MIN_EVENTS})'
        )

        if DEBUG:
            ratio = stats.get('neck_ratio')
            print(
                f"[POSE DEBUG] t={now:.2f} people={n_raw} raw / {len(persons)} distinct "
                f"scene={self.scene}({self.scene_people}) "
                f"| neck: closest wrist {'-' if ratio is None else f'{ratio:.2f}'}x radius "
                f"(<=1 counts), held {neck_secs:.1f}s "
                f"| kick: {stats.get('kick_fail', 'no legs seen')}, "
                f"max leg angle {stats.get('kick_angle', 0.0):.0f} "
                f"(need {KICK_HIGH_ANGLE_DEG:.0f}), events {len(self.kick_events)}/{KICK_MIN_EVENTS}"
                + (" [held too long -> pose, not kick]" if held_too_long else "")
            )

        events = []

        if neck_confirmed:
            _, conf, box = self.neck_last
            events.append({
                'action': 'neck_hold', 'name': 'neck hold',
                'confidence': float(conf), 'xyxy': box,
            })

        if kick_confirmed:
            _, _, box = max(self.kick_events, key=lambda e: e[0])
            events.append({
                'action': 'kick', 'name': 'kick',
                'confidence': float(max(e[1] for e in self.kick_events)),
                'xyxy': box,
            })

        return events



# ============================================================ single image

# A photo has no time to confirm over, so a single-frame neck hold also
# needs: both people clearly seen (at least this many keypoints at
# KP_CONF -- a figure in a wall poster had 6-8, real people 9+), and the
# attacker's BOTH wrists raised to chest height or higher (people walking
# side by side, arms hanging, can line up with a neck by perspective),
# reaching toward the victim (not held by the attacker's own chin, as in a
# boxer's guard).
SINGLE_MIN_VISIBLE_KPTS = 9


def _visible(person):
    return int((person['kpts'][:, 2] >= KP_CONF).sum())


def _hands_raised(person):
    """Both wrists at or above the attacker's mid-chest line."""
    k = person['kpts']
    if not (_ok(k, L_WRI, KICK_KP_CONF) and _ok(k, R_WRI, KICK_KP_CONF)):
        return False
    shos = [k[i][1] for i in (L_SHO, R_SHO) if _ok(k, i, KICK_KP_CONF)]
    hips = [k[i][1] for i in (L_HIP, R_HIP) if _ok(k, i, KICK_KP_CONF)]
    if shos and hips:
        line = (np.mean(shos) + np.mean(hips)) / 2.0
    else:
        x1, y1, x2, y2 = person['box']
        line = y1 + 0.55 * (y2 - y1)
    return k[L_WRI][1] <= line and k[R_WRI][1] <= line


def _hands_reach_victim(attacker, victim):
    """Both hands closer to the victim's throat than to the attacker's own
    (boxers in a guard stance hold their hands by their own chin)."""
    own = _throat_line(attacker)
    vic = _throat_line(victim)
    if own is None or vic is None:
        return False
    k = attacker['kpts']
    for w, e in ((L_WRI, L_ELB), (R_WRI, R_ELB)):
        hand = _hand_point(k, w, e)
        if _dist_to_segment(hand, vic[0], vic[1]) >= _dist_to_segment(hand, own[0], own[1]):
            return False
    return True


def single_frame_actions(persons):
    """Neck hold / kick in ONE image (photo uploads), where there is no
    time to confirm over. Only the strong single-frame signs count:
      - a TWO-handed grip on another person's neck (one hand isn't enough
        without the video's duration check);
      - a leg raised like a kick (KICK_HIGH_ANGLE_DEG) with the foot on
        another person.
    Returns the same event dicts as PoseActionDetector.update()."""

    persons = dedupe_people(persons)
    events = []

    best_neck = None
    for ai, a in enumerate(persons):
        for vi, v in enumerate(persons):
            if ai == vi or not _similar_size(a, v):
                continue
            if (_visible(a) < SINGLE_MIN_VISIBLE_KPTS
                    or _visible(v) < SINGLE_MIN_VISIBLE_KPTS
                    or not _hands_raised(a)
                    or not _hands_reach_victim(a, v)):
                continue
            conf, two_hands = neck_hold_detail(a, v)
            if conf > 0 and two_hands and (best_neck is None or conf > best_neck[0]):
                best_neck = (conf, _union_box(a['box'], v['box']))

    if best_neck is not None:
        events.append({'action': 'neck_hold', 'name': 'neck hold',
                       'confidence': float(best_neck[0]), 'xyxy': best_neck[1]})

    best_kick = None
    for pi, p in enumerate(persons):
        others = [q for qi, q in enumerate(persons) if qi != pi]
        result = kick_check(p, others, None, 0.0)
        if result is not None and (best_kick is None or result[0] > best_kick[0]):
            best_kick = result

    if best_kick is not None:
        events.append({'action': 'kick', 'name': 'kick',
                       'confidence': float(best_kick[0]), 'xyxy': best_kick[1]})

    return events

# Printed once when the app starts, so the terminal shows which rules are
# loaded (the app must be restarted after this file changes).
print(
    f"[POSE] Rules loaded: neck hold = two hands {NECK_MIN_SECONDS:.1f}s "
    f"or one hand {NECK_ONE_HAND_MIN_SECONDS:.1f}s; kick = {KICK_MIN_EVENTS} "
    f"kick frames in {KICK_WINDOW_SECONDS:.1f}s"
)
