"""
Feature Extraction for ASL Hand Sign Recognition

SINGLE SOURCE OF TRUTH — used identically during training and inference.

Features (93-dimensional):
  A. 60D  Normalized landmark positions (landmarks 1-20, xyz)
  B. 10D  Joint angles (2 per finger × 5 fingers)
  C. 10D  All fingertip-pair distances
  D.  4D  Thumb tip → each other fingertip  (KEY for A/S/E/T confusion)
  E.  5D  Finger curl (tip-to-MCP distance per finger)
  F.  4D  Adjacent fingertip spread (index-middle, middle-ring, etc.)
"""

import numpy as np

# ─── MediaPipe Hand Landmark Indices ──────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_JOINTS = [
    [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],
    [INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP],
    [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
    [RING_MCP, RING_PIP, RING_DIP, RING_TIP],
    [PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP],
]

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

FINGER_MCP_TIP = [
    (THUMB_MCP, THUMB_TIP),
    (INDEX_MCP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_TIP),
    (RING_MCP, RING_TIP),
    (PINKY_MCP, PINKY_TIP),
]

# Adjacent fingertip pairs (for spread measurement)
ADJACENT_TIPS = [
    (THUMB_TIP, INDEX_TIP),
    (INDEX_TIP, MIDDLE_TIP),
    (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP),
]

# ─── Feature dimensions ──────────────────────────────────────────────────────
N_POSITION = 60   # 20 landmarks × 3 coords
N_ANGLES = 10     # 5 fingers × 2 angles
N_INTER_TIP = 10  # C(5,2) = 10 fingertip pairs
N_THUMB_TIP = 4   # thumb tip → index/middle/ring/pinky tips
N_CURL = 5        # per-finger curl (MCP→TIP distance)
N_SPREAD = 4      # adjacent fingertip distances
TOTAL_FEATURES = N_POSITION + N_ANGLES + N_INTER_TIP + N_THUMB_TIP + N_CURL + N_SPREAD  # 93


def _angle(p1, p2, p3):
    """Angle (radians) at vertex p2 formed by rays p2→p1 and p2→p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1, v2) / d, -1.0, 1.0)))


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def extract_features(hand_landmarks):
    """
    Extract normalised, position/scale invariant features.

    Accepts a (21, 3) numpy array OR a MediaPipe landmark object.
    Returns (93,) numpy array, or None on failure.
    """
    # ── Parse input ───────────────────────────────────────────────────────
    if hasattr(hand_landmarks, 'landmark'):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    else:
        pts = np.asarray(hand_landmarks, dtype=np.float64)

    if pts.shape != (21, 3):
        return None

    # ── Normalise ─────────────────────────────────────────────────────────
    centered = pts - pts[WRIST]
    hand_size = np.linalg.norm(centered[MIDDLE_MCP])
    if hand_size < 1e-6:
        return None
    n = centered / hand_size   # n[i] is normalised landmark i

    # A. Positions (60D) — landmarks 1-20 flattened
    position = n[1:].flatten()

    # B. Joint angles (10D) — 2 per finger
    angles = []
    for finger in FINGER_JOINTS:
        for i in range(len(finger) - 2):
            angles.append(_angle(n[finger[i]], n[finger[i + 1]], n[finger[i + 2]]))

    # C. All fingertip-pair distances (10D)
    inter_tip = []
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            inter_tip.append(_dist(n[FINGERTIPS[i]], n[FINGERTIPS[j]]))

    # D. Thumb tip → other fingertips (4D)
    #    Critical: separates A/S/E/T (all closed fists, thumb in different positions)
    thumb_tip = [_dist(n[THUMB_TIP], n[tip]) for tip in FINGERTIPS[1:]]

    # E. Finger curl (5D) — MCP-to-TIP distance per finger
    #    Short = curled, long = extended
    curl = [_dist(n[mcp], n[tip]) for mcp, tip in FINGER_MCP_TIP]

    # F. Adjacent fingertip spread (4D)
    #    Separates U/V (fingers together vs apart), W/other
    spread = [_dist(n[a], n[b]) for a, b in ADJACENT_TIPS]

    # ── Combine ───────────────────────────────────────────────────────────
    return np.concatenate([
        position,                # 60
        np.array(angles),        # 10
        np.array(inter_tip),     # 10
        np.array(thumb_tip),     #  4
        np.array(curl),          #  5
        np.array(spread),        #  4
    ])                           # = 93
