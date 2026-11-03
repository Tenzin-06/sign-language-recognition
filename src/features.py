"""Shared feature code for training (Colab) and the real-time/Streamlit app."""
import numpy as np
import pandas as pd

SEQ_LEN = 32
N_FEATURES = 258   # pose 33*4 + left hand 21*3 + right hand 21*3


# ---------- MediaPipe landmarker ----------
def make_landmarker(model_path):
    import mediapipe as mp
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5)
    return mp.tasks.vision.HolisticLandmarker.create_from_options(options)


# ---------- keypoint extraction ----------
def lm_array(lms, n, with_vis):
    """Flatten a landmark list into a vector; zeros if not detected."""
    if lms and isinstance(lms[0], (list, tuple)):
        lms = lms[0]
    if not lms or len(lms) != n:
        return np.zeros(n * (4 if with_vis else 3))
    if with_vis:
        return np.array([[l.x, l.y, l.z, l.visibility or 0.0] for l in lms]).flatten()
    return np.array([[l.x, l.y, l.z] for l in lms]).flatten()


def extract_keypoints(result):
    """HolisticLandmarkerResult -> vector of 258 values."""
    pose = lm_array(result.pose_landmarks, 33, True)
    lh = lm_array(result.left_hand_landmarks, 21, False)
    rh = lm_array(result.right_hand_landmarks, 21, False)
    return np.concatenate([pose, lh, rh])


# ---------- frame selection ----------
def select_frames(frames, start, end, n=SEQ_LEN):
    """Trim to the sign (frame_start/frame_end) and sample n evenly spaced frames."""
    total = len(frames)
    s = int(start) - 1 if pd.notna(start) and start > 0 else 0    # frame_start is 1-indexed
    e = int(end) - 1 if pd.notna(end) and end > 0 else total - 1  # end <= 0 means "to the end"
    s = max(0, min(s, total - 1))
    e = max(s, min(e, total - 1))
    idx = np.linspace(s, e, n).astype(int)
    return [frames[i] for i in idx]


# ---------- normalization ----------
def normalize(feats):
    """feats: (T, 258). Center on the shoulder midpoint, scale by shoulder width."""
    f = feats.astype(np.float32).copy()
    pose = f[:, :132].reshape(-1, 33, 4).copy()
    ls, rs = pose[:, 11, :3], pose[:, 12, :3]               # left / right shoulder
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1)
    scale[scale < 1e-3] = 1.0                                # pose missing -> no scaling

    def adjust(block, n_points, n_vals):
        pts = block.reshape(-1, n_points, n_vals).copy()
        xyz = pts[:, :, :3]
        valid = (xyz != 0).any(axis=2, keepdims=True)         # keep missing points at zero
        pts[:, :, :3] = np.where(valid, (xyz - center[:, None, :]) / scale[:, None, None], 0)
        return pts.reshape(len(block), -1)

    return np.concatenate([adjust(f[:, :132], 33, 4),
                           adjust(f[:, 132:195], 21, 3),
                           adjust(f[:, 195:258], 21, 3)], axis=1)