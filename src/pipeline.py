import cv2, numpy as np, mediapipe as mp
from features import make_landmarker, extract_keypoints, select_frames, SEQ_LEN


def read_video(path, max_frames=1500):
    """Return (list of BGR frames, fps)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def frames_to_keypoints(frames, landmarker_path):
    feats = []
    with make_landmarker(landmarker_path) as lm:
        for i, f in enumerate(frames):
            rgb = np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), i * 40)
            feats.append(extract_keypoints(res))
    return np.array(feats)


def clip_sequence(frames, landmarker_path, seq_len=SEQ_LEN):
    """One clip -> (seq_len, 258) keypoints, sampled the same way as in training."""
    sel = select_frames(frames, np.nan, np.nan, n=seq_len)
    return frames_to_keypoints(sel, landmarker_path)


def hand_fraction(seq):
    """Share of frames with at least one detected hand."""
    return float(np.mean(seq[:, 132:258].any(axis=1)))