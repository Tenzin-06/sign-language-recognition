import time, threading, collections
import cv2, numpy as np, mediapipe as mp
from features import make_landmarker, extract_keypoints


class StreamRecognizer:
    """process(): landmarks for one frame (fast path).
    predict_step(): classify the recent window (slow path).
    The two can run in different threads."""

    def __init__(self, predictor, window_sec=1.5, min_hand_frac=0.30,
                 vote_n=4, vote_min=2, min_conf=0.25, max_side=384):
        self.p = predictor
        self.window_sec, self.min_hand_frac = window_sec, min_hand_frac
        self.vote_min, self.min_conf, self.max_side = vote_min, min_conf, max_side
        self.lm = make_landmarker(predictor.landmarker_path)
        self._lock = threading.Lock()
        self.buf = collections.deque()
        self.votes = collections.deque(maxlen=vote_n)
        self.probs_hist = collections.deque(maxlen=vote_n)
        self.ts_hist = collections.deque(maxlen=30)
        self.t0, self.last_ts = time.time(), -1
        self.status, self.label, self.conf, self.top3 = 'Warming up...', '', 0.0, []
        self.pred_ms, self.hand_frac = 0.0, 0.0

    def process(self, frame_bgr):
        """frame_bgr must NOT be mirrored. Returns this frame's 258 keypoints."""
        h, w = frame_bgr.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:                                         # smaller picture = faster landmarks
            frame_bgr = cv2.resize(frame_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        t = time.time() - self.t0
        self.ts_hist.append(t)
        ts = max(int(t * 1000), self.last_ts + 1); self.last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        kp = extract_keypoints(res)
        with self._lock:
            self.buf.append((t, kp))
            while self.buf and t - self.buf[0][0] > self.window_sec:
                self.buf.popleft()
        return kp

    def predict_step(self):
        with self._lock:
            items = list(self.buf)
        if not items:
            return
        span = items[-1][0] - items[0][0]
        hand_frac = self.hand_frac = float(np.mean([bool(k[132:258].any()) for _, k in items]))
        if hand_frac < self.min_hand_frac:
            self.votes.clear(); self.probs_hist.clear()
            self.label, self.conf, self.top3 = '', 0.0, []
            self.status = 'Show your hands and sign'
            return
        if span < 0.8 * self.window_sec:
            self.status = 'Warming up...'
            return
        self.status = 'Recognizing...'
        kps = np.stack([k for _, k in items])
        seq = kps[np.linspace(0, len(kps) - 1, self.p.seq_len).astype(int)]   # 32 evenly spaced frames
        t1 = time.time()
        probs = self.p.predict_probs(seq)
        self.pred_ms = (time.time() - t1) * 1000
        self.probs_hist.append(probs); self.votes.append(int(probs.argmax()))
        avg = np.mean(self.probs_hist, axis=0)
        top = int(np.argmax(avg))
        self.top3 = self.p.top_k(avg, 3)
        stable = self.votes.count(top) >= self.vote_min and avg[top] >= self.min_conf
        self.label = str(self.p.actions[top]) if stable else ''
        self.conf = float(avg[top]) if stable else 0.0

    def state(self):
        ts = list(self.ts_hist)
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0
        return dict(status=self.status, label=self.label, conf=self.conf, top3=list(self.top3),
                    fps=fps, pred_ms=self.pred_ms, hand_frac=self.hand_frac)

    def close(self):
        self.lm.close()