"""Real-time sign recognition. Run from the repo root:  python app/realtime_demo.py"""
import sys, time, collections
import cv2, numpy as np, mediapipe as mp
sys.path.append('src')
from features import make_landmarker, extract_keypoints
from predictor import SignPredictor

MODEL_DIR = 'models/v1'
WINDOW_SEC = 2.0         # length of video fed to the model (a WLASL clip is about 1-2 s)
PREDICT_EVERY = 0.4      # seconds between predictions
MIN_HAND_FRAC = 0.30     # share of the window's frames that must contain a hand
VOTE_N, VOTE_MIN = 5, 3  # a label is shown only if it won >= VOTE_MIN of the last VOTE_N predictions
MIN_CONF = 0.30          # averaged probability needed to show a label (confidence is unreliable, so keep it low)


def hand_present(kp):
    return bool(np.any(kp[132:258]))


def draw_hands(img, kp):
    h, w = img.shape[:2]
    for x, y, _ in kp[132:258].reshape(-1, 3):
        if x or y:
            cv2.circle(img, (int((1 - x) * w), int(y * h)), 3, (0, 255, 0), -1)   # preview is mirrored, so use 1 - x


def main():
    pred = SignPredictor(MODEL_DIR)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit('Cannot open the webcam. Close other apps using it, or try index 1.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    buf = collections.deque()                      # (time, keypoints)
    votes = collections.deque(maxlen=VOTE_N)
    probs_hist = collections.deque(maxlen=VOTE_N)
    shown, top3 = '', []
    t0, last_pred, last_ts = time.time(), 0.0, -1

    with make_landmarker(pred.landmarker_path) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.time() - t0
            ts = max(int(t * 1000), last_ts + 1); last_ts = ts            # timestamps must keep increasing
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # the frame is NOT mirrored
            result = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            kp = extract_keypoints(result)
            buf.append((t, kp))
            while buf and t - buf[0][0] > WINDOW_SEC:
                buf.popleft()

            span = buf[-1][0] - buf[0][0]
            hand_frac = float(np.mean([hand_present(k) for _, k in buf]))
            status = ''
            if hand_frac < MIN_HAND_FRAC:
                votes.clear(); probs_hist.clear(); shown, top3 = '', []
                status = 'Show your hands and sign'
            elif span < 0.8 * WINDOW_SEC:
                status = 'Warming up...'
            elif t - last_pred >= PREDICT_EVERY:
                last_pred = t
                kps = np.stack([k for _, k in buf])
                seq = kps[np.linspace(0, len(kps) - 1, pred.seq_len).astype(int)]   # 32 evenly spaced frames
                probs = pred.predict_probs(seq)
                probs_hist.append(probs); votes.append(int(probs.argmax()))
                avg = np.mean(probs_hist, axis=0)
                top3 = pred.top_k(avg, 3)
                top = int(np.argmax(avg))
                shown = str(pred.actions[top]) if (votes.count(top) >= VOTE_MIN and avg[top] >= MIN_CONF) else ''

            disp = cv2.flip(frame, 1)                                 # mirror only the preview
            draw_hands(disp, kp)
            if status:
                cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.putText(disp, shown if shown else '...', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            for i, (name, p) in enumerate(top3):
                cv2.putText(disp, f'{i + 1}. {name} {p:.2f}', (10, 80 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Sign recognition (q to quit)', disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()