import sys, time
import cv2, numpy as np, mediapipe as mp
sys.path.append('src')
from features import make_landmarker, extract_keypoints

cap = cv2.VideoCapture(0)
t0, n, last_ts, hand_frames, times = time.time(), 0, -1, 0, []
with make_landmarker('models/v1/holistic_landmarker.task') as lm:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        ts = max(int((time.time() - t0) * 1000), last_ts + 1); last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t1 = time.time()
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        times.append(time.time() - t1)
        kp = extract_keypoints(res)
        n += 1; hand_frames += bool(kp[132:258].any())

        disp = cv2.flip(frame, 1); h, w = disp.shape[:2]
        for x, y, _ in kp[132:258].reshape(-1, 3):
            if x or y:
                cv2.circle(disp, (int((1 - x) * w), int(y * h)), 3, (0, 255, 0), -1)
        cv2.putText(disp, f'keypoints: {kp.shape[0]}  hands seen: {bool(kp[132:258].any())}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('landmarks (q to quit)', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release(); cv2.destroyAllWindows()
print(f'frames: {n} | hands detected in {hand_frames / max(n, 1):.0%} of frames | '
      f'landmarker time per frame: {np.mean(times) * 1000:.0f} ms')