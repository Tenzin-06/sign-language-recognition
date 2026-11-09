"""Run from anywhere:  streamlit run app/streamlit_app.py"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')    # hide TensorFlow info lines
os.environ.setdefault('GLOG_minloglevel', '2')         # hide MediaPipe info lines

import sys, tempfile, threading, time, traceback
from datetime import datetime
from pathlib import Path

import av, cv2, numpy as np, pandas as pd, streamlit as st
import tensorflow as tf
try:                                                   # keep TensorFlow from hogging every CPU core
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except (RuntimeError, ValueError):                     # Streamlit re-runs the script; this can only be set once
    pass
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
from predictor import SignPredictor
from recognizer import StreamRecognizer
from pipeline import read_video, clip_sequence, hand_fraction

MODEL_DIR = ROOT / 'models' / 'v1'
RTC_CONFIG = {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---- tuning knobs for the live webcam page ----
PREDICT_EVERY = 0.3                                    # seconds between classifier runs
REC_KW = dict(window_sec=1.5,                          # seconds of video the classifier looks at
              vote_n=4, vote_min=2,                    # label must win vote_min of the last vote_n predictions
              min_conf=0.25,                           # averaged probability needed to show a label
              min_hand_frac=0.30,                      # share of the window that must contain a hand
              max_side=384)                            # frames are shrunk to this size before landmarks

st.set_page_config(page_title='Sign Language Recognition', page_icon='🤟', layout='wide')


@st.cache_resource(show_spinner='Loading model...')
def load_predictor():
    p = SignPredictor(str(MODEL_DIR))
    p.predict_probs(np.zeros((p.seq_len, 258), dtype=np.float32))     # warm-up: first call is slow
    return p


def add_history(source, sign, conf):
    h = st.session_state.setdefault('history', [])
    h.insert(0, dict(time=datetime.now().strftime('%H:%M:%S'), source=source, sign=sign,
                     confidence=round(float(conf), 2)))
    del h[50:]                                         # keep the 50 most recent


# ---------------- webcam processor ----------------
class SignProcessor(VideoProcessorBase):
    """recv() only shows the video (fast). A worker thread extracts landmarks from the newest frame,
    and a second thread runs the classifier, so neither can slow the picture down."""

    def __init__(self, predictor):
        self.pred = predictor
        self.show_dots = True
        self._lock = threading.Lock()
        self._frame, self._frame_id, self._kp = None, 0, None
        self._pred_error = ''
        self._result = dict(status='Loading models...', label='', conf=0.0, top3=[], error='',
                            ms=0.0, fps=0.0, pred_ms=0.0, hand_frac=0.0)
        self._last_recv = time.time()
        self._thread = None

    def _ensure_worker(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        rec, pt, stop = None, None, threading.Event()
        try:
            rec = StreamRecognizer(self.pred, **REC_KW)

            def predict_loop():                         # slow path (classifier), its own thread
                while not stop.is_set():
                    try:
                        rec.predict_step()
                        self._pred_error = ''
                    except Exception as e:
                        self._pred_error = f'{type(e).__name__}: {e}'
                    stop.wait(PREDICT_EVERY)

            pt = threading.Thread(target=predict_loop, daemon=True)
            pt.start()

            seen = -1
            while time.time() - self._last_recv < 30:   # ends 30 s after the last frame (recv restarts it)
                with self._lock:
                    frame, fid = self._frame, self._frame_id
                if frame is None or fid == seen:
                    time.sleep(0.005)
                    continue
                seen = fid                              # always analyse only the NEWEST frame
                t1, kp, error = time.time(), None, ''
                try:
                    kp = rec.process(frame)
                except Exception as e:
                    error = f'{type(e).__name__}: {e}'
                ms = (time.time() - t1) * 1000
                with self._lock:
                    if kp is not None:
                        self._kp = kp
                    self._result = {**rec.state(), 'error': error or self._pred_error, 'ms': ms}
        except Exception:
            err = traceback.format_exc()
            print(err)
            with self._lock:
                self._result = dict(status='Analysis stopped', label='', conf=0.0, top3=[],
                                    error=err.strip().splitlines()[-1], ms=0.0, fps=0.0,
                                    pred_ms=0.0, hand_frac=0.0)
        finally:
            stop.set()
            if pt is not None:
                pt.join(timeout=2)
            if rec is not None:
                try:
                    rec.close()
                except Exception:
                    pass

    def recv(self, frame):
        self._ensure_worker()
        img = frame.to_ndarray(format='bgr24')          # NOT mirrored: this is what the model sees
        with self._lock:
            self._frame, self._frame_id = img, self._frame_id + 1
            self._last_recv = time.time()
            kp, r = self._kp, dict(self._result)
        disp = cv2.flip(img, 1)                         # mirror only what the user sees
        if kp is not None and self.show_dots:
            h, w = disp.shape[:2]
            for x, y, _ in kp[132:258].reshape(-1, 3):
                if x or y:
                    cv2.circle(disp, (int((1 - x) * w), int(y * h)), 3, (0, 255, 0), -1)
        cv2.putText(disp, r['label'] or r['status'], (10, 40), FONT, 1.0, (0, 255, 0), 2)
        if r['error']:
            cv2.putText(disp, r['error'][:60], (10, 75), FONT, 0.5, (0, 0, 255), 1)
        return av.VideoFrame.from_ndarray(disp, format='bgr24')

    def get_result(self):
        with self._lock:
            return dict(self._result)


# ---------------- pages ----------------
def home_page(pred):
    st.title('🤟 Sign Language Recognition')
    st.write('Recognizes **100 American Sign Language words** from video, using MediaPipe hand/body '
             'landmarks and an LSTM trained on the WLASL100 dataset.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Signs supported', len(pred.actions))
    if 'test_top1' in pred.cfg:
        c2.metric('Test top-1 accuracy', f"{pred.cfg['test_top1']:.0%}")
        c3.metric('Test top-5 accuracy', f"{pred.cfg['test_top5']:.0%}")
    st.subheader('How to use')
    st.markdown(
        '1. Open **Webcam** and press *START*, allow the camera.\n'
        '2. Sit so your **shoulders and both hands** are visible, with good light.\n'
        '3. Perform one sign, then pause briefly. The top 3 guesses are shown.\n'
        '4. Or open **Upload video** and analyse a short clip of one sign.\n'
        '5. Open **History** to see recent predictions.')
    st.warning('This is a research prototype. The model is right on the first guess about 1 time in 3 '
               'on new videos, and the correct sign is within the top 5 about 2 times in 3. '
               'Results depend on camera, lighting, distance and signing speed.')
    with st.expander(f'All {len(pred.actions)} supported signs'):
        st.write(', '.join(sorted(map(str, pred.actions))))


def webcam_page(pred):
    st.header('Live webcam recognition')
    st.caption('Click START and allow the camera in your browser. If the video stays black, check the '
               'browser camera permission and close other apps that use the camera.')
    show_dots = st.sidebar.checkbox('Show landmarks on video', value=True)
    diag = st.sidebar.empty()
    ctx = webrtc_streamer(
        key='sign-webcam', mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: SignProcessor(pred),
        media_stream_constraints={'video': {'width': {'ideal': 480}, 'height': {'ideal': 360},
                                            'frameRate': {'ideal': 20}}, 'audio': False},
        async_processing=True)

    box_label, box_top = st.empty(), st.empty()
    last = ('', 0.0)
    while ctx.state.playing:                            # refresh the panel while the stream runs
        proc = ctx.video_processor
        if proc is not None:
            proc.show_dots = show_dots
            r = proc.get_result()
            diag.caption(f"Landmarks: {r['fps']:.1f} fps ({r['ms']:.0f} ms/frame) | "
                         f"classifier: {r['pred_ms']:.0f} ms | hands seen in window: {r['hand_frac']:.0%}")
            if r['error']:
                box_label.error(f"Processing error: {r['error']}")
            elif r['label']:
                box_label.success(f"Recognized sign: **{r['label']}**  (confidence {r['conf']:.2f})")
                if r['label'] != last[0] or time.time() - last[1] > 3:
                    add_history('webcam', r['label'], r['conf']); last = (r['label'], time.time())
            else:
                box_label.info(r['status'])
            if r['top3']:
                box_top.table(pd.DataFrame(r['top3'], columns=['sign', 'probability'])
                              .round(2).set_index('sign'))
        time.sleep(0.3)
    if not ctx.state.playing:
        box_label.info('Press START to begin.')


def upload_page(pred):
    st.header('Recognize a video clip')
    f = st.file_uploader('Upload a short video (about 1-3 seconds, one sign)',
                         type=['mp4', 'mov', 'avi', 'webm', 'mkv'])
    mode = st.radio('Mode', ['Whole clip (one sign)', 'Sliding window (several signs)'], horizontal=True)
    if f is None:
        st.info('Upload a clip to begin.')
        return
    data = f.getvalue()
    st.video(data)
    if not st.button('Recognize', type='primary'):
        return

    with tempfile.NamedTemporaryFile(suffix=Path(f.name).suffix, delete=False) as tmp:
        tmp.write(data); path = tmp.name
    try:
        with st.spinner('Reading video and extracting landmarks...'):
            frames, fps = read_video(path)
            if len(frames) < 8:
                st.error('Could not read enough frames from this file. Try another video or format.')
                return
            if mode.startswith('Whole'):
                seq = clip_sequence(frames, pred.landmarker_path, pred.seq_len)
                hf = hand_fraction(seq)
                probs = pred.predict_probs(seq)
            else:
                win, step = int(2 * fps), int(1 * fps)
                starts = list(range(0, max(1, len(frames) - win + 1), step))[:15]
                rows = []
                for s in starts:
                    seq = clip_sequence(frames[s:s + win], pred.landmarker_path, pred.seq_len)
                    if hand_fraction(seq) < 0.3:
                        continue
                    top = pred.top_k(pred.predict_probs(seq), 1)[0]
                    rows.append(dict(start_s=round(s / fps, 1), sign=top[0], confidence=round(top[1], 2)))
    except Exception as e:
        st.error(f'Processing failed: {e}')
        return
    finally:
        os.unlink(path)

    if mode.startswith('Whole'):
        if hf < 0.1:
            st.warning('Almost no hands were detected, so the result is unreliable.')
        top5 = pred.top_k(probs, 5)
        st.success(f'Best guess: **{top5[0][0]}**  (confidence {top5[0][1]:.2f})')
        st.bar_chart(pd.DataFrame({'probability': [p for _, p in top5]}, index=[n for n, _ in top5]))
        add_history('upload', top5[0][0], top5[0][1])
    else:
        if not rows:
            st.warning('No windows with visible hands were found.')
            return
        st.table(pd.DataFrame(rows).set_index('start_s'))
        for r in rows:
            add_history('upload', r['sign'], r['confidence'])


def history_page():
    st.header('Prediction history')
    h = st.session_state.get('history', [])
    if h:
        st.dataframe(pd.DataFrame(h), width='stretch', hide_index=True)
    else:
        st.info('No predictions yet in this session.')
    if st.button('Clear history'):
        st.session_state['history'] = []
        st.rerun()


# ---------------- main ----------------
try:
    predictor = load_predictor()
except Exception as e:
    st.error(f'Could not load the model from {MODEL_DIR}: {e}')
    st.stop()

page = st.sidebar.radio('Navigation', ['Home', 'Webcam', 'Upload video', 'History'])
st.sidebar.caption(f"Model {predictor.cfg.get('model_version', 'v1')} - {len(predictor.actions)} signs")

if page == 'Home':
    home_page(predictor)
elif page == 'Webcam':
    webcam_page(predictor)
elif page == 'Upload video':
    upload_page(predictor)
else:
    history_page()