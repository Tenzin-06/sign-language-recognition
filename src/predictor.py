import json, os
import numpy as np
import tensorflow as tf
from features import normalize


class SignPredictor:
    def __init__(self, models_dir='models/v1'):
        self.cfg = json.load(open(os.path.join(models_dir, 'model_config.json')))
        self.actions = np.load(os.path.join(models_dir, 'actions.npy'), allow_pickle=True)
        self.model = tf.keras.models.load_model(os.path.join(models_dir, 'best_model.keras'))
        a, b = self.cfg['feature_slice']
        self.cols = slice(a, b)
        self.seq_len = int(self.cfg['seq_len'])
        self.landmarker_path = os.path.join(models_dir, 'holistic_landmarker.task')

    def _prep(self, seq_258):
        x = np.asarray(seq_258, dtype=np.float32)
        if self.cfg.get('normalize', True):
            x = normalize(x)               # normalize on all 258 values first (needs the shoulders)
        return x[:, self.cols][None]       # then keep only the columns the model was trained on

    def predict_probs(self, seq_258):
        return self.model(self._prep(seq_258), training=False).numpy()[0]

    def top_k(self, probs, k=3):
        idx = np.argsort(probs)[::-1][:k]
        return [(str(self.actions[i]), float(probs[i])) for i in idx]