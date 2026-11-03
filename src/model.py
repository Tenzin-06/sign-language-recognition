"""LSTM/GRU model for WLASL100 sign recognition."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from features import SEQ_LEN, N_FEATURES, normalize

# Final configuration chosen in Phase 3 (hands-only, normalized, 3-seed mean ~45% val top-1)
FINAL_MODEL_KW = dict(act='tanh', units=(128, 256, 128), dropout=0.5, dense=(128,))
FINAL_AUGMENT_COPIES = 8
HAND_SLICE = slice(132, 258)      # columns of the left + right hand inside the 258-vector


def build_model(n_classes, units=(64, 128, 64), act='relu', dropout=0.0, dense=(64, 32),
                bidirectional=False, rnn='lstm', lr=1e-3, label_smoothing=0.0,
                input_shape=(SEQ_LEN, N_FEATURES)):
    """Stacked recurrent classifier.

    Defaults reproduce the original baseline (LSTM 64-128-64, relu, Dense 64-32).
    The last recurrent layer returns one vector for the whole clip; the earlier
    ones return full sequences so the next layer can keep reading over time.
    """
    Rnn = layers.LSTM if rnn == 'lstm' else layers.GRU
    m = models.Sequential([layers.Input(shape=input_shape)])
    for i, u in enumerate(units):
        last = (i == len(units) - 1)
        layer = Rnn(u, return_sequences=not last, activation=act)
        m.add(layers.Bidirectional(layer) if bidirectional else layer)
        if dropout:
            m.add(layers.Dropout(dropout))
    for d in dense:
        m.add(layers.Dense(d, activation='relu'))
        if dropout:
            m.add(layers.Dropout(dropout))
    m.add(layers.Dense(n_classes, activation='softmax'))
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['categorical_accuracy',
                 tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')])
    return m


def prepare_input(seq_258, feature_slice=HAND_SLICE):
    """Turn raw keypoints (32, 258) into model input.

    Order matters: normalize on all 258 values first (it needs the shoulders from
    the pose), THEN keep only the columns the model was trained on.
    Used by the evaluation notebook and by the real-time / Streamlit app.
    """
    x = normalize(np.asarray(seq_258, dtype=np.float32))
    return x[:, feature_slice]