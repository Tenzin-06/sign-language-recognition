import sys
sys.path.insert(0, 'src')

import numpy as np
from predictor import SignPredictor

sys.path.append('src')

pred = SignPredictor('models/v1')

d = np.load('data/processed_val.npz', allow_pickle=True)

hits = 0

for x, y in zip(d['X'], d['y']):
    prediction = pred.predict_probs(x).argmax()

    if prediction == y:
        hits += 1

accuracy = hits / len(d['y'])

print("Local validation accuracy:", round(accuracy, 3))
print("Number of signs:", len(pred.actions))
print("First 15 signs:", list(pred.actions)[:15])