"""Keypoint-sequence augmentation (applied to NORMALIZED 258-value frames)."""
import numpy as np

N_POSE, N_HAND = 33, 21


def coord_indices():
    """Column indices of all x, y and z values in the 258-vector.

    Layout: pose 33x(x,y,z,visibility) = 132, then left hand 21x(x,y,z) = 63,
    then right hand 21x(x,y,z) = 63. Visibility columns are excluded.
    Each returned array has 75 entries (33 pose + 42 hand points).
    """
    pose = np.arange(N_POSE * 4).reshape(N_POSE, 4)
    hands = np.arange(N_POSE * 4, N_POSE * 4 + 2 * N_HAND * 3).reshape(2 * N_HAND, 3)
    xi = np.concatenate([pose[:, 0], hands[:, 0]])
    yi = np.concatenate([pose[:, 1], hands[:, 1]])
    zi = np.concatenate([pose[:, 2], hands[:, 2]])
    return xi, yi, zi


XI, YI, ZI = coord_indices()
ALL_COORDS = np.concatenate([XI, YI, ZI])


def augment(seq, rng):
    """Return one randomly modified copy of a (T, 258) sequence.

    Transforms: time warp (random crop + resample), scale, small rotation in the
    x-y plane, shift, and noise. Missing landmarks (zeros) stay zero.
    """
    T = len(seq)
    a, b = rng.integers(0, 4), T - 1 - rng.integers(0, 4)
    s = seq[np.linspace(a, b, T).astype(int)].copy()        # time warp
    present = (s != 0)                                       # remember which values are real

    s[:, ALL_COORDS] *= rng.uniform(0.9, 1.1)                # scale

    ang = np.deg2rad(rng.uniform(-10, 10))                   # rotation
    x, y = s[:, XI].copy(), s[:, YI].copy()
    s[:, XI] = x * np.cos(ang) - y * np.sin(ang)
    s[:, YI] = x * np.sin(ang) + y * np.cos(ang)

    s[:, XI] += rng.uniform(-0.1, 0.1)                       # shift
    s[:, YI] += rng.uniform(-0.1, 0.1)

    s[:, ALL_COORDS] += rng.normal(0, 0.005, size=(T, len(ALL_COORDS)))   # noise
    return (s * present).astype('float32')                   # missing points stay zero


def make_augmented(X, y, copies=4, seed=0):
    """Original data plus `copies` augmented versions of every sequence."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(copies):
        Xs.append(np.stack([augment(x, rng) for x in X]))
        ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)