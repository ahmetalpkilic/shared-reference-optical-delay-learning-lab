"""Common covariance estimate; shared physical channel labels, seconds."""
import numpy as np
from model import SCALE, lag


def covariance(samples):
    """Unbiased covariance from independent reset observations, NOT an AR trace."""
    x = np.asarray(samples, float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('Need >=3 finite two-channel calibration observations')
    x = x - x.mean(axis=0)
    return x.T @ x / (len(x)-1)


def inverse_root(c):
    c = np.asarray(c, float)
    if c.shape != (2, 2) or not np.isfinite(c).all() or not np.allclose(c, c.T):
        raise ValueError('Need symmetric finite 2x2 covariance')
    d, q = np.linalg.eigh(c)
    if np.min(d) <= 0:
        raise ValueError('Covariance must be positive definite; no hidden ridge')
    return (q / np.sqrt(d)) @ q.T


def matrix_score(old, pilot, c):
    # lag divides by SCALE, so multiply back to dimensional covariance (s^2).
    delta = (lag(pilot) - lag(old)) * SCALE**2
    w = inverse_root(c)
    return float(np.linalg.norm(w @ delta @ w.T, 'fro'))


def calibration(rng, size):
    # Each row is a fresh independent reset. Mean is unknown and subtracted.
    return covariance(rng.normal(0, SCALE, size=(size, 2)))
