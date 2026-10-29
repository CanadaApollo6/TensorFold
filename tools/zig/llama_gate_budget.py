"""Whole-container admission with per-class first-unit wall-time calibration."""
import math

SETUP_SECONDS = 1500
SYNC_SECONDS = 180


def job_class(job):
    label = job['label']
    if label.startswith('p4-'): return 'kv'
    if label.endswith('-wire') or label == 'p2': return 'wire'
    if label.startswith('p3-'): return 'draft-speed' if label.endswith('-speed') else 'draft-basic'
    return label


def units(job):
    label = job['label']
    if label.startswith('p4-'):
        length = int(label.split('-')[1])
        return (75 + 2 * length / 300 * max(1, length / 60000) + 2294 / 45) / (75 + 400 + 2294 / 45)
    return 1


def estimate(job, rates=None):
    kind = job_class(job)
    if rates and kind in rates: return rates[kind] * units(job)
    base = {'kv': 75 + 400 + 2294 / 45, 'wire': 150, 'draft-basic': 180, 'draft-speed': 300,
            'p1-reference': 300, 'p1-logits': 120, 'p1-greedy': 120, 'p1-speed': 900}
    return base.get(kind, 180) * units(job)


def calibrate(job, seconds):
    if not math.isfinite(seconds) or seconds <= 0: raise ValueError('invalid calibration time')
    return seconds * 1.25 / units(job)
