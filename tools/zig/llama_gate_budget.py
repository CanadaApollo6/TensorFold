"""Whole-container admission with per-class first-unit wall-time calibration."""
import math

SETUP_SECONDS = 1500
SYNC_SECONDS = 180


def job_class(job):
    label = job['label']
    if label.startswith(('prefill-speed-', 'profile-prefill-')): return 'prefill-speed'
    if label.startswith('prefill-'): return 'prefill-equivalence'
    if label.startswith('p4-'): return 'kv'
    if label.startswith('p4ref-exl3-'): return 'kv-reference'
    if label.endswith('-wire'): return 'draft-wire'
    if label == 'p2': return 'p2-wire'
    if label.startswith('p3-'): return 'draft-speed' if label.endswith('-speed') else 'draft-basic'
    return label


def units(job):
    label = job['label']
    if label.startswith('p4ref-exl3-'):
        return int(label.split('-')[2]) / 20000
    if label.startswith('p4-'):
        length = int(label.split('-')[1])
        return (75 + 2 * (length / 2000 + length * length / 60_000_000) + 2294 / 45) / 306
    if label.startswith(('prefill-speed-', 'profile-prefill-')):
        return int(label.split('-')[2]) / 20000
    if label.startswith('prefill-'):
        return int(label.split('-')[1]) / 2000
    return 1


def estimate(job, rates=None):
    kind = job_class(job)
    if rates and kind in rates: return rates[kind] * units(job)
    # Reference planning covers five prefills and 3200 single-row forwards; calibrate separately from TF timings.
    base = {'kv': 306, 'kv-reference': 600, 'prefill-speed': 180, 'prefill-equivalence': 120, 'p1-prefill': 120, 'p2-wire': 150, 'draft-wire': 150, 'draft-basic': 180, 'draft-speed': 300,
            'kv4-unit': 120, 'exl3-r32-unit': 120, 'draft-lattice-unit': 60, 'draft-rope-unit': 60, 'draft-profile': 240, 'p1-reference': 300, 'p1-logits': 120, 'p1-greedy': 120, 'p1-speed': 900}
    return base.get(kind, 180) * units(job)


def calibrate(job, seconds):
    if not math.isfinite(seconds) or seconds <= 0: raise ValueError('invalid calibration time')
    return seconds * 1.25 / units(job)
