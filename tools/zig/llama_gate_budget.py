"""Conservative runtime admission calibrated from the first real KV job."""
import math


def estimate(job, rates=None):
    rates = rates or dict(prefill=240, decode=35)
    label = job['label']
    if label.startswith('p4-'):
        length = int(label.split('-')[1])
        # Attention grows with history; the measured 60K rate is not extrapolated as a constant at 100K.
        scale = max(1, length / 60000)
        return 75 + 2 * length / rates['prefill'] * scale + 3 * 1024 / rates['decode'] + 1270 / rates['decode'] * (scale + 2)
    if label.endswith('-wire') or label == 'p2':
        return 150
    if label.startswith('p3-'):
        return 90 + (3 * 5 * 2 * 255) / rates['decode'] + 240
    if label == 'p1-reference':
        return 180
    if label == 'p1-speed':
        return 60 + 10 * 256 / rates['decode']
    return 60


def calibrate(result):
    prefill = result.get('prefill', [])
    runs = result.get('runs', [])
    if len(prefill) != 2 or not runs:
        raise ValueError('first job did not provide prefill and decode calibration')
    rates = dict(prefill=min(r['prefill_tok_s'] for r in prefill), decode=min(r['decode_tok_s'] for r in runs))
    if any(not math.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError('invalid calibration rates')
    # A 25 percent runtime reserve covers loading and timing variability, not just ideal bandwidth.
    return {k: v / 1.25 for k, v in rates.items()}
