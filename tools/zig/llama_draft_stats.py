"""Paired speed confidence intervals."""
import math
import random
import statistics


def speed(ratios):
    if len(ratios) < 5 or any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError('five positive speed ratios required')
    rng = random.Random(1937)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return dict(median=statistics.median(ratios), spread=[min(ratios), max(ratios)],
                interval_95=[boot[250], boot[9749]], method='paired median percentile bootstrap, 10000 resamples')
