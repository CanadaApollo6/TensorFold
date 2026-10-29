"""Predeclared token-ID modulo-eight frequency test and paired speed confidence intervals."""
import math
import random
import statistics


def distribution(records):
    expected = {(p, row, v) for p in range(3) for row in [1, 4, 7] for v in range(4)}
    if {(r['prompt'], r['row'], r['variant']) for r in records} != expected or len(records) != 36:
        raise ValueError('missing fixed-position sampling cells')
    results = []
    for r in records:
        a, b = r['counts']
        n = r['draws']
        if n < 8192 or len(a) != 8 or len(b) != 8 or sum(a) != n or sum(b) != n:
            raise ValueError('invalid sampling counts')
        # A union Hoeffding bound controls all 288 bin differences at family alpha 0.01.
        radius = 2 * math.sqrt(math.log(4 * 36 * 8 / .01) / (2 * n))
        difference = max(abs(x - y) / n for x, y in zip(a, b))
        upper = difference + radius
        if upper >= .10:
            raise ValueError(f'sampled distributions fail equivalence margin 0.10: upper={upper:.4f}')
        results.append(dict(prompt=r['prompt'], row=r['row'], variant=r['variant'], max_bin_difference=difference,
                            simultaneous_upper=upper))
    return dict(test='two-sample simultaneous Hoeffding equivalence on token_id % 8 bins', alpha=.01,
                draws_per_side=8192, cells=results, equivalence_margin=.10,
                minimum_detectable_bin_difference=.10, rejection_power_at_margin_lower_bound=.99,
                power_at_equal_distributions_lower_bound=1 - 4 * 36 * 8 * math.exp(-8192 * (.10 - radius)**2 / 2),
                limitation='eight predeclared frequency bins; not proof of token-level distribution identity')


def speed(ratios):
    if len(ratios) < 5 or any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError('five positive speed ratios required')
    rng = random.Random(1937)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return dict(median=statistics.median(ratios), spread=[min(ratios), max(ratios)],
                interval_95=[boot[250], boot[9749]], method='paired median percentile bootstrap, 10000 resamples')
