"""Independent categorical oracle and predeclared bug directions, with no production sampler imports."""
import math
import struct

MUTATIONS = ['residual_replaced', 'residual_reversed', 'draft_support', 'temperature_twice',
             'penalties_draft_only', 'within_bin_transfer']
DRAWS = 32768
FAMILY = 4 * 36 * (166912 + 7)
RADIUS = math.sqrt(math.log(2 * FAMILY / .01) / (2 * DRAWS))


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return struct.unpack('<f', struct.pack('<I', (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000))[0]


def probabilities(c, draft=False, temperature_twice=False, omit_penalties=False):
    logits = list(c['draft' if draft else 'target'])
    if not omit_penalties:
        p = c.get('penalties', {})
        from collections import Counter
        for t, count in Counter([*c.get('history', []), c['pending']]).items():
            logits[t] = bf16(bf16(logits[t]) - p.get('presence', 0) - count * p.get('frequency', 0))
    values = [bf16(v) for v in logits]
    rule = c['sampling']
    temperature = rule['temperature'] ** (2 if temperature_twice else 1)
    peak = max(values)
    weights = [math.exp((v - peak) / temperature) for v in values]
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    if rule['top_k']: order = order[:rule['top_k']]
    norm = sum(weights[i] for i in order)
    keep, cumulative = [], 0
    for i in order:
        keep.append(i)
        cumulative += weights[i] / norm
        if 0 < rule['top_p'] < 1 and cumulative >= rule['top_p']: break
    floor = max(weights) * rule.get('min_p', 0)
    keep = [i for i in keep if weights[i] >= floor]
    norm = sum(weights[i] for i in keep)
    result = [0.] * len(values)
    for i in keep: result[i] = weights[i] / norm
    return result


def alternatives(c):
    p, q = probabilities(c), probabilities(c, draft=True)
    reject = sum(max(pv - qv, 0) for pv, qv in zip(p, q))
    rev = sum(max(qv - pv, 0) for pv, qv in zip(p, q))
    residual = [min(pv, qv) + reject * pv for pv, qv in zip(p, q)]
    reversed_p = [min(pv, qv) + reject * max(qv - pv, 0) / rev if rev else pv for pv, qv in zip(p, q)]
    support = [pv if qv > 0 else 0 for pv, qv in zip(p, q)]
    z = sum(support)
    support = [v / z for v in support] if z else q
    transfer = list(p)
    top = max(range(len(p)), key=p.__getitem__)
    transfer[(top + 8) % len(p)] += transfer[top] * .5
    transfer[top] *= .5
    return dict(zip(MUTATIONS, [residual, reversed_p, support, probabilities(c, temperature_twice=True),
                               probabilities(c, omit_penalties=True), transfer]))


def evaluate(c, record):
    p = probabilities(c)
    counts = record['counts']
    if record['draws'] != DRAWS or len(counts) != len(p) or sum(counts) != DRAWS:
        raise ValueError('invalid replay shape')
    groups = [{i for i, (pv, bv) in enumerate(zip(p, bug)) if pv > bv}
              for bug in alternatives(c).values()]
    errors = [abs(n / DRAWS - pv) for n, pv in zip(counts, p)]
    errors += [abs(sum(counts[i] for i in group) / DRAWS - sum(p[i] for i in group)) for group in groups]
    worst = max(errors)
    return dict(state='PASS' if worst <= RADIUS else 'FAIL', max_error=worst, radius=RADIUS,
                accepted=record['accepted'], proposed=record['proposed'])
