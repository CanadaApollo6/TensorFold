"""Replay cached target and draft logits through the CPU production adapter and lane engine."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
from llama_sample_oracle import MUTATIONS, DRAWS, RADIUS, alternatives, probabilities, evaluate


def fixtures():
    result = []
    for context in range(3):
        target = [round(-.12 * i + .3 * math.sin(i + context), 3) for i in range(64)]
        draft = [round(-.10 * i + .9 * math.sin(i + 1 + context), 3) for i in range(64)]
        for t, k, p, penalty in [(.3, 0, 1, 0), (.7, 3, .8, 0), (1, 20, .95, 0), (1.3, 50, .9, 1)]:
            result.append(dict(target=target, draft=draft, pending=context, draws=DRAWS,
                               sampling=dict(seed=0, temperature=t, top_k=k, top_p=p),
                               penalties=dict(presence=.7 * penalty, frequency=.3 * penalty)))
    return result


def replay(runner, cases):
    with tempfile.TemporaryDirectory(prefix='llama-replay-') as tmp:
        file, out = Path(tmp) / 'cases.json', Path(tmp) / 'counts.jsonl'
        file.write_text(json.dumps(cases))
        subprocess.run([str(runner.resolve()), file, out], check=True)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        if len(rows) != len(cases): raise ValueError('missing replay cells')
        return rows


def check(runner, cases):
    rows = replay(runner, cases)
    results = [evaluate(c, r) for c, r in zip(cases, rows)]
    if any(r['state'] != 'PASS' for r in results): raise ValueError(f'sampling oracle failed: {results}')
    if not sum(r['accepted'] for r in rows) or not sum(r['proposed'] - r['accepted'] for r in rows):
        raise ValueError('replay did not cover both acceptance and rejection')
    return results


def power(runner, trials):
    cases = fixtures()
    baseline = check(runner, cases)
    table = []
    for mutation in MUTATIONS:
        choices = []
        for c in cases:
            p, b = probabilities(c), alternatives(c)[mutation]
            tv = sum(abs(x - y) for x, y in zip(p, b)) / 2
            if tv >= .03: choices.append((tv, c))
        tv, c = min(choices, key=lambda row: row[0])
        failures = 0
        for trial in range(trials):
            mutant = dict(c, mutation=mutation, seed_start=trial * DRAWS)
            record = replay(runner, [mutant])[0]
            failures += evaluate(c, record)['state'] == 'FAIL'
        table.append(dict(mutation=mutation, tv=tv, trials=trials, rejected=failures,
                          power_lower_95=((.05 / len(MUTATIONS)) ** (1 / trials) if failures == trials else 0),
                          analytic_power_lower=1 - math.exp(-2 * DRAWS * (.03 - RADIUS)**2)))
        print(json.dumps(table[-1]), flush=True)
        if failures != trials: raise ValueError('real lane mutant survived')
    return dict(state='PASS', baseline_cells=len(baseline), draws=DRAWS, family_alpha=.01,
                radius=RADIUS, power=table, scope='predeclared mutation directions with TV >= .03; not arbitrary alternatives')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, default=Path('zig-out/bin/tf-llama-replay'))
    p.add_argument('--mutation', choices=MUTATIONS)
    p.add_argument('--cache', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--power-trials', type=int, default=0)
    a = p.parse_args()
    if a.power_trials:
        result = power(a.runner, a.power_trials)
    else:
        cases = json.loads(a.cache.read_text()) if a.cache else fixtures()
        if a.cache and len(cases) != 36: raise ValueError('expected 36 cached conditional cells')
        if a.mutation:
            cases = [dict(c, mutation=a.mutation) for c in cases]
            cells = [evaluate(c, r) for c, r in zip(cases, replay(a.runner, cases))]
            result = dict(state='FAIL' if any(c['state'] == 'FAIL' for c in cells) else 'PASS', cells=cells)
        else:
            result = dict(state='PASS', cells=check(a.runner, cases), family_alpha=.01, draws=DRAWS)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    if result['state'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__':
    main()
