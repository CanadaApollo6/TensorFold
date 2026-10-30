"""Gate P4 KV4 mean KL against exllamav3 k4-v4 over all nine length/sequence pairs."""
import argparse
import json
import math
from pathlib import Path

from llama_gate_state import atomic

CASES = tuple((length, sequence) for length in (20000, 60000, 100000) for sequence in range(3))
SEQUENCE_BAR = 1.25
POOLED_BAR = 1.05


def labels(length, sequence):
    phase = 'timing' if length == 20000 else 'kl'
    return f'p4-{length}-{phase}-{sequence}', f'p4ref-exl3-{length}-{sequence}'


def metrics(record):
    if record['rows'] != 512:
        raise ValueError('expected 512 scored rows')
    for key in ('mean_kl', 'max_kl'):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'invalid {key}: {value}')
    matching = record['greedy_matching']
    if type(matching) is not int or not 0 <= matching <= 128:
        raise ValueError('invalid greedy_matching')
    return dict(mean_kl=record['mean_kl'], max_kl=record['max_kl'],
                greedy_matching=matching, greedy_equal=matching == 128)


def ratio(value, reference):
    result = value / reference if reference else (1.0 if value == 0 else None)
    return result if result is None or math.isfinite(result) else None


def evaluate(receipts, gate_summary=None):
    states = None
    if gate_summary is not None:
        states = {job['label']: job['state'] for job in gate_summary['jobs']}
    rows = []
    for length, sequence in CASES:
        tf_label, ref_label = labels(length, sequence)
        quality = receipts / tf_label / f'quality-{length}.json'
        reference = receipts / ref_label / 'summary.json'
        row = dict(history_tokens=length, sequence=sequence, tensorfold_receipt=str(quality),
                   exllamav3_receipt=str(reference), state='FAIL')
        try:
            if states is not None:
                for label in (tf_label, ref_label):
                    if states.get(label) != 'PASS':
                        raise ValueError(f'input job {label} is {states.get(label, "missing")}; expected PASS')
            tf, exl = json.loads(quality.read_text()), json.loads(reference.read_text())
            if tf.get('state', 'PASS') != 'PASS' or tf.get('error'):
                raise ValueError('TensorFold measurement did not complete')
            if len(tf['results']) != 1:
                raise ValueError('expected one TensorFold sequence per receipt')
            tf_summary = quality.parent / 'summary.json'
            if tf_summary.exists():
                measurement = json.loads(tf_summary.read_text())
                if measurement['state'] != 'PASS':
                    raise ValueError('TensorFold measurement summary did not pass')
                if ('sequence' in measurement and measurement['sequence'] != sequence
                        or 'history_tokens' in measurement and measurement['history_tokens'] != length):
                    raise ValueError('TensorFold length/sequence mismatch')
            if exl['state'] != 'COMPLETE' or exl.get('error') or exl['configs']['k4-v4']['state'] != 'MEASURED':
                raise ValueError('exllamav3 k4-v4 counterpart did not complete')
            if (exl['history_tokens'], exl['sequence']) != (length, sequence):
                raise ValueError('exllamav3 length/sequence mismatch')
            tf_metrics, exl_metrics = metrics(tf['results'][0]), metrics(exl['configs']['k4-v4'])
            passed = tf_metrics['mean_kl'] <= SEQUENCE_BAR * exl_metrics['mean_kl']
            row.update(tensorfold=tf_metrics, exllamav3=exl_metrics,
                       ratio=ratio(tf_metrics['mean_kl'], exl_metrics['mean_kl']), state='PASS' if passed else 'FAIL')
        except (OSError, ValueError, KeyError, TypeError) as error:
            row['error'] = f'{type(error).__name__}: {error}'
        rows.append(row)
    complete = all('tensorfold' in row for row in rows)
    pooled = dict(state='FAIL', ratio=None)
    if complete:
        try:
            tf_sum = math.fsum(row['tensorfold']['mean_kl'] for row in rows)
            exl_sum = math.fsum(row['exllamav3']['mean_kl'] for row in rows)
            pooled.update(tensorfold_sum_mean_kl=tf_sum, exllamav3_sum_mean_kl=exl_sum,
                          ratio=ratio(tf_sum, exl_sum), state='PASS' if tf_sum <= POOLED_BAR * exl_sum else 'FAIL')
        except OverflowError:
            pooled['error'] = 'pooled mean KL sum is not finite'
    else:
        pooled['error'] = 'all nine finite TensorFold/exllamav3 counterparts are required'
    return dict(state='PASS' if pooled['state'] == 'PASS' and all(r['state'] == 'PASS' for r in rows) else 'FAIL',
                sequence_ratio_bar=SEQUENCE_BAR, pooled_ratio_bar=POOLED_BAR, results=rows, pooled=pooled)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipts', type=Path, required=True)
    p.add_argument('--gate-summary', type=Path)
    args = p.parse_args()
    out = args.receipts / 'p4-parity'
    out.mkdir(parents=True, exist_ok=True)
    try:
        gate_summary = json.loads(args.gate_summary.read_text()) if args.gate_summary else None
        report = evaluate(args.receipts, gate_summary)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = dict(state='FAIL', error=f'{type(error).__name__}: {error}')
    atomic(out / 'summary.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['state'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
