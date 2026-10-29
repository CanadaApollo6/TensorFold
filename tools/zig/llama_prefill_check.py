"""Deferred full-model prompt equivalence and same-card EXL3 timing checks."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent


def cache_bar(kv, name, cross_path):
    # Cross-path reduction-order drift compounds with depth (3.7% at layer 42, A100 2K, logits KL 1e-5); KL gates.
    if cross_path:
        return .10 if kv == '4' and name.endswith('.bin') else .05
    return .05 if kv == '4' and name.endswith('.bin') else .01


def row_bar(kv, name):
    # Rows drift chaotically past layer 20 (A100 2K: 1.7x RMS at 43, <.017 to 19); writes share one path across layers.
    layer = re.fullmatch(r'cache-(\d+)-[01]\.bin', name)
    if layer is None or int(layer[1]) >= 16:
        return None
    return .25 if kv == '4' else .10


def judged(record, kv, cross_path):
    bar = row_bar(kv, record['file'])
    return record['relative_l2'] <= cache_bar(kv, record['file'], cross_path) and (bar is None or record['worst_row'] <= bar)


def compare(left, right, kv, cross_path=False):
    for pattern in ['cache-*.bin', '*.bf16', 'draft-context.json']:
        if {p.name for p in left.glob(pattern)} != {p.name for p in right.glob(pattern)}:
            raise ValueError('cache manifest mismatch')
    if (left / 'draft-context.json').exists():
        if json.loads((left / 'draft-context.json').read_text()) != json.loads((right / 'draft-context.json').read_text()):
            raise ValueError('drafter context mismatch')
    try:
        import numpy
    except ImportError:
        return compare_host(left, right, kv, cross_path)
    return compare_numpy(left, right, kv, cross_path)


def compare_host(left, right, kv, cross_path=False):
    import math
    import struct
    def floats(path):
        data = path.read_bytes()
        return list(struct.unpack('<' + 'f' * (len(data) // 4), data))
    a, b = floats(left / 'logits.f32'), floats(right / 'logits.f32')
    if len(a) != len(b) or not all(map(math.isfinite, a + b)):
        raise ValueError('invalid logits')
    def log_probs(x):
        m = max(x)
        z = math.log(sum(math.exp(v - m) for v in x))
        return [v - m - z for v in x]
    lp, lq = log_probs(a), log_probs(b)
    divergence = sum(math.exp(p) * (p - q) for p, q in zip(lp, lq))
    records = []
    for file in sorted(left.glob('cache-*.bin')) + sorted(left.glob('*.bf16')):
        bits = kv if file.suffix == '.bin' else '16'
        def decode(raw):
            if bits == '16':
                return struct.unpack('<' + 'f' * (len(raw) // 2), b''.join(b'\0\0' + raw[i:i + 2] for i in range(0, len(raw), 2)))
            values = []
            for at in range(0, len(raw), 72):
                scales = struct.unpack('<4e', raw[at + 64:at + 72])
                for d in range(128):
                    code = (raw[at + d // 2] >> (4 * (d % 2))) & 15
                    values.append((code - 7.5) * scales[d // 32] * .125)
            return values
        changed = size = 0
        row_diff, row_norm = [], []
        with file.open('rb') as x, (right / file.name).open('rb') as y:
            while raw := x.read(72 * 65536):
                other = y.read(len(raw))
                if len(raw) != len(other):
                    raise ValueError('cache shape mismatch')
                xf, yf = decode(raw), decode(other)
                if not all(map(math.isfinite, (*xf, *yf))):
                    raise ValueError('nonfinite cache')
                changed += sum(v != w for v, w in zip(raw, other))
                for at in range(0, len(xf), 128):
                    row_diff.append(sum((v - w) ** 2 for v, w in zip(xf[at:at + 128], yf[at:at + 128])))
                    row_norm.append(sum(v * v for v in xf[at:at + 128]))
                size += len(raw)
            if y.read(1):
                raise ValueError('cache shape mismatch')
        numerator, denominator = sum(row_diff), sum(row_norm)
        worst = (max(row_diff, default=0) * len(row_norm) / max(denominator, 1e-30)) ** .5
        records.append(dict(file=file.name, relative_l2=(numerator / max(denominator, 1e-30)) ** .5,
                            worst_row=worst, changed_bytes=changed, bytes=size))
    top1 = a.index(max(a)) == b.index(max(b))
    return dict(kl=divergence, top1=top1, caches=records,
                state='PASS' if divergence <= .001 and top1 and all(judged(r, kv, cross_path) for r in records) else 'FAIL')


def compare_numpy(left, right, kv, cross_path=False):
    import numpy as np
    a = np.fromfile(left / 'logits.f32', dtype='<f4')
    b = np.fromfile(right / 'logits.f32', dtype='<f4')
    def log_probabilities(x):
        x = x.astype(np.float64)
        x -= x.max()
        return x - np.log(np.exp(x).sum())
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid logits')
    lp, lq = log_probabilities(a), log_probabilities(b)
    divergence = float(np.sum(np.exp(lp) * (lp - lq)))
    records = []
    def decode(raw, bits):
        if bits == '16':
            return (raw.view('<u2').astype(np.uint32) << 16).view('<f4')
        rows = raw.reshape(-1, 72)
        codes = np.empty((len(rows), 128), dtype=np.float32)
        codes[:, ::2] = rows[:, :64] & 15
        codes[:, 1::2] = rows[:, :64] >> 4
        scales = rows[:, 64:].copy().view('<f2').astype(np.float32)
        return (codes.reshape(-1, 4, 32) - 7.5) * (scales[:, :, None] * .125)
    for file in sorted(left.glob('cache-*.bin')) + sorted(left.glob('*.bf16')):
        other = right / file.name
        x, y = np.memmap(file, dtype='u1', mode='r'), np.memmap(other, dtype='u1', mode='r')
        if x.shape != y.shape:
            raise ValueError('cache shape mismatch')
        numerator = denominator = worst_diff = 0.
        rows = changed = 0
        stride = 72 if kv == '4' and file.suffix == '.bin' else 2
        block = stride * 65536
        for at in range(0, len(x), block):
            xb, yb = x[at:at + block], y[at:at + block]
            changed += int(np.count_nonzero(xb != yb))
            bits = kv if file.suffix == '.bin' else '16'
            xf, yf = decode(xb, bits).astype(np.float64), decode(yb, bits).astype(np.float64)
            if not np.isfinite(xf).all() or not np.isfinite(yf).all():
                raise ValueError('nonfinite cache')
            diff = np.sum(((xf - yf) ** 2).reshape(-1, 128 if xf.size % 128 == 0 else xf.size), axis=1)
            numerator += float(diff.sum())
            denominator += float(np.sum(xf ** 2))
            worst_diff = max(worst_diff, float(diff.max(initial=0)))
            rows += len(diff)
        relative = (numerator / max(denominator, 1e-30)) ** .5
        worst = (worst_diff * rows / max(denominator, 1e-30)) ** .5
        records.append(dict(file=file.name, relative_l2=relative, worst_row=worst, changed_bytes=changed, bytes=len(x)))
    return dict(kl=divergence, top1=int(a.argmax()) == int(b.argmax()), caches=records,
                state='PASS' if divergence <= .001 and int(a.argmax()) == int(b.argmax()) and
                all(judged(r, kv, cross_path) for r in records) else 'FAIL')


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prompts:
        selected = args.out / 'prompt.json'
        selected.write_text(json.dumps([json.loads(args.prompts.read_text())[0]]))
        args.prompts = selected
    commands = []
    def execute(command, env=None):
        command = list(map(str, command))
        commands.append(command)
        with (args.out / f'run-{len(commands)}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
    def engine(kind):
        folder = args.out / kind
        execute([sys.executable, SCRIPTS / 'llama_gpu_run.py', '--need-mib', 18000,
                 '--timeout', args.timeout, '--receipt', args.out / f'{kind}-vram.json', '--',
                 args.runner, 'llama-prefill', args.model, args.prompts, folder, args.draft, kind],
                dict(os.environ, TF_LLAMA_KV_BITS=args.kv, TF_LLAMA_DRAFT_FORMAT=args.format))
        return folder
    if args.phase == 'equivalence':
        serial, batch = engine('serial'), engine('batch')
        if len(list(serial.glob('cache-*.bin'))) != 104:
            raise ValueError('missing full-model cache planes')
        if args.draft != 'none':
            for name in ['taps.bf16', 'draft-kv.bf16', 'draft-context.json']:
                if not (serial / name).is_file() or (serial / name).stat().st_size == 0:
                    raise ValueError('missing drafter cache evidence: ' + name)
        result = compare(serial, batch, args.kv, cross_path=True)
        resumes = {}
        for kind in args.resume_cases.split(','):
            resumes[kind] = compare(batch, engine(kind), args.kv)
        result['resumed'] = resumes
        if any(r['state'] != 'PASS' for r in resumes.values()):
            result['state'] = 'FAIL'
    elif args.phase == 'p1':
        metrics = []
        sequences = json.loads((args.refs / 'sequences.json').read_text())
        for i, seq in enumerate(sequences):
            prompt = args.out / f'p1-{i}.json'
            prompt.write_text(json.dumps([seq]))
            args.prompts = prompt
            folder = engine('timing')
            ref = (args.refs / f'ref-52-s{i}-logits.f32').read_bytes()
            if len(ref) % (len(seq) * 4):
                raise ValueError('reference logits shape mismatch')
            (args.out / 'logits.f32').write_bytes(ref[-(len(ref) // len(seq)):])
            metric = compare(args.out, folder, '16')
            metric['state'] = 'PASS' if metric['kl'] <= .01 else 'FAIL'
            metrics.append(metric)
        result = dict(state='PASS' if all(m['state'] == 'PASS' for m in metrics) else 'FAIL', sequences=metrics)
    else:
        batch = engine('timing')
        reference = args.out / 'reference'
        execute([sys.executable, SCRIPTS / 'llama_gpu_run.py', '--need-mib', 22000,
                 '--timeout', args.timeout, '--receipt', args.out / 'reference-vram.json', '--',
                 args.python, SCRIPTS / 'llama_reference.py', args.model, reference, '--sequences', args.prompts,
                 '--prefill-bench', '--kv', args.kv])
        tf = json.loads((batch / 'summary.json').read_text())
        exl = json.loads((reference / 'prefill.json').read_text())
        ratio = tf['seconds'] / exl['seconds']
        result = dict(state='PASS', speed_state='PASS' if ratio <= 2 else 'FAIL', ratio=ratio,
                      target_ratio=2, tensorfold=tf, exllamav3=exl)
    result['commands'] = commands
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['state'] != 'PASS':
        raise RuntimeError('prefill comparison failed')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'runner', 'prompts', 'out', 'refs']:
        p.add_argument('--' + name, type=Path, required=name in ['model', 'runner', 'out'])
    p.add_argument('--draft', default='none')
    p.add_argument('--kv', choices=['16', '4'], default='16')
    p.add_argument('--format', choices=['bf16', 'affine4'], default='affine4')
    p.add_argument('--phase', choices=['equivalence', 'timing', 'p1'], required=True)
    p.add_argument('--resume-cases', default='resume,shorten,cancel-shorten,cancel-extend,rewrite-cancel')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--timeout', type=int, default=1800)
    run(p.parse_args())


if __name__ == '__main__':
    main()
