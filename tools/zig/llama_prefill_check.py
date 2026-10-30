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


def row_rule(kv, name, cross_path, rows, positions):
    """Per-row bar and checked rows `row % period < count`, or None where cross-path drift outgrows any row bound."""
    # Rows drift chaotically past layer 20 (A100 2K: 1.7x RMS at 43, <.017 to 19); writes share one path across layers.
    layer = re.fullmatch(r'cache-(\d+)-[01]\.bin', name)
    if layer:
        # Resume shares the batched kernels with cold prefill (bit-identical on the A100): same-path checks all layers.
        return (.25 if kv == '4' else .10, rows, rows) if int(layer[1]) < 16 or not cross_path else None
    if name == 'bos.bf16':
        # KV4 attention reads BOS from this bf16 copy; position 0 sees only itself and matched exactly on the A100.
        return .10, rows, rows
    if name not in ('taps.bf16', 'draft-kv.bf16'):
        return None
    if not cross_path:
        return .10, rows, rows
    if name == 'draft-kv.bf16':
        return None
    # Taps are [position][5 planes: layers 4, 15, 26, 38, 49][hidden]; only the first two stay below layer 20.
    if not positions or rows % (5 * positions):
        raise ValueError('tap layout mismatch')
    return .10, 2 * rows // (5 * positions), rows // positions


def row_score(diff, norm, rms):
    # A row's error against its own norm, floored at 5% of the plane's RMS row so near-zero rows cannot dominate.
    return diff ** .5 / max(norm ** .5, .05 * rms, 1e-30)


def judged(record, kv, cross_path):
    return record['relative_l2'] <= cache_bar(kv, record['file'], cross_path) and (
        record['row_bar'] is None or record['worst_row'] <= record['row_bar'])


def drafter_positions(folder):
    context = folder / 'draft-context.json'
    if not context.exists():
        return None
    window = json.loads(context.read_text())
    return window['end'] - window['begin']


def compare(left, right, kv, cross_path=False):
    for pattern in ['cache-*.bin', '*.bf16', 'draft-context.json']:
        if {p.name for p in left.glob(pattern)} != {p.name for p in right.glob(pattern)}:
            raise ValueError('cache manifest mismatch')
        if any(p.stat().st_size == 0 for d in [left, right] for p in d.glob(pattern)):
            raise ValueError('empty cache evidence')
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
    positions = drafter_positions(left)
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
        rule = row_rule(kv, file.name, cross_path, len(row_norm), positions)
        worst = None
        if rule:
            rms = (denominator / max(len(row_norm), 1)) ** .5
            worst = max((row_score(d, n, rms) for i, (d, n) in enumerate(zip(row_diff, row_norm))
                         if i % rule[2] < rule[1]), default=0)
        records.append(dict(file=file.name, relative_l2=(numerator / max(denominator, 1e-30)) ** .5,
                            worst_row=worst, row_bar=rule and rule[0], changed_bytes=changed, bytes=size))
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
    positions = drafter_positions(left)
    for file in sorted(left.glob('cache-*.bin')) + sorted(left.glob('*.bf16')):
        other = right / file.name
        x, y = np.memmap(file, dtype='u1', mode='r'), np.memmap(other, dtype='u1', mode='r')
        if x.shape != y.shape:
            raise ValueError('cache shape mismatch')
        changed = 0
        diffs, norms = [], []
        stride = 72 if kv == '4' and file.suffix == '.bin' else 2
        block = stride * 65536
        for at in range(0, len(x), block):
            xb, yb = x[at:at + block], y[at:at + block]
            changed += int(np.count_nonzero(xb != yb))
            bits = kv if file.suffix == '.bin' else '16'
            xf, yf = decode(xb, bits).astype(np.float64), decode(yb, bits).astype(np.float64)
            if not np.isfinite(xf).all() or not np.isfinite(yf).all():
                raise ValueError('nonfinite cache')
            starts = np.arange(0, xf.size, 128)
            diffs.append(np.add.reduceat(((xf - yf) ** 2).ravel(), starts))
            norms.append(np.add.reduceat((xf ** 2).ravel(), starts))
        diff, norm = np.concatenate(diffs), np.concatenate(norms)
        relative = (float(diff.sum()) / max(float(norm.sum()), 1e-30)) ** .5
        rule = row_rule(kv, file.name, cross_path, len(norm), positions)
        worst = None
        if rule:
            rms = float(np.sqrt(norm.mean()))
            checked = np.arange(len(norm)) % rule[2] < rule[1]
            score = np.sqrt(diff[checked]) / np.maximum(np.sqrt(norm[checked]), max(.05 * rms, 1e-30))
            worst = float(score.max(initial=0))
        records.append(dict(file=file.name, relative_l2=relative, worst_row=worst, row_bar=rule and rule[0],
                            changed_bytes=changed, bytes=len(x)))
    return dict(kl=divergence, top1=int(a.argmax()) == int(b.argmax()), caches=records,
                state='PASS' if divergence <= .001 and int(a.argmax()) == int(b.argmax()) and
                all(judged(r, kv, cross_path) for r in records) else 'FAIL')


def engine_need(args, tokens):
    # The gate sizes context as in llama_prompt_gate.zig; A100 peaks ran 600-640 MiB over the allocator's NEED at 2K.
    context = tokens + min(1000, tokens // 4) + 1
    mode = 'q4' if args.kv == '4' else 'bf16'
    if args.draft == 'none':
        command, pattern = [args.runner, 'llama-capacity', args.model, '49152', context], rf'NEED mode={mode} context={context} '
    else:
        command = [args.runner, 'llama-draft-capacity', args.model, args.draft, '49152', '512', context]
        pattern = rf'NEED draft={args.format} kv={mode} context={context} '
    text = subprocess.run(list(map(str, command)), capture_output=True, text=True, check=True)
    need = re.search(pattern + r'bytes=\d+ need_mib=(\d+)', text.stdout + text.stderr)
    if not need:
        raise RuntimeError('no matching allocator requirement')
    return int(need[1]) + 1024


def reference_need(args, tokens):
    # exllamav3 peaked at 14.4 GB without a cache; its cache holds 52 layers of K and V for the rounded-up prompt.
    capacity = (tokens + 255) // 256 * 256
    return 16384 + 52 * 2 * capacity * 1024 * (2 if args.kv == '16' else 1) // (1 << 20)


def prefill_profile(text, seconds):
    """Mean warm-run (2-4) phase table from TF_LLAMA_PREFILL_PROFILE log lines; idle is host gaps inside chunk spans."""
    fields = lambda line: dict(item.split('=', 1) for item in line.split()[1:])
    runs, phases = {}, {}
    for line in text.splitlines():
        at = line.find('LLAMA_PREFILL_')
        if at < 0:
            continue
        kind, values = line[at:].split()[0], fields(line[at:])
        if kind == 'LLAMA_PREFILL_RUN':
            runs[int(values['run'])] = {k: float(values[k]) for k in ['tokens', 'chunks', 'span_ms', 'busy_ms', 'idle_ms']}
        elif kind == 'LLAMA_PREFILL_PHASE':
            phases.setdefault(int(values['run']), {})[values['phase']] = float(values['ms'])
    warm = [r for r in sorted(runs) if r > 1 and r in phases]
    if not warm:
        raise ValueError('no warm LLAMA_PREFILL_RUN lines')
    mean = lambda xs: sum(xs) / len(xs)
    span = mean([runs[r]['span_ms'] for r in warm])
    ms = {name: mean([phases[r][name] for r in warm]) for name in phases[warm[0]]}
    gemm = sum(ms[name] for name in ['q', 'k', 'v', 'o', 'gate', 'up', 'down', 'head'])
    return dict(runs=warm, tokens=runs[warm[0]]['tokens'], chunks=runs[warm[0]]['chunks'], span_ms=span,
                idle_ms=mean([runs[r]['idle_ms'] for r in warm]), host_ms=seconds * 1000 - span,
                gemm_ms=gemm, attention_ms=ms['attention'], other_ms=sum(ms.values()) - gemm - ms['attention'], phases_ms=ms)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prompts:
        selected = args.out / 'prompt.json'
        selected.write_text(json.dumps([json.loads(args.prompts.read_text())[0]]))
        args.prompts = selected
    commands = []
    profile = getattr(args, 'profile', False)
    def execute(command, env=None):
        command = list(map(str, command))
        commands.append(command)
        with (args.out / f'run-{len(commands)}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
    def tokens():
        return len(json.loads(args.prompts.read_text())[0])
    def engine(kind):
        folder = args.out / kind
        execute([sys.executable, SCRIPTS / 'llama_gpu_run.py', '--need-mib', engine_need(args, tokens()),
                 '--timeout', args.timeout, '--receipt', args.out / f'{kind}-vram.json', '--',
                 args.runner, 'llama-prefill', args.model, args.prompts, folder, args.draft, kind],
                dict(os.environ, TF_LLAMA_KV_BITS=args.kv, TF_LLAMA_DRAFT_FORMAT=args.format,
                     **({'TF_LLAMA_PREFILL_PROFILE': '1'} if profile and kind == 'timing' else {})))
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
        log = args.out / f'run-{len(commands)}.log'
        reference = args.out / 'reference'
        execute([sys.executable, SCRIPTS / 'llama_gpu_run.py', '--need-mib', reference_need(args, tokens()),
                 '--timeout', args.timeout, '--receipt', args.out / 'reference-vram.json', '--',
                 args.python, SCRIPTS / 'llama_reference.py', args.model, reference, '--sequences', args.prompts,
                 '--prefill-bench', '--kv', args.kv] + (['--prefill-profile'] if profile else []))
        tf = json.loads((batch / 'summary.json').read_text())
        exl = json.loads((reference / 'prefill.json').read_text())
        ratio = tf['seconds'] / exl['seconds']
        result = dict(state='PASS', speed_state='PASS' if ratio <= 2 else 'FAIL', ratio=ratio,
                      target_ratio=2, tensorfold=tf, exllamav3=exl)
        if profile:
            # Event timing perturbs the clock: the ratio above is diagnostic, the plain prefill-speed jobs are the gate.
            result['tensorfold_profile'] = prefill_profile(log.read_text(), tf['seconds'])
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
    p.add_argument('--resume-cases', default='resume,shorten,resume-shorten,cancel-shorten,cancel-extend,rewrite-cancel')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--profile', action='store_true', help='timing only: TF per-phase CUDA events and exllamav3 kernel table')
    run(p.parse_args())


if __name__ == '__main__':
    main()
