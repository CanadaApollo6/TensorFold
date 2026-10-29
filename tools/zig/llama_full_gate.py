"""Run every pending Nanbeige full-model gate on a reserved 48 GB CUDA card."""
import argparse
import fnmatch
import time
from llama_gate_state import atomic, digest, identity, fingerprint, resume
from llama_gate_budget import estimate, calibrate
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "tools/zig"


def main():
    import signal
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "draft", "histories", "pod", "smoke", "baseline-runner", "work", "out"]:
        p.add_argument("--" + name, type=Path, required=False)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    p.add_argument("--binary", type=Path, default=ROOT / "zig-out/native/bin/tensorfold-native")
    p.add_argument("--wire-runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-wire-test")
    p.add_argument("--reference-python", default=sys.executable)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--sm", choices=["86", "89"], required=False)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--budget-minutes", type=float, default=150)
    p.add_argument("--only", help="comma-separated job labels or glob patterns")
    p.add_argument("--skip", help="comma-separated job labels or glob patterns")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.dry_run:
        from llama_gate_dry import exercise
        exercise(args.out)
        return
    for name in ["model", "draft", "histories", "pod", "smoke", "baseline_runner", "work", "out", "sm"]:
        if getattr(args, name) is None:
            p.error("missing --" + name.replace("_", "-"))
    if args.budget_minutes <= 0 or args.budget_minutes > 150 or args.timeout <= 0:
        p.error("budget must be in (0, 150] minutes and timeout must be positive")
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    histories = json.loads(args.histories.read_text())
    if not isinstance(histories, list) or len(histories) < 3 or any(
            not isinstance(h, list) or len(h) < 100000 or h[0] != 166100 for h in histories):
        p.error("--histories must contain at least three BOS-prefixed histories of at least 100000 tokens")
    if any(type(token) is not int or token < 0 or token >= 166912 for h in histories for token in h):
        p.error("histories contain an invalid Nanbeige token ID")
    inputs = {}
    for length in [60000, 100000, 20000]:
        distinct = list(dict.fromkeys(tuple(h[:length]) for h in histories))
        if len(distinct) < 3:
            p.error(f"histories are not distinct at {length}")
        file = args.work / f"histories-{length}.json"
        file.write_text(json.dumps(distinct[:3]) + "\n")
        inputs[length] = file
    short = list(dict.fromkeys(tuple([h[0]] + h[-127:]) for h in histories))
    if len(short) < 3:
        p.error("short BOS-prefixed suffixes must be distinct")
    short_file = args.work / "short.json"
    short_file.write_text(json.dumps(short[:3]) + "\n")
    refs = args.work / "p1-reference"
    refs.mkdir(exist_ok=True)
    jobs = []
    def add(label, cmd):
        jobs.append(dict(label=label, command=list(map(str, cmd)), state="NOT RUN"))
    def guarded(label, need, cmd):
        add(label, [sys.executable, SCRIPTS / "llama_gpu_run.py", "--need-mib", need,
                    "--timeout", args.timeout, "--receipt", args.out / f"{label}-vram.json", "--", *cmd])
    guarded("p1-reference", 20000, [args.reference_python, SCRIPTS / "llama_reference.py", args.model, refs,
                                   "--layers", 52, "--greedy", 128])
    guarded("p1-logits", 17000, [args.runner, "llama-logits", args.model, refs, args.work / "p1-logits"])
    guarded("p1-greedy", 17000, [args.runner, "llama-greedy", args.model, refs, 128, args.work / "p1-greedy.json"])
    add("p1-speed", [sys.executable, SCRIPTS / "llama_fix_bench.py", args.baseline_runner, args.runner,
                     args.model, refs, args.out / "p1-speed"])
    add("p2", [sys.executable, SCRIPTS / "llama_draft_wire.py", "--model", args.model, "--binary", args.binary,
               "--need-mib", 18000, "--pod", args.pod, "--reference-ids", refs / "ref-52-greedy.json",
               "--smoke", args.smoke, "--out", args.out / "p2", "--timeout", args.timeout])
    for i in range(3):
        label = f"p4-100000-{i}"
        add(label, [sys.executable, SCRIPTS / "llama_kv_full.py", "--model", args.model, "--prompts", inputs[100000],
                    "--sequence", i, "--sweep", "--reps", 5, "--runner", args.runner, "--work", args.work / label,
                    "--receipts", args.out / label, "--timeout", args.timeout])
    for fmt in ["bf16", "affine4"]:
        for kv in ["16", "4"]:
            label = f"p3-{fmt}-kv{kv}"
            add(label, [sys.executable, SCRIPTS / "llama_draft_full.py", "--model", args.model, "--draft", args.draft,
                        "--prompts", short_file, "--format", fmt, "--kv", kv, "--reps", 5, "--samples", 8192,
                        "--runner", args.runner, "--out", args.out / label, "--timeout", args.timeout])
            add(label + "-wire", [sys.executable, SCRIPTS / "llama_draft_wire.py", "--model", args.model,
                                   "--draft", args.draft, "--binary", args.binary, "--runner", args.runner,
                                   "--format", fmt, "--kv", kv, "--out", args.out / (label + "-wire"),
                                   "--timeout", args.timeout, "--wire-runner", args.wire_runner])
    def priority(job):
        label = job['label']
        if label == 'p4-100000-0': return 0
        if label.startswith('p3-') and not label.endswith('wire'): return 2
        if label in ['p1-reference', 'p1-logits']: return 3
        return 4
    jobs.sort(key=priority)
    required = [args.model / "config.json", args.draft / "config.json", args.pod, args.smoke,
                args.baseline_runner, args.runner, args.binary, args.wire_runner]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        p.error("missing inputs: " + ", ".join(missing))
    inputs = {name: identity(getattr(args, name)) for name in
              ['model', 'draft', 'histories', 'pod', 'smoke', 'baseline_runner', 'runner', 'binary', 'wire_runner']}
    inputs['fatbins'] = identity(ROOT / 'zig-out/fatbin') if (ROOT / 'zig-out/fatbin').exists() else None
    inputs['source_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    inputs['scripts'] = {str(f.relative_to(ROOT)): digest(f) for f in sorted(SCRIPTS.glob('*.py'))}
    binding = fingerprint(dict(inputs=inputs, jobs=jobs, sm=args.sm))
    execute(args, jobs, binding, inputs)


def select(jobs, only, skip):
    labels = {j['label'] for j in jobs}
    def matches(patterns):
        found = set()
        for pattern in (patterns or '').split(','):
            if not pattern: continue
            group = {label for label in labels if fnmatch.fnmatchcase(label, pattern)}
            if not group: raise ValueError(f'job selector matched nothing: {pattern}')
            found.update(group)
        return found
    excluded = matches(skip)
    chosen = (matches(only) if only else labels) - excluded
    if not chosen: raise ValueError('no selected jobs')
    # Downstream P1/P2 jobs consume the exl3 output and cannot quietly use stale files.
    if chosen & {'p1-logits', 'p1-greedy', 'p1-speed', 'p2'}:
        if 'p1-reference' in excluded:
            raise ValueError('cannot skip p1-reference while selecting its consumers')
        chosen.add('p1-reference')
    return chosen


def execute(args, jobs, binding, inputs, worker=None):
    import fcntl
    with (args.out / '.gate.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another gate owns this --out directory') from error
        return execute_locked(args, jobs, binding, inputs, worker)


def execute_locked(args, jobs, binding, inputs, worker=None):
    path = args.out / 'summary.json'
    summary = resume(path, binding, jobs)
    summary.update(inputs=inputs, budget_minutes=args.budget_minutes)
    chosen = select(jobs, args.only, args.skip)
    summary['selected'] = sorted(chosen)
    if args.plan_only:
        atomic(args.out / 'plan.json', summary)
        print(json.dumps(summary, indent=2))
        return summary
    if worker is None:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap,memory.total',
                                       '--format=csv,noheader,nounits', '--id=0'], text=True).strip()
        name, capability, total = [part.strip() for part in gpu.split(',')]
        if capability.replace('.', '') != args.sm or int(total) < 45000:
            raise ValueError('expected a 48 GB card matching --sm')
        summary['gpu'] = dict(name=name, sm=capability, total_mib=int(total))
    env = dict(os.environ, TF_LLAMA_KV_BITS='16', TF_LLAMA_DRAFT_FORMAT='affine4',
               TF_LLAMA_ALLOW_UNQUALIFIED_SM=args.sm)
    start = time.monotonic()
    rates = summary.get('rates')
    summary['state'] = 'RUNNING'
    atomic(path, summary)
    for job in jobs:
        if job['label'] not in chosen or job['state'] == 'PASS': continue
        remaining = [j for j in jobs if j['label'] in chosen and j['state'] != 'PASS']
        projections = {j['label']: round(estimate(j, rates), 2) for j in remaining}
        total = sum(projections.values())
        elapsed = time.monotonic() - start
        summary['projection'] = dict(seconds_by_job=projections, total_seconds=total, rates=rates,
                                     basis='measured with 25% reserve' if rates else '860 GB/s planning assumption',
                                     elapsed_seconds=elapsed, projected_session_seconds=elapsed + total)
        summary.setdefault('projection_history', []).append(summary['projection'])
        print('PROJECTED ' + json.dumps(summary['projection']), flush=True)
        if elapsed + projections[job['label']] > args.budget_minutes * 60 or (rates and elapsed + total > args.budget_minutes * 60):
            summary['state'] = 'BUDGET'
            atomic(path, summary)
            raise RuntimeError(f"budget would be exceeded before {job['label']}; increase rates or select fewer jobs; see summary.json")
        job.update(state='RUNNING', attempts=job.get('attempts', 0) + 1)
        atomic(path, summary)
        limit = max(1, int(min(args.timeout, args.budget_minutes * 60 - elapsed)))
        before = time.monotonic()
        log_path = args.out / (job['label'] + '.log')
        try:
            with log_path.open('w') as log:
                code = worker(job, log, limit) if worker else bounded(job['command'], env, log, limit)
            job.update(state='PASS' if code == 0 else 'FAIL', exit_code=code, seconds=time.monotonic() - before)
            nested = args.out / job['label'] / ('comparison.json' if job['label'] == 'p1-speed' else 'summary.json')
            if nested.exists(): job['result'] = json.loads(nested.read_text())
            if code: raise RuntimeError(f"{job['label']} failed; see {log_path}")
            if job['label'].startswith('p4-'):
                rates = calibrate(job['result'])
                # Convert measurements at other lengths back to a conservative 60K baseline.
                scale = max(1, int(job['label'].split('-')[1]) / 60000)
                rates = {key: value * scale for key, value in rates.items()}
                summary['rates'] = rates
            artifacts = [log_path]
            for folder in [args.out / job['label'], args.work / job['label']]:
                if folder.exists(): artifacts += [f for f in folder.rglob('*') if f.is_file()]
            if job['label'] == 'p1-reference':
                artifacts += [f for f in (args.work / 'p1-reference').rglob('*') if f.is_file()]
            job['artifacts'] = {str(f): digest(f) for f in artifacts}
        except (Exception, KeyboardInterrupt) as error:
            job.update(state='FAIL', error=str(error))
            summary['state'] = 'FAIL'
            atomic(path, summary)
            raise
        atomic(path, summary)
    summary['state'] = 'PASS' if len(chosen) == len(jobs) else 'PARTIAL'
    atomic(path, summary)
    return summary


def bounded(command, env, log, limit):
    import signal
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=limit)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise


if __name__ == '__main__':
    main()
