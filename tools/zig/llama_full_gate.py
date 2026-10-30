"""Run every pending Nanbeige full-model gate on a reserved 48 GB CUDA card."""
import argparse
import contextlib
import fnmatch
import math
import time
from llama_gate_state import atomic, digest, identity, fingerprint, resume
from llama_gate_budget import estimate, calibrate, job_class, SYNC_SECONDS
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "tools/zig"


class CleanupFailed(RuntimeError):
    """A job's processes outlived SIGKILL and may still hold the GPU, so no later job may start."""


def main():
    container_start = time.time()
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
    p.add_argument("--replay-runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-replay")
    p.add_argument("--reference-python", default=sys.executable)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--sm", choices=["80", "86", "89"], required=False)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--budget-minutes", type=float, default=210)
    p.add_argument("--deadline", type=float, default=float(os.environ.get("GATE_DEADLINE", container_start + 12600)))
    p.add_argument("--fast", action="store_true", help="P1, 2K prefill equivalence, all prefill timings, P2 and P3")
    p.add_argument("--only", help="comma-separated job labels or glob patterns")
    p.add_argument("--skip", help="comma-separated job labels or glob patterns")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="record a failed job and run the rest; p1-reference still stops")
    args = p.parse_args()
    if args.dry_run:
        from llama_gate_dry import exercise
        exercise(args.out)
        return
    for name in ["model", "draft", "histories", "pod", "smoke", "baseline_runner", "work", "out", "sm"]:
        if getattr(args, name) is None:
            p.error("missing --" + name.replace("_", "-"))
    if args.budget_minutes <= 0 or args.budget_minutes > 210 or args.timeout <= 0:
        p.error("budget must be in (0, 210] minutes and timeout must be positive")
    args.deadline = min(args.deadline, container_start + args.budget_minutes * 60)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline - time.time() - SYNC_SECONDS))
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
    for length in [2000, 60000, 100000, 20000]:
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
    add("p1-speed", [sys.executable, SCRIPTS / "llama_decode_speed.py", "--runner", args.runner,
                     "--model", args.model, "--refs", refs, "--python", args.reference_python,
                     "--out", args.out / "p1-speed"])
    add("p2", [sys.executable, SCRIPTS / "llama_draft_wire.py", "--model", args.model, "--binary", args.binary,
               "--need-mib", 18000, "--pod", args.pod, "--reference-ids", refs / "ref-52-greedy.json",
               "--smoke", args.smoke, "--out", args.out / "p2", "--timeout", args.timeout])
    add("p1-prefill", [sys.executable, SCRIPTS / "llama_prefill_check.py", "--model", args.model,
        "--runner", args.runner, "--refs", refs, "--phase", "p1", "--out", args.out / "p1-prefill",
        "--timeout", args.timeout])
    for length in [2000, 20000]:
        for fmt in ["none", "bf16", "affine4"]:
            for kv in ["16", "4"]:
                label = f"prefill-{length}-{fmt}-kv{kv}"
                add(label, [sys.executable, SCRIPTS / "llama_prefill_check.py", "--model", args.model,
                    "--runner", args.runner, "--prompts", inputs[length], "--phase", "equivalence",
                    "--draft", "none" if fmt == "none" else args.draft, "--format", "affine4" if fmt == "none" else fmt,
                    "--kv", kv, "--resume-cases", "resume" if fmt == "none" else
                    "resume,shorten,resume-shorten,cancel-shorten,cancel-extend,rewrite-cancel",
                    "--out", args.out / label, "--timeout", args.timeout])
    for length in [2000, 20000, 60000]:
        for kv in ["16", "4"]:
            label = f"prefill-speed-{length}-kv{kv}"
            add(label, [sys.executable, SCRIPTS / "llama_prefill_check.py", "--model", args.model,
                "--runner", args.runner, "--prompts", inputs[length], "--phase", "timing", "--python",
                args.reference_python, "--kv", kv, "--out", args.out / label, "--timeout", args.timeout])
    for length in [60000, 100000, 20000]:
        for i in range(3):
            phase = 'kl' if length != 20000 else 'timing'
            label = f"p4-{length}-{phase}-{i}"
            add(label, [sys.executable, SCRIPTS / "llama_kv_full.py", "--model", args.model,
                        "--prompts", inputs[length], "--sequence", i, "--reps", 5, "--runner", args.runner,
                        "--work", args.work / label, "--receipts", args.out / label, "--timeout", args.timeout])
    for fmt in ["bf16", "affine4"]:
        for kv in ["16", "4"]:
            label = f"p3-{fmt}-kv{kv}"
            for phase in ['basic', 'speed']:
                name = label if phase == 'basic' else label + '-speed'
                add(name, [sys.executable, SCRIPTS / "llama_draft_full.py", "--model", args.model,
                           "--draft", args.draft, "--prompts", short_file, "--format", fmt, "--kv", kv,
                           "--replay-runner", args.replay_runner, "--phase", phase, "--reps", 1 if phase == 'basic' else 5, "--samples", 16,
                           "--runner", args.runner, "--out", args.out / name, "--timeout", args.timeout])
            add(label + "-wire", [sys.executable, SCRIPTS / "llama_draft_wire.py", "--model", args.model,
                                   "--draft", args.draft, "--binary", args.binary, "--runner", args.runner,
                                   "--format", fmt, "--kv", kv, "--out", args.out / (label + "-wire"),
                                   "--timeout", args.timeout, "--wire-runner", args.wire_runner])
    if args.fast:
        if args.only:
            p.error("--fast and --only are mutually exclusive")
        args.only = "p1-*,prefill-2000-*,prefill-speed-*,p2,p3-*"
    def priority(job):
        label = job['label']
        if label in ['p1-reference', 'p1-logits', 'p1-greedy', 'p1-prefill']: return 0
        if label.startswith('prefill-2000-'): return 1
        if label.startswith('prefill-speed-'): return 2
        if label.startswith('p4-20000-'): return 3
        if label.startswith(('p4-60000-', 'p4-100000-')): return 4
        if label.startswith('prefill-'): return 5
        if label.startswith('p3-') and not label.endswith(('wire', 'speed')): return 6
        if label.endswith('wire') or label == 'p2': return 7
        return 8
    jobs.sort(key=priority)
    required = [args.model / "config.json", args.draft / "config.json", args.pod, args.smoke,
                args.baseline_runner, args.runner, args.binary, args.wire_runner, args.replay_runner]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        p.error("missing inputs: " + ", ".join(missing))
    preflight = args.out / 'preflight.json'
    atomic(preflight, dict(state='RUNNING', jobs=jobs, deadline=args.deadline))
    try:
        inputs = {name: identity(getattr(args, name)) for name in
                  ['model', 'draft', 'histories', 'pod', 'smoke', 'baseline_runner', 'runner', 'binary', 'wire_runner', 'replay_runner']}
        inputs['fatbins'] = identity(ROOT / 'zig-out/fatbin') if (ROOT / 'zig-out/fatbin').exists() else None
        inputs['source_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        inputs['scripts'] = {str(f.relative_to(ROOT)): digest(f) for f in sorted(SCRIPTS.iterdir())
                             if f.suffix in ['.py', '.sh', '.lock', '.in']}
        inputs['reference_environment'] = subprocess.check_output([args.reference_python, '-c',
            'import importlib.metadata as m,sys,json; print(json.dumps([sys.version, sorted((d.metadata["Name"],d.version) for d in m.distributions())]))'], text=True).strip()
        binding = fingerprint(dict(inputs=inputs, jobs=jobs, sm=args.sm))
    except BaseException as error:
        atomic(preflight, dict(state='FAIL', error=str(error), jobs=jobs, deadline=args.deadline))
        raise
    atomic(preflight, dict(state='PASS', jobs=jobs, deadline=args.deadline))
    if execute(args, jobs, binding, inputs)['state'] == 'FAIL':
        raise SystemExit('gate FAIL; see summary.json')


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
    if chosen & {'p1-logits', 'p1-greedy', 'p1-speed', 'p1-prefill', 'p2'}:
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
    deadline = min(getattr(args, 'deadline', time.time() + args.budget_minutes * 60),
                   time.time() + args.budget_minutes * 60)
    summary['deadline_epoch'] = deadline
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
                                     basis='per-class wall seconds with 25% reserve; uncalibrated classes use planning assumptions',
                                     elapsed_seconds=elapsed, remaining_wall_seconds=deadline - time.time(),
                                     final_sync_seconds=SYNC_SECONDS, container_elapsed_seconds=max(0, time.time() - float(os.environ.get('GATE_STARTED', time.time()))), projected_session_seconds=elapsed + total + SYNC_SECONDS)
        summary.setdefault('projection_history', []).append(summary['projection'])
        print('PROJECTED ' + json.dumps(summary['projection']), flush=True)
        if projections[job['label']] + SYNC_SECONDS > deadline - time.time():
            summary['budget_exhausted_before'] = job['label']
            failed = any(j['label'] in chosen and j['state'] == 'FAIL' for j in jobs)
            summary['state'] = 'FAIL' if failed else 'BUDGET'
            atomic(path, summary)
            raise RuntimeError(f"budget would be exceeded before {job['label']}; increase rates or select fewer jobs; see summary.json")
        job.update(state='RUNNING', attempts=job.get('attempts', 0) + 1)
        atomic(path, summary)
        limit = max(1, int(min(args.timeout, deadline - time.time() - SYNC_SECONDS)))
        if job['label'].startswith('p4-'):
            limit = min(limit, max(600, math.ceil(estimate(job, rates) * 3)))
        job['timeout_seconds'] = limit
        before = time.monotonic()
        log_path = args.out / (job['label'] + '.log')
        try:
            with log_path.open('w') as log:
                code = worker(job, log, limit) if worker else bounded(job['command'], env, log, limit)
            job.update(state='PASS' if code == 0 else 'FAIL', exit_code=code, seconds=time.monotonic() - before)
            nested = args.out / job['label'] / ('comparison.json' if job['label'] == 'p1-speed' else 'summary.json')
            if nested.exists(): job['result'] = json.loads(nested.read_text())
            if job['label'] == 'p1-speed': summary['p1_speed'] = job.get('result', {}).get('speed_state', 'NOT RUN')
            if job.get('result', {}).get('state') in ['FAIL', 'PARTIAL', 'NOT RUN']: code = 1
            if code: raise RuntimeError(f"{job['label']} failed; see {log_path}")
            rates = dict(rates or {})
            kind = job_class(job)
            rates[kind] = max(rates.get(kind, 0), calibrate(job, job['seconds']))
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
            if not (getattr(args, 'keep_going', False) and isinstance(error, Exception)
                    and not isinstance(error, CleanupFailed) and job['label'] != 'p1-reference'):
                raise
            print(f"KEEP GOING after {job['label']}: {error}", flush=True)
            continue
        atomic(path, summary)
    if any(j['label'] in chosen and j['state'] == 'FAIL' for j in jobs): summary['state'] = 'FAIL'
    else: summary['state'] = 'PASS' if len(chosen) == len(jobs) else 'PARTIAL'
    atomic(path, summary)
    return summary


def bounded(command, env, log, limit, grace=25):
    with deferred_interrupts() as pending:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        end = time.monotonic() + limit
        try:
            while (code := process.poll()) is None and not pending:
                if time.monotonic() > end: raise subprocess.TimeoutExpired(command, limit)
                time.sleep(.05)
        except BaseException:
            if not stop_group(process, grace): raise CleanupFailed(f'job processes survived SIGKILL: {command[:2]}')
            if not pending: raise
        if not stop_group(process, grace): raise CleanupFailed(f'job processes survived SIGKILL: {command[:2]}')
    return code


@contextlib.contextmanager
def deferred_interrupts():
    """Queues SIGINT, SIGTERM and deadline SIGALRM until job cleanup ends, then runs the first one's handler."""
    import signal
    pending = []
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    saved = {s: signal.signal(s, lambda signum, frame: pending.append(signum)) for s in signals}
    try:
        yield pending
    finally:
        for s, handler in saved.items(): signal.signal(s, handler)
    if pending:
        handler = saved[pending[0]]
        if callable(handler): handler(pending[0], None)
        raise KeyboardInterrupt(f'signal {pending[0]}')


def group_alive(group):
    for stat in Path('/proc').glob('[0-9]*/stat'):
        try: fields = stat.read_text().rsplit(')', 1)[1].split()
        except OSError: continue
        if fields[0] != 'Z' and int(fields[2]) == group: return True
    return False


def stop_group(process, grace):
    """Stops every process in the job's session, including ones that outlive the wrapper; False if any survive."""
    import signal
    # The native server drains for up to 20 s on SIGTERM; later jobs need its VRAM back.
    for sig, wait in [(signal.SIGTERM, grace), (signal.SIGKILL, 10)]:
        if process.poll() is None or group_alive(process.pid):
            try: os.killpg(process.pid, sig)
            except ProcessLookupError: pass
        end = time.monotonic() + wait
        while (process.poll() is None or group_alive(process.pid)) and time.monotonic() < end: time.sleep(.1)
        if process.poll() is not None and not group_alive(process.pid): return True
    return False


if __name__ == '__main__':
    main()
