"""Run every pending Nanbeige full-model gate on a reserved 48 GB CUDA card."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "tools/zig"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["model", "draft", "histories", "pod", "smoke", "baseline-runner", "work", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    p.add_argument("--binary", type=Path, default=ROOT / "zig-out/native/bin/tensorfold-native")
    p.add_argument("--reference-python", default=sys.executable)
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--sm", choices=["86", "89"], required=True)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    histories = json.loads(args.histories.read_text())
    if len(histories) < 3 or any(len(h) < 100000 or h[0] != 166100 for h in histories):
        p.error("--histories must contain at least three BOS-prefixed histories of at least 100000 tokens")
    inputs = {}
    for length in [20000, 60000, 100000]:
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
    for length, prompts in inputs.items():
        for i in range(3):
            label = f"p4-{length}-{i}"
            add(label, [sys.executable, SCRIPTS / "llama_kv_full.py", "--model", args.model, "--prompts", prompts,
                        "--sequence", i, "--reps", 5, "--runner", args.runner, "--work", args.work / label,
                        "--receipts", args.out / label, "--timeout", args.timeout])
    for fmt in ["bf16", "affine4"]:
        for kv in ["16", "4"]:
            label = f"p3-{fmt}-kv{kv}"
            add(label, [sys.executable, SCRIPTS / "llama_draft_full.py", "--model", args.model, "--draft", args.draft,
                        "--prompts", short_file, "--format", fmt, "--kv", kv, "--reps", 5, "--seeds", 32,
                        "--runner", args.runner, "--out", args.out / label, "--timeout", args.timeout])
            add(label + "-wire", [sys.executable, SCRIPTS / "llama_draft_wire.py", "--model", args.model,
                                   "--draft", args.draft, "--binary", args.binary, "--runner", args.runner,
                                   "--format", fmt, "--kv", kv, "--out", args.out / (label + "-wire"),
                                   "--timeout", args.timeout])
    summary = dict(state="NOT RUN", jobs=jobs, source_head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_sha256=hashlib.sha256(args.histories.read_bytes()).hexdigest(),
        hardware="48 GB sm_86 or sm_89; build for the actual SM before running",
        artifacts="large logits stay in --work; logs and summary stay in --out")
    def save():
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    save()
    if args.plan_only:
        return
    required = [args.model / "config.json", args.draft / "config.json", args.pod, args.smoke,
                args.baseline_runner, args.runner, args.binary]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        summary.update(state="FAIL", preflight_error=dict(missing=missing))
        save()
        raise SystemExit("missing inputs; see summary.json")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                                   "--format=csv,noheader,nounits", "--id=0"], text=True).strip()
    name, capability, total = [part.strip() for part in gpu.split(",")]
    summary["gpu"] = dict(name=name, sm=capability, total_mib=int(total))
    if capability.replace(".", "") != args.sm or int(total) < 45000:
        summary.update(state="FAIL", preflight_error="expected a 48 GB card matching --sm")
        save()
        raise SystemExit(summary["preflight_error"])
    env = dict(os.environ, TF_LLAMA_KV_BITS="16", TF_LLAMA_DRAFT_FORMAT="affine4",
               TF_LLAMA_ALLOW_UNQUALIFIED_SM=args.sm)
    for job in jobs:
        try:
            result = subprocess.run(job["command"], cwd=ROOT, env=env, capture_output=True, text=True)
        except OSError as error:
            job.update(state="FAIL", error=str(error))
            summary["state"] = "FAIL"
            save()
            raise SystemExit(f"{job['label']} could not start; see summary.json") from error
        (args.out / (job["label"] + ".log")).write_text(result.stdout + result.stderr)
        job["state"] = "PASS" if result.returncode == 0 else "FAIL"
        job["exit_code"] = result.returncode
        nested = args.out / job["label"] / ("comparison.json" if job["label"] == "p1-speed" else "summary.json")
        if nested.exists():
            job["result"] = json.loads(nested.read_text())
        save()
        if result.returncode:
            summary["state"] = "FAIL"
            save()
            raise SystemExit(f"{job['label']} failed; remaining checks are NOT RUN, see summary.json")
    summary["state"] = "PASS"
    save()


if __name__ == "__main__":
    main()
