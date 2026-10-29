"""Run the full production lane check under the GPU admission guard and summarize paired measurements."""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=["bf16", "affine4"], required=True)
    p.add_argument("--kv", choices=["16", "4"], required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--driver-mib", type=int, default=512)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    args = p.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if len(prompts) != 3 or args.tokens < 256 or args.reps < 5:
        p.error("the gate needs three prompts, at least 256 tokens and five benchmark pairs")
    args.out.mkdir(parents=True, exist_ok=True)
    context = max(map(len, prompts)) + args.tokens + 8
    total = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"], text=True).strip()
    capacity = subprocess.run([args.runner, "llama-draft-capacity", args.model, args.draft, total,
                               str(args.driver_mib), str(context)], capture_output=True, text=True, check=True)
    cap_text = capacity.stdout + capacity.stderr
    (args.out / "capacity.log").write_text(cap_text)
    kv = "q4" if args.kv == "4" else "bf16"
    match = re.search(rf"NEED draft={args.format} kv={kv} .* need_mib=(\d+)", cap_text)
    if not match:
        raise RuntimeError("capacity command supplied no matching allocator requirement")
    need = int(match.group(1)) + args.driver_mib
    output = args.out / "pairs.json"
    command = [sys.executable, ROOT / "tools/zig/llama_gpu_run.py", "--need-mib", str(need),
               "--timeout", str(args.timeout), "--receipt", args.out / "vram.json", "--",
               args.runner, "llama-draft-full", args.model, args.draft, args.prompts, output,
               str(args.tokens), str(args.seeds), str(args.reps)]
    command = list(map(str, command))
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = dict(os.environ, TF_LLAMA_DRAFT_FORMAT=args.format, TF_LLAMA_KV_BITS=args.kv)
    with (args.out / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((args.out / "run.log").read_text())
        raise SystemExit(result.returncode)
    pairs = json.loads(output.read_text())
    greedy = [r for r in pairs if r["seed"] is None]
    ratios = [r["plain"]["seconds"] / r["drafted"]["seconds"] for r in greedy]
    for pair in pairs:
        if pair["plain"]["tokens"] != pair["drafted"]["tokens"] or pair["drafted"]["drafted"] == 0:
            raise RuntimeError("tokens differ or the drafter did not execute")
    summary = dict(format=args.format, kv=args.kv, equality_pairs=len(pairs),
                   sampled_pairs=sum(r["seed"] is not None for r in pairs), tokens=args.tokens,
                   speed_ratio_median=statistics.median(ratios), speed_ratio_range=[min(ratios), max(ratios)],
                   accepted=sum(r["drafted"]["accepted"] for r in pairs),
                   proposed=sum(r["drafted"]["drafted"] for r in pairs),
                   mean_tokens_per_round=sum(len(r["drafted"]["tokens"]) for r in pairs) /
                   sum(r["drafted"]["rounds"] for r in pairs),
                   sampling_check="same keyed tokens across seeds; stronger than marginal histogram agreement")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
