"""Run the full production lane check under the GPU admission guard and summarize paired measurements."""
from llama_draft_stats import speed
from llama_sample_gate import check
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay-runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-replay")
    p.add_argument("--phase", choices=["basic", "speed"], default="basic")
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--format", choices=["bf16", "affine4"], required=True)
    p.add_argument("--kv", choices=["16", "4"], required=True)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--driver-mib", type=int, default=512)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    args = p.parse_args()
    prompts = json.loads(args.prompts.read_text())
    if len({tuple(ids) for ids in prompts}) != 3:
        p.error("three distinct prompts required")
    if len(prompts) != 3 or args.tokens < 256 or args.reps < (1 if args.phase == "basic" else 5) or args.samples < 16:
        p.error("the gate needs three prompts, 256 tokens, 16 coupled seeds per fixed position and five benchmark pairs")
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
    scratch = tempfile.TemporaryDirectory(prefix="llama-draft-cache-")
    output = Path(scratch.name) / "pairs.json"
    command = [sys.executable, ROOT / "tools/zig/llama_gpu_run.py", "--need-mib", str(need),
               "--timeout", str(args.timeout), "--receipt", args.out / "vram.json", "--",
               args.runner, "llama-draft-full" if args.phase == "basic" else "llama-draft-speed", args.model, args.draft, args.prompts, output,
               str(args.tokens), str(args.samples), str(args.reps)]
    command = list(map(str, command))
    (args.out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = dict(os.environ, TF_LLAMA_DRAFT_FORMAT=args.format, TF_LLAMA_KV_BITS=args.kv)
    with (args.out / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((args.out / "run.log").read_text())
        raise SystemExit(result.returncode)
    if args.phase == "basic" and (args.out / "run.log").read_text().count("FORCED full_accept_then_reject") != 3:
        raise RuntimeError("missing full-acceptance then rejection coverage")
    pairs = json.loads(output.read_text())
    (args.out / "pairs.json").write_text(json.dumps(pairs) + "\n")
    expected = {(prompt, seed, variant, rep)
                for prompt in range(3) for seed in [None]
                for variant in [0 if seed is None else seed % 4]
                for rep in range(args.reps if seed is None else 1)}
    actual = {(r["prompt"], r["seed"], r["variant"], r["rep"]) for r in pairs}
    if actual != expected or len(actual) != len(pairs):
        raise RuntimeError("missing or duplicate equality/timing cells")
    greedy = [r for r in pairs if r["seed"] is None]
    if any(len(r["plain"]["tokens"]) < 2 for r in greedy):
        raise RuntimeError("greedy timing requires actual decode forwards")
    ratios = [r["plain"]["seconds"] / r["drafted"]["seconds"] for r in greedy]
    for pair in pairs:
        if pair["plain"]["tokens"] != pair["drafted"]["tokens"] or pair["drafted"]["drafted"] == 0:
            raise RuntimeError("tokens differ or the drafter did not execute")
    if args.phase == 'speed':
        medians = [statistics.median([r['plain']['seconds'] / r['drafted']['seconds'] for r in greedy if r['prompt'] == i]) for i in range(3)]
        summary = dict(state='PASS' if min(medians) > 1 else 'FAIL', pairs=len(pairs), speed=speed(ratios), medians_by_prompt=medians)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if summary['state'] != 'PASS': raise RuntimeError('draft decode speed did not improve')
        scratch.cleanup()
        return
    cache = output.with_suffix(".json.distribution.jsonl")
    cached = [json.loads(line) for line in cache.read_text().splitlines()]
    expected_cells = {(p, r, v) for p in range(3) for r in [0, 3, 6] for v in range(4)}
    if len(cached) != 36 or {(r['prompt'], r['row'], r['variant']) for r in cached} != expected_cells or any(r['coupled'] != 16 for r in cached):
        raise RuntimeError("missing cached sampling cells")
    if any(len(row['target']) != 166912 or len(row['draft']) != 166912 for row in cached):
        raise RuntimeError('cached logits must cover the complete Nanbeige vocabulary')
    for row in cached:
        for name in ['prompt', 'row', 'variant', 'coupled']: row.pop(name)
        row['draws'] = 32768
    summary = dict(format=args.format, kv=args.kv, equality_pairs=len(pairs),
                   sampled_pairs=sum(r["seed"] is not None for r in pairs), tokens=args.tokens,
                   speed_ratio_median=statistics.median(ratios), speed_ratio_range=[min(ratios), max(ratios)],
                   accepted=sum(r["drafted"]["accepted"] for r in pairs),
                   proposed=sum(r["drafted"]["drafted"] for r in pairs),
                   mean_tokens_per_round=sum(len(r["drafted"]["tokens"]) - 1 for r in pairs) /
                   sum(r["drafted"]["rounds"] for r in pairs),
                   sampling_check=check(args.replay_runner, cached),
                   speed_ratio=dict(median=statistics.median(ratios)))
    scratch.cleanup()
    summary["stopped_pairs"] = sum(r["drafted"]["stopped"] for r in pairs)
    summary["sampling_variants"] = 4
    summary["sampling_matrix"] = [dict(temperature=t, top_k=k, top_p=p,
                                       presence=0.7 if i == 3 else 0, frequency=0.3 if i == 3 else 0)
                                  for i, (t, k, p) in enumerate(zip([0.3, 0.7, 1, 1.3], [0, 3, 20, 50], [1, .8, .95, .9]))]
    summary["speed_by_prompt"] = []
    for prompt in range(3):
        group = [r for r in greedy if r["prompt"] == prompt]
        values = [r["plain"]["seconds"] / r["drafted"]["seconds"] for r in group]
        if any(r["plain"]["tokens"] != group[0]["plain"]["tokens"] for r in group):
            raise RuntimeError("greedy timing changed tokens across repetitions")
        summary["speed_by_prompt"].append(dict(prompt=prompt, pairs=len(group), **dict(median=statistics.median(values), spread=[min(values), max(values)])))
    summary["mean_acceptance_length"] = 1 + summary["accepted"] / sum(r["drafted"]["rounds"] for r in pairs)
    summary["state"] = "PASS" if summary["mean_acceptance_length"] > 1 else "FAIL"
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["state"] != "PASS":
        raise RuntimeError("mean acceptance must exceed one")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
