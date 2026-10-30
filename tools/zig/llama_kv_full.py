"""Compare late logits and paired decode timings with each prompt prefilled once per KV format."""
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--sequence", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    # CUDA context and library workspaces sit outside the allocator estimate; P3 on an A100 peaked ~240 MiB above +512.
    p.add_argument("--driver-mib", type=int, default=1024)
    args = p.parse_args()
    seqs = json.loads(args.prompts.read_text())
    if len(seqs) != 3 or len({tuple(s) for s in seqs}) != 3 or args.reps < 5:
        p.error("three distinct histories and at least five pairs required")
    seq = seqs[args.sequence]
    if args.sweep and len(seq) != 100000:
        p.error("sweep requires 100000 prompt tokens")
    if len(seq) < 512:
        p.error("history must have at least 512 positions")
    args.work.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=True)
    selected = args.work / "selected"
    selected.mkdir(exist_ok=True)
    (selected / "sequences.json").write_text(json.dumps([seq]) + "\n")
    context = max(256, len(seq) + 128)
    capacity = subprocess.run([args.runner, "llama-capacity", args.model, "49152", str(context)],
                              capture_output=True, text=True, check=True)
    cap = capacity.stdout + capacity.stderr
    (args.receipts / "capacity.log").write_text(cap)
    match = re.search(rf"NEED mode=pair context={context} bytes=\d+ need_mib=(\d+)", cap)
    if not match:
        raise ValueError("missing paired allocator estimate")
    cmd = [sys.executable, ROOT / "tools/zig/llama_gpu_run.py", "--need-mib", int(match[1]) + args.driver_mib,
           "--timeout", args.timeout, "--receipt", args.receipts / "pair-vram.json", "--", args.runner,
           "llama-kv-sweep" if args.sweep else "llama-kv-pair", args.model, selected, args.work, args.reps]
    result = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True, text=True)
    (args.receipts / "pair.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("paired resident-cache run failed; see pair.log")
    records = json.loads((args.work / "timings.json").read_text())
    prefill = [r for r in records if r["rep"] is None]
    lengths = [100000, 60000, 20000] if args.sweep else [len(seq)]
    results = []
    for length in lengths:
        work = args.work / str(length) if args.sweep else args.work
        runs = [r for r in records if r["rep"] is not None and r.get("history_tokens", length) == length]
        expected = [(mode, rep) for rep in range(args.reps)
                    for mode in (["bf16", "q4"] if rep % 2 == 0 else ["q4", "bf16"])]
        if [(r["mode"], r["rep"]) for r in runs] != expected or any(r["decode_tok_s"] <= 0 for r in runs):
            raise ValueError("missing paired timings")
        for mode in ["bf16", "q4"]:
            tokens = json.loads((work / mode / "0/greedy-0.json").read_text())
            if len(tokens) != 128 or any(json.loads((work / mode / str(rep) / "greedy-0.json").read_text()) != tokens
                                        for rep in range(1, args.reps)):
                raise ValueError("timed runs changed tokens")
        quality = args.receipts / f"quality-{length}.json"
        subprocess.run([sys.executable, ROOT / "tools/zig/llama_kv_compare.py", work / "bf16/0",
                        work / "q4/0", "--out", quality], check=True)
        summary = dict(history_tokens=length, scored_positions=512, runs=runs, quality=json.loads(quality.read_text()))
        for mode in ["bf16", "q4"]:
            rates = [r["decode_tok_s"] for r in runs if r["mode"] == mode]
            summary[mode] = dict(median=statistics.median(rates), spread=[min(rates), max(rates)])
        ratios = [next(r["decode_tok_s"] for r in runs if r["rep"] == i and r["mode"] == "q4") /
                  next(r["decode_tok_s"] for r in runs if r["rep"] == i and r["mode"] == "bf16")
                  for i in range(args.reps)]
        summary["paired_speed_ratio"] = dict(median=statistics.median(ratios), spread=[min(ratios), max(ratios)])
        results.append(summary)
    summary = dict(state="PASS", sequence=args.sequence, history_tokens=len(seq), prefill=prefill,
                   runs=[r for r in records if r["rep"] is not None], lengths=results, prefill_passes=2, decode_forwards=2 * args.reps * 127 * len(lengths))
    (args.receipts / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
