"""Compare 512 late logits and five interleaved timings for one full-model history."""
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
    p.add_argument("--sequence", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    p.add_argument("--driver-mib", type=int, default=512)
    args = p.parse_args()
    seqs = json.loads(args.prompts.read_text())
    if len(seqs) != 3 or len({tuple(s) for s in seqs}) != 3 or args.reps < 5:
        p.error("three distinct histories and at least five pairs required")
    seq = seqs[args.sequence]
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
    needs = {mode: int(need) + args.driver_mib for mode, length, need in re.findall(
        r"NEED mode=(\w+) context=(\d+) bytes=\d+ need_mib=(\d+)", cap) if int(length) == context}
    if set(needs) != {"bf16", "q4"}:
        raise ValueError("missing allocator estimates")
    runs = []
    quality = args.receipts / "quality.json"
    for rep in range(args.reps):
        order = [("bf16", "16"), ("q4", "4")]
        if rep % 2:
            order.reverse()
        for mode, bits in order:
            out = args.work / mode / str(rep)
            cmd = [sys.executable, ROOT / "tools/zig/llama_gpu_run.py", "--need-mib", str(needs[mode]),
                   "--timeout", str(args.timeout), "--receipt", args.receipts / f"{mode}-{rep}-vram.json",
                   "--", args.runner, "llama-kv-dump", args.model, selected, bits, out,
                   "512" if rep == 0 else "0"]
            result = subprocess.run(list(map(str, cmd)), cwd=ROOT, capture_output=True, text=True)
            (args.receipts / f"{mode}-{rep}.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"{mode} repetition {rep} failed, see run log")
            match = re.search(r"decode_tok_s=([\d.]+)", result.stdout + result.stderr)
            if not match or float(match[1]) <= 0:
                raise ValueError("missing decode timing")
            tokens = json.loads((out / "greedy-0.json").read_text())
            if rep and tokens != json.loads((args.work / mode / "0/greedy-0.json").read_text()):
                raise ValueError("timed run changed tokens")
            runs.append(dict(mode=mode, rep=rep, decode_tok_s=float(match[1])))
        if rep == 0:
            comparison = subprocess.run([sys.executable, ROOT / "tools/zig/llama_kv_compare.py", args.work / "bf16/0",
                                         args.work / "q4/0", "--out", quality])
            if comparison.returncode:
                failed = dict(state="FAIL", runs=runs, scored_positions=512,
                              quality=json.loads(quality.read_text()) if quality.exists() else None)
                (args.receipts / "summary.json").write_text(json.dumps(failed, indent=2) + "\n")
                raise SystemExit("KV quality failed; remaining timing repetitions are NOT RUN")
    summary = dict(history_tokens=len(seq), scored_positions=512, runs=runs,
                   quality=json.loads(quality.read_text()))
    for mode in needs:
        rates = [r["decode_tok_s"] for r in runs if r["mode"] == mode]
        summary[mode] = dict(median=statistics.median(rates), spread=[min(rates), max(rates)])
    ratios = [next(r["decode_tok_s"] for r in runs if r["rep"] == i and r["mode"] == "q4") /
              next(r["decode_tok_s"] for r in runs if r["rep"] == i and r["mode"] == "bf16")
              for i in range(args.reps)]
    summary["paired_speed_ratio"] = dict(median=statistics.median(ratios), spread=[min(ratios), max(ratios)])
    (args.receipts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
