"""Five alternating warm Llama benchmark runs with matched token checks and GPU admission."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("refs", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = []
    tokens = None
    for rep in range(5):
        for label, binary in [("before", a.before), ("after", a.after)]:
            receipt = a.out / f"{label}-{rep}-vram.json"
            cmd = [sys.executable, "tools/zig/llama_gpu_run.py", "--need-mib", "17000", "--timeout", "180",
                   "--receipt", str(receipt), "--", str(binary.resolve()), "llama-bench-once", str(a.model), str(a.refs)]
            result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (a.out / f"{label}-{rep}.log").write_text(result.stdout)
            if result.returncode:
                raise SystemExit(result.stdout)
            emitted = re.search(r"BENCH_TOKENS (\[.*\])", result.stdout)
            if not emitted:
                raise SystemExit("missing timed-work token receipt, rebuild both binaries with this harness")
            actual = json.loads(emitted.group(1))
            if len(actual) != 128 or (tokens is not None and actual != tokens):
                raise SystemExit("benchmark tokens differ")
            tokens = actual
            match = re.search(r"median_tok_s=([0-9.]+)", result.stdout)
            runs.append({"side": label, "rep": rep, "tok_s": float(match.group(1)), "command": cmd})
            print(json.dumps(runs[-1]), flush=True)
    summary = {"runs": runs, "tokens": tokens, "warmup_per_run": 1, "repetitions_per_run": 1,
               "binaries": {label: hashlib.sha256(binary.read_bytes()).hexdigest()
                            for label, binary in [("before", a.before), ("after", a.after)]}}
    for label in ["before", "after"]:
        rates = [r["tok_s"] for r in runs if r["side"] == label]
        summary[label] = {"median_tok_s": statistics.median(rates), "range": [min(rates), max(rates)]}
    (a.out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
