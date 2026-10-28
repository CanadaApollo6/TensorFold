"""Run full-model KV quality checks only when the desktop-safe admission guard allows them."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--kind", choices=("short", "long"), default="long")
    p.add_argument("--sequence", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--timeout", type=int, default=1200)
    args = p.parse_args()
    runner = Path("zig-out/bin/tf-llama-test")
    seqs = json.loads((args.prompts / args.kind / "sequences.json").read_text())
    seqs = [seqs[args.sequence]]
    selected = args.work / f"selected-{args.kind}-{args.sequence}"
    selected.mkdir(parents=True, exist_ok=True)
    (selected / "sequences.json").write_text(json.dumps(seqs) + "\n")
    context = max(map(len, seqs)) + 128
    label = f"{args.kind}-s{args.sequence}"
    capacity = subprocess.run([str(runner), "llama-capacity", str(args.model), "24576"],
                              text=True, capture_output=True, check=True)
    needs = {mode: int(need) + 512 for mode, length, need in re.findall(
        r"NEED mode=(\w+) context=(\d+) bytes=\d+ need_mib=(\d+)", capacity.stderr)
             if int(length) == context}
    if set(needs) != {"bf16", "q4"}:
        raise ValueError(f"missing allocator estimate for context {context}")
    args.receipts.mkdir(parents=True, exist_ok=True)
    codes = []
    for mode, bits in (("bf16", "16"), ("q4", "4")):
        log = args.receipts / f"full-{label}-{mode}.log"
        receipt = args.receipts / f"full-{label}-{mode}-vram.json"
        command = [sys.executable, "tools/zig/llama_gpu_run.py", "--need-mib", str(needs[mode]),
                   "--timeout", str(args.timeout), "--receipt", str(receipt), "--", str(runner),
                   "llama-kv-dump", str(args.model), str(selected), bits,
                   str(args.work / label / mode)]
        print(" ".join(command), flush=True)
        with log.open("w") as f:
            code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT).returncode
        codes.append(code)
        print(log.read_text(), flush=True)
    if any(codes):
        raise SystemExit(1)
    subprocess.run([sys.executable, "tools/zig/llama_kv_compare.py", str(args.work / label / "bf16"),
                    str(args.work / label / "q4"), "--out",
                    str(args.receipts / f"quality-{label}.json")], check=True)


if __name__ == "__main__":
    main()
