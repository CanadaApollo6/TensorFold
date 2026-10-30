"""Compare teacher-forced distributions and greedy output from llama-kv-dump."""
import argparse
import array
import json
import math
from pathlib import Path


def floats(path):
    out = array.array("f")
    with path.open("rb") as f:
        out.frombytes(f.read())
    if any(not math.isfinite(v) for v in out):
        raise ValueError(f"nonfinite logit in {path}")
    return out


def distributions(a, b, vocab):
    if len(a) != len(b) or len(a) % vocab:
        raise ValueError("logit shape mismatch")
    import numpy as np
    kl, flips = [], []
    for row in range(len(a) // vocab):
        x = np.asarray(a[row * vocab:(row + 1) * vocab], dtype=np.float64)
        y = np.asarray(b[row * vocab:(row + 1) * vocab], dtype=np.float64)
        xi, yi = int(x.argmax()), int(y.argmax())
        xz = float(x[xi] + np.log(np.exp(x - x[xi]).sum()))
        yz = float(y[yi] + np.log(np.exp(y - y[yi]).sum()))
        kl.append(max(0, float((np.exp(x - xz) * (x - xz - y + yz)).sum())))
        if xi != yi:
            xs, ys = np.partition(x, -2)[-2:], np.partition(y, -2)[-2:]
            flips.append({"row": row, "bf16_top": xi, "q4_top": yi,
                          "bf16_margin": float(max(xs) - min(xs)), "q4_margin": float(max(ys) - min(ys))})
    ordered = sorted(kl)
    return {"rows": len(kl), "mean_kl": sum(kl) / len(kl), "max_kl": max(kl),
            "worst_row": kl.index(max(kl)), "p99_kl": ordered[math.ceil(.99 * len(kl)) - 1],
            "teacher_top1_agreement": 1 - len(flips) / len(kl), "top1_flips": flips}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bf16", type=Path)
    p.add_argument("q4", type=Path)
    p.add_argument("--vocab", type=int, default=166912)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    results = []
    files = sorted(args.bf16.glob("logits-*.f32"))
    if not files:
        p.error("no bf16 distributions found")
    for f in files:
        entry = distributions(floats(f), floats(args.q4 / f.name), args.vocab)
        token_file = f.name.replace("logits-", "greedy-").replace(".f32", ".json")
        a = json.loads((args.bf16 / token_file).read_text())
        b = json.loads((args.q4 / token_file).read_text())
        if len(a) != 128 or len(b) != 128:
            raise ValueError("expected 128 greedy tokens")
        entry.update(sequence=f.stem, greedy_matching=sum(x == y for x, y in zip(a, b)),
                     greedy_first_divergence=next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
        results.append(entry)
    report = {"state": "PASS", "quality_gate": "p4-parity", "results": results,
              "greedy_equal": all(x["greedy_matching"] == 128 for x in results)}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
