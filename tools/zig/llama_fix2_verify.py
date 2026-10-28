"""Verify late Llama cache slots and per-position bounds with guarded realistic controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CONTROLS = ("clean", "zeros", "rope_theta", "norm_plus_one", "swapped_kv")


def guarded(args, name, need, command, expected=0):
    cmd = [sys.executable, "tools/zig/llama_gpu_run.py", "--need-mib", str(need),
           "--timeout", "180", "--receipt", str(args.out / f"{name}-vram.json"), "--", *map(str, command)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out / f"{name}.log").write_text(result.stdout)
    print(f"{name}: exit {result.returncode}", flush=True)
    if result.returncode != expected or (expected and "FAIL llama-layer-control: LlamaLayerDrift" not in result.stdout):
        raise SystemExit(result.stdout)
    receipt = json.loads((args.out / f"{name}-vram.json").read_text())
    if receipt["minimum_free_mib"] < 2048 or receipt["peak_process_vram_mib"] > need:
        raise SystemExit(f"{name}: memory estimate or reserve failed")
    return receipt


def clean_metrics(args):
    import numpy as np

    worst = {}
    checked = 0
    seqs = json.loads((args.refs / "sequences.json").read_text())
    for si, seq in enumerate(seqs):
        for li in range(12):
            filename = f"12-s{si}-layer{li}.f32"
            ref = np.fromfile(args.refs / f"ref-{filename}", dtype="<f4").reshape(len(seq), -1).astype("f8")
            got = np.fromfile(args.out / "layers12" / f"engine-{filename}", dtype="<f4").reshape(ref.shape)
            diff = got - ref
            peak = abs(ref).max(axis=1)
            values = {
                "relative_l2": np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-15),
                "scaled_max": abs(diff).max(axis=1) / np.maximum(peak, 1e-30),
                "max_bound_utilization": abs(diff).max(axis=1) / (1e-4 + 0.03 * peak),
            }
            for metric, value in values.items():
                pos = int(value.argmax())
                if metric not in worst or value[pos] > worst[metric]["value"]:
                    worst[metric] = {"value": float(value[pos]), "sequence": si, "layer": li, "position": pos}
            if (values["relative_l2"] > 0.03).any() or (values["max_bound_utilization"] > 1).any():
                raise SystemExit("clean per-position bounds failed")
            checked += len(seq)
    return {"layer_positions": checked, "worst": worst,
            "relative_l2_margin_percent": 100 * (0.03 / worst["relative_l2"]["value"] - 1),
            "scaled_max_margin_percent": 100 * (0.03 / worst["scaled_max"]["value"] - 1)}


def control_metrics(args, name):
    rows = [json.loads(line) for line in (args.out / f"control-{name}.jsonl").read_text().splitlines()]
    l2 = max(rows, key=lambda row: row["relative_l2"])
    scaled = max(rows, key=lambda row: row["scaled_max"])
    failed_l2 = sum(row["relative_l2"] > 0.03 for row in rows)
    failed_max = sum(row["max_abs"] > 1e-4 + 0.03 * row["ref_peak"] for row in rows)
    if (name == "clean") != (failed_l2 == 0 and failed_max == 0):
        raise SystemExit(f"{name}: control did not separate")
    return {"worst_relative_l2": l2, "worst_scaled_max": scaled,
            "relative_l2_bound": 0.03, "max_error_fraction": 0.03, "max_error_floor": 1e-4,
            "failed_l2_positions": failed_l2, "failed_max_positions": failed_max}


def full_checks(args):
    probe = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0"
    ], text=True).strip()
    free = int(probe)
    if free < 17000 + 2048:
        cmd = [sys.executable, "tools/zig/llama_gpu_run.py", "--need-mib", "17000", "--receipt",
               str(args.out / "full52-vram.json"), "--", str(args.binary), "llama-logits",
               str(args.model), str(args.refs / "dense"), str(args.out / "full52")]
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (args.out / "full52-blocked.log").write_text(result.stdout)
        if "BLOCKED" not in result.stdout:
            raise SystemExit("full-model admission changed; rerun verification")
        return {"state": "BLOCKED", "free_mib": free, "checks": ["52-layer logits", "128-token greedy", "five alternating speed pairs"]}
    guarded(args, "full52-logits", 17000, [args.binary, "llama-logits", args.model, args.refs / "dense", args.out / "full52"])
    guarded(args, "full52-greedy", 17000, [args.binary, "llama-greedy", args.model, args.refs / "dense", "128", args.out / "full52-greedy.json"])
    cmd = [sys.executable, "tools/zig/llama_fix_bench.py", str(args.refs / "fix1/tf-llama-before"),
           str(args.binary), str(args.model), str(args.refs), str(args.out / "bench")]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out / "bench.log").write_text(result.stdout)
    if result.returncode:
        raise SystemExit(result.stdout)
    return {"state": "PASS", "free_mib": free, "bench": json.loads((args.out / "bench/comparison.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("refs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    vram = []
    vram.append(guarded(args, "boundary", 5200, [args.binary, "llama-boundary", args.model, args.refs / "fix1/boundary", args.out / "boundary"]))
    vram.append(guarded(args, "layers12", 4600, [args.binary, "llama-layers", args.model, args.refs, "12", args.out / "layers12"]))
    controls = {}
    for name in CONTROLS:
        vram.append(guarded(args, f"control-{name}", 4600, [args.binary, "llama-layer-control", args.model,
                            args.refs, name, args.out / f"control-{name}.jsonl"], int(name != "clean")))
        controls[name] = control_metrics(args, name)
    summary = {"binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "clean": clean_metrics(args), "controls": controls,
               "minimum_free_mib": min(run["minimum_free_mib"] for run in vram),
               "maximum_process_vram_mib": max(run["peak_process_vram_mib"] for run in vram),
               "maximum_observed_free_mib": max(run["initial_free_mib"] for run in vram),
               "full_model": full_checks(args)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
