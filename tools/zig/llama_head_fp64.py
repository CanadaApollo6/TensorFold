"""Compare both EXL3 heads on identical FP16 norm rows against chunked CPU FP64 dots."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from llama_reference import dump


def prepare(a):
    import numpy as np
    import torch
    from exllamav3 import Config, Model

    torch.set_grad_enabled(False)
    cfg = Config.from_directory(str(a.model))
    cfg.infer_params.no_reconstruct = True
    model = Model.from_config(cfg)
    norm, head = model.modules[-2:]
    device = torch.device("cuda:0")
    norm.load(device)
    head.load(device)
    cases = [(a.receipts, 2, 7), (a.receipts, 3, 110), (a.receipts / "div17", 0, -1)]
    rows = []
    labels = []
    exl_one = []
    exl_many = []
    inputs = []
    for directory, seq, pos in cases:
        file = directory / f"ref-52-s{seq}-layer51.f32"
        data = np.fromfile(file, dtype="<f4").reshape(-1, cfg.hidden_size)
        x = norm.forward(torch.from_numpy(data).unsqueeze(0).to(device), {}).half()
        row = x[:, pos:pos + 1] if pos >= 0 else x[:, -1:]
        rows.append(row.cpu())
        labels.append(f"seq{seq}-pos{pos}" if directory == a.receipts else "greedy-token17-prefix")
        exl_one.append(head.forward(row, {}, out_dtype=torch.float).cpu())
        exl_many.append(head.forward(x, {}, out_dtype=torch.float)[:, pos].cpu())
        inputs.append({"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                       "position": pos if pos >= 0 else len(data) - 1, "rows": len(data)})
    dump(torch.cat(rows, dim=1), a.out / "norm-input.f32")
    dump(torch.cat(exl_one, dim=1), a.out / "exl-one.f32")
    dump(torch.cat(exl_many, dim=0), a.out / "exl-many.f32")
    norm.unload()
    inner = head if hasattr(head, "reconstruct_columns") else head.inner
    k, n = inner.in_features, inner.out_features
    weights = np.lib.format.open_memmap(a.out / "head-fp16.npy", mode="w+", dtype="<f2", shape=(k, n))
    for start in range(0, n, 1024):
        end = min(n, start + 1024)
        chunk = torch.empty((k, end - start), dtype=torch.half, device=device)
        inner.reconstruct_columns(torch.arange(start, end, device=device), chunk, max_blocks=8)
        weights[:, start:end] = chunk.cpu().numpy()
        del chunk
    weights.flush()
    receipt = {"labels": labels, "inputs": inputs, "weight_shape": [k, n], "input_dtype": "fp16",
               "torch": torch.__version__, "exllamav3": importlib.metadata.version("exllamav3"),
               "config_sha256": hashlib.sha256((a.model / "config.json").read_bytes()).hexdigest(),
               "norm_input_sha256": hashlib.sha256((a.out / "norm-input.f32").read_bytes()).hexdigest(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (a.out / "inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    head.unload()


def analyze(a):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    meta = json.loads((a.out / "inputs.json").read_text())
    k, n = meta["weight_shape"]
    x = torch.from_numpy(np.fromfile(a.out / "norm-input.f32", dtype="<f4").reshape(-1, k)).double()
    w = np.load(a.out / "head-fp16.npy", mmap_mode="r")
    y = torch.empty((len(x), n), dtype=torch.float64)
    for start in range(0, n, 1024):
        end = min(n, start + 1024)
        chunk = torch.from_numpy(w[:, start:end].copy()).double()
        y[:, start:end] = x @ chunk
    y.numpy().astype("<f8").tofile(a.out / "fp64-logits.f64")
    outputs = {name: torch.from_numpy(np.fromfile(a.out / file, dtype="<f4").reshape(-1, n)).double()
               for name, file in [("engine", "engine.f32"), ("exl_one", "exl-one.f32"), ("exl_many", "exl-many.f32")]}
    result = []
    for i, label in enumerate(meta["labels"]):
        top = torch.topk(y[i], 2)
        case = {"case": label, "fp64_top": int(top.indices[0]), "fp64_top2_margin": float(top.values[0] - top.values[1])}
        for name, logits in outputs.items():
            t = torch.topk(logits[i], 2)
            err = (logits[i] - y[i]).abs()
            case[name] = {"max_abs_error": float(err.max()), "worst_column": int(err.argmax()),
                          "top": int(t.indices[0]), "top2_margin": float(t.values[0] - t.values[1]),
                          "fp64_winner_error": float(err[top.indices[0]])}
        case["closer_by_max_abs"] = min(outputs, key=lambda name: case[name]["max_abs_error"])
        result.append(case)
        print(json.dumps(case), flush=True)
    receipt = {"reference": "CPU float64 dot of identical fp16 norm inputs and reconstructed fp16 head weights",
               "chunk_columns": 1024, "cases": result, "inputs": meta}
    (a.out / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "analyze"])
    p.add_argument("model", type=Path)
    p.add_argument("receipts", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "analyze": analyze}[a.action](a)


if __name__ == "__main__":
    main()
