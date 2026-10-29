"""Dump exllamav3 teacher-forced Llama states, logits and a greedy continuation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("EXL3_INT8_GEMV", "0")


def prompts(model, out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    messages = [
        {"role": "system", "content": "You are a careful assistant. Explain your reasoning and answer clearly."},
        {"role": "user", "content": (
            "A research group has one graphics card shared by a training job, a desktop, and an inference engine. "
            "The card has twenty four gigabytes of memory. The desktop must always have two gigabytes free. "
            "The training job holds fourteen gigabytes and cannot be interrupted. The inference engine needs "
            "fourteen gigabytes for its weights and one gigabyte for its context cache. Describe a safe plan "
            "for testing the engine while training continues, then for measuring full model correctness and "
            "speed after training finishes. Explain how you would compare the engine with a trusted reference "
            "on identical tokens and weights. Include a short checklist that distinguishes measurements "
            "actually completed from checks blocked by insufficient memory. Keep the answer practical. "
            "For the speed test, explain why prompt processing must be measured separately from token generation, "
            "and why warmup runs, repeated measurements, and token equality checks are necessary."
        )},
    ]
    chat = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(chat, "keys"):
        chat = chat["input_ids"]
    seqs = [[166100], [166100] + tok.encode("The capital of France is", add_special_tokens=False),
            [166100] + tok.encode("def add(a, b):\n    return a + b\n", add_special_tokens=False), chat]
    if any(s[0] != 166100 for s in seqs):
        raise ValueError("every sequence must begin with BOS 166100")
    (out / "sequences.json").write_text(json.dumps(seqs) + "\n")
    (out / "chat.txt").write_text(tok.decode(chat))
    print(json.dumps({"sequence_lengths": list(map(len, seqs)), "chat_tokens": len(chat)}), flush=True)
    return seqs


def guard(model, layers, context, dense):
    import struct

    row = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "--id=0"
    ], text=True).strip()
    cfg = json.loads((model / "config.json").read_text())
    need = 0
    for file in model.glob("*.safetensors"):
        with file.open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, t in header.items():
            if name == "__metadata__" or name.startswith("model.embed_tokens"):
                continue
            if name.startswith("model.layers.") and int(name.split(".")[2]) >= layers:
                continue
            need += t["data_offsets"][1] - t["data_offsets"][0]
    need += layers * context * cfg["num_key_value_heads"] * cfg["head_dim"] * 4
    need += 1536 << 20
    need += cfg["vocab_size"] * cfg["hidden_size"] * 2 if dense else 0
    free = int(row) << 20
    if need + (2 << 30) > free:
        raise SystemExit(f"BLOCKED GPU needs {need / 2**20:.0f} MiB plus 2048 MiB reserve; free {row} MiB")
    print(json.dumps({"gpu_free_mib": int(row), "estimated_peak_mib": need / 2**20}), flush=True)


def head_fn(head, device, dense):
    """The head as exllamav3 runs it, or as fp32 sums over its reconstructed weights (no small-row kernel)."""
    import torch

    head.load(device)
    if not dense:
        return lambda x, params: head.forward(x, params, out_dtype=torch.float)
    w = (head if hasattr(head, "get_weight_tensor") else head.inner).get_weight_tensor()
    head.unload()
    torch.cuda.empty_cache()

    def project(x, params):
        x = x.float()
        return torch.cat([x @ w[:, j:j + 8192].float() for j in range(0, w.shape[1], 8192)], dim=-1)
    return project


def dump(tensor, path):
    tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tofile(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--layers", type=int, default=52)
    p.add_argument("--prefill-bench", action="store_true")
    p.add_argument("--kv", choices=["16", "4"], default="16")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--greedy", type=int, default=0)
    p.add_argument("--dense-head", action="store_true", help="FP32 diagnostic from reconstructed FP16 weights; use llama_head_fp64.py for the independent audit")
    p.add_argument("--sequences", type=Path, help="token ids to use, e.g. zig/tests/fixtures/llama/boundary.json")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids_file = a.out / "sequences.json"
    if a.sequences:
        ids_file.write_bytes(a.sequences.read_bytes())
    if a.prepare:
        prompts(a.model, a.out)
        return
    seqs = json.loads(ids_file.read_text()) if ids_file.exists() else prompts(a.model, a.out)
    cfg_json = json.loads((a.model / "config.json").read_text())
    if not 1 <= a.layers <= cfg_json["num_hidden_layers"]:
        raise ValueError("invalid layer subset")
    if any(not s or s[0] != cfg_json["bos_token_id"] for s in seqs):
        raise ValueError("missing BOS")
    guard(a.model, a.layers, max(map(len, seqs)) + a.greedy, a.dense_head)
    import torch
    from exllamav3 import Config, Model

    torch.set_grad_enabled(False)
    cfg = Config.from_directory(str(a.model))
    model = Model.from_config(cfg)
    cfg.infer_params.no_reconstruct = True
    if a.prefill_bench:
        from llama_exl_prefill import benchmark
        benchmark(model, seqs, a.out, a.kv)
        return
    device = torch.device("cuda:0")
    embed = model.modules[0]
    blocks = model.modules[1:1 + a.layers]
    norm, head = model.modules[-2:]
    torch.cuda.reset_peak_memory_stats()
    embed.load(torch.device("cpu"))
    project = head_fn(head, device, a.dense_head)
    for module in [*blocks, norm]:
        module.load(device)
    prefix = f"ref-{a.layers}"
    for si, seq in enumerate(seqs):
        params = {"attn_mode": "flash_attn_nc"}
        x = embed.forward(torch.tensor([seq]), params).to(device)
        for li, block in enumerate(blocks):
            x = block.forward(x, params)
            dump(x, a.out / f"{prefix}-s{si}-layer{li}.f32")
        logits = project(norm.forward(x, params), params)
        dump(logits, a.out / f"{prefix}-s{si}-logits.f32")
        print(f"reference sequence {si} positions {len(seq)} layers {a.layers}", flush=True)
    generated = []
    if a.greedy:
        # no cache: every step reruns the whole sequence through the path the teacher-forced logits use
        tokens = list(seqs[-1])
        for _ in range(a.greedy):
            params = {"attn_mode": "flash_attn_nc"}
            x = embed.forward(torch.tensor([tokens]), params).to(device)
            for block in blocks:
                x = block.forward(x, params)
            y = project(norm.forward(x[:, -1:], params), params)
            generated.append(int(y[0, -1].argmax()))
            tokens.append(generated[-1])
        (a.out / f"{prefix}-greedy.json").write_text(json.dumps(generated) + "\n")
    torch.cuda.synchronize()
    receipt = {
        "torch": torch.__version__, "exllamav3": importlib.metadata.version("exllamav3"),
        "layers": a.layers, "sequence_lengths": list(map(len, seqs)), "greedy_tokens": len(generated),
        "sequences_sha256": hashlib.sha256(ids_file.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((a.model / "config.json").read_bytes()).hexdigest(),
        "quant_config_sha256": hashlib.sha256((a.model / "quantization_config.json").read_bytes()).hexdigest(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "int8_gemv": os.environ["EXL3_INT8_GEMV"], "no_reconstruct": True, "dense_head": a.dense_head,
        "attention_dtype": "float16", "residual_dtype": "float32",
    }
    (a.out / f"{prefix}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    for module in [norm, *reversed(blocks), embed] + ([] if a.dense_head else [head]):
        module.unload()


if __name__ == "__main__":
    main()
