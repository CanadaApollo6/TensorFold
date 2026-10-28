"""Build a bounded exllamav3 two-layer oracle at the default Llama context seams."""
import argparse
import hashlib
import json
from pathlib import Path

from llama_reference import dump, guard


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path("zig/tests/fixtures/llama/boundary.json").read_text())[0]
    seq = [source[0]] + [source[1 + i % (len(source) - 1)] for i in range(16383)]
    ids_file = a.out / "sequences.json"
    ids_file.write_text(json.dumps([seq]) + "\n")
    guard(a.model, 2, 16384, False)
    import torch
    from exllamav3 import Cache, Config, Model
    from exllamav3.modules.attn import prepare_for_attn

    torch.set_grad_enabled(False)
    cfg = Config.from_directory(str(a.model))
    cfg.infer_params.no_reconstruct = True
    model = Model.from_config(cfg)
    embed, blocks, norm, head = model.modules[0], model.modules[1:3], *model.modules[-2:]
    model.modules = [embed, *blocks, norm, head]
    cache = Cache(model, max_num_tokens=16384)
    device = torch.device("cuda:0")
    embed.load(torch.device("cpu"))
    for module in [*blocks, norm, head]:
        module.load(device)
    positions = [511, 512, 2047, 2048, 16383]
    ends = sorted(set([*range(128, 16385, 128), *[x + 1 for x in positions]]))
    start = 0
    for end in ends:
        ids = torch.tensor([seq[start:end]])
        params = {"attn_mode": "flash_attn", "batch_shape": (1, 16384), "cache": cache, "past_len": start}
        prepare_for_attn(ids, params)
        x = embed.forward(ids, params).to(device)
        for li, block in enumerate(blocks):
            x = block.forward(x, params)
            if end - 1 in positions:
                dump(x[:, -1], a.out / f"boundary-p{end - 1}-layer{li}.f32")
        if end - 1 in positions:
            y = head.forward(norm.forward(x[:, -1:], params), params, out_dtype=torch.float)
            dump(y, a.out / f"boundary-p{end - 1}-logits.f32")
            print(f"REFERENCE boundary position={end - 1} layers=2", flush=True)
        start = end
    params = {"attn_mode": "flash_attn_nc"}
    x = embed.forward(torch.tensor([[seq[0]]]), params).to(device)
    for block in blocks:
        x = block.forward(x, params)
    dump(head.forward(norm.forward(x, params), params, out_dtype=torch.float), a.out / "boundary-fresh-logits.f32")
    receipt = {"layers": 2, "context": 16384, "positions": positions, "heads": [56, 8],
               "reference_cache": "fp16", "engine_cache": "bf16", "prefill_chunk_max": 128,
               "sequences_sha256": hashlib.sha256(ids_file.read_bytes()).hexdigest(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (a.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
