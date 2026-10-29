"""Time exllamav3's cached prompt path on the same IDs as TensorFold."""
import importlib.metadata
import json
import time


def benchmark(model, seqs, out, kv):
    import torch
    from exllamav3 import Cache, CacheLayer_fp16, CacheLayer_quant
    if len(seqs) != 1:
        raise ValueError('timing requires exactly one sequence')
    seq = seqs[0]
    capacity = (len(seq) + 255) // 256 * 256
    kwargs = dict(k_bits=4, v_bits=4) if kv == '4' else {}
    cache = Cache(model, max_num_tokens=capacity, max_batch_size=1,
                  layer_type=CacheLayer_quant if kv == '4' else CacheLayer_fp16, **kwargs)
    chunk = 2048
    model.load(progressbar=True, max_chunk_size=chunk, max_output_size=1, reserve_per_device=[2.0])
    ids = torch.tensor([seq])
    elapsed = []
    for rep in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        for pos in range(0, len(seq), chunk):
            end = min(pos + chunk, len(seq))
            params = dict(attn_mode='flash_attn', cache=cache, batch_shape=(1, capacity), past_len=pos,
                          last_tokens_only=1)
            if end == len(seq):
                logits = model.forward(ids[:, pos:end], params)
            else:
                model.prefill(ids[:, pos:end], params)
        torch.cuda.synchronize()
        if rep:
            elapsed.append(time.perf_counter() - started)
    seconds = sum(elapsed) / len(elapsed)
    report = dict(seconds=seconds, tokens=len(seq), tok_s=len(seq) / seconds, repetitions=elapsed,
                  chunk=chunk, cache_bits=int(kv), attention='flash_attn', torch=torch.__version__,
                  exllamav3=importlib.metadata.version('exllamav3'))
    (out / 'prefill.json').write_text(json.dumps(report, indent=2) + '\n')
    logits.detach().float().cpu().contiguous().numpy().astype('<f4').tofile(out / 'logits.f32')
    model.unload()
