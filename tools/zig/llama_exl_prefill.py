"""Time exllamav3's cached prompt path on the same IDs as TensorFold."""
import importlib.metadata
import json
import time


def kernel_table(events, tokens):
    """Self CUDA time per kernel name; exl3 GEMM template names carry the tile shape that autotune picked."""
    rows = []
    for e in events:
        us = getattr(e, 'self_device_time_total', None)
        us = getattr(e, 'self_cuda_time_total', 0) if us is None else us
        if us > 0:
            name = e.key
            kind = 'gemm' if 'exl3' in name or 'gemm' in name.lower() else 'attention' if 'attn' in name or 'flash' in name else 'other'
            rows.append(dict(kind=kind, name=name, calls=e.count, ms=us / 1000, us_per_token=us / tokens))
    rows.sort(key=lambda r: -r['ms'])
    totals = {kind: sum(r['ms'] for r in rows if r['kind'] == kind) for kind in ['gemm', 'attention', 'other']}
    return dict(totals_ms=totals, kernels=rows)


def benchmark(model, seqs, out, kv, profile=False):
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
    def once():
        for pos in range(0, len(seq), chunk):
            end = min(pos + chunk, len(seq))
            params = dict(attn_mode='flash_attn', cache=cache, batch_shape=(1, capacity), past_len=pos,
                          last_tokens_only=1)
            if end == len(seq):
                return model.forward(ids[:, pos:end], params)
            model.prefill(ids[:, pos:end], params)
    elapsed = []
    for rep in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = once()
        torch.cuda.synchronize()
        if rep:
            elapsed.append(time.perf_counter() - started)
    seconds = sum(elapsed) / len(elapsed)
    if profile:
        # One extra rep after the timed ones, so the profiler never touches the reported seconds.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            once()
            torch.cuda.synchronize()
        table = kernel_table(prof.key_averages(), len(seq))
        (out / 'exl3-kernels.json').write_text(json.dumps(table, indent=2) + '\n')
        (out / 'exl3-kernels.txt').write_text(''.join(f"{r['kind']:9} {r['calls']:6} {r['ms']:10.3f} ms {r['us_per_token']:8.3f} us/tok  {r['name']}\n" for r in table['kernels']))
    report = dict(seconds=seconds, tokens=len(seq), tok_s=len(seq) / seconds, repetitions=elapsed,
                  chunk=chunk, cache_bits=int(kv), attention='flash_attn', torch=torch.__version__,
                  exllamav3=importlib.metadata.version('exllamav3'))
    if profile:
        report['profile_ms'] = table['totals_ms']
    (out / 'prefill.json').write_text(json.dumps(report, indent=2) + '\n')
    logits.detach().float().cpu().contiguous().numpy().astype('<f4').tofile(out / 'logits.f32')
    model.unload()
