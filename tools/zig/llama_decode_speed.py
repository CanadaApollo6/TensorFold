"""Matched warm plain decode versus exllamav3 with identical prompt IDs and forward counts."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def reference(model_dir, refs, out):
    import torch
    from exllamav3 import Config, Model, Cache
    torch.set_grad_enabled(False)
    cfg = Config.from_directory(str(model_dir))
    cfg.infer_params.no_reconstruct = True
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=1)
    model.load(device='cuda:0', max_chunk_size=4096, max_output_size=1)
    sequences = json.loads((refs / 'sequences.json').read_text())
    rows = []
    for prompt in sequences:
        prompt = prompt[:32]
        def generate():
            params = dict(attn_mode='flash_attn', cache=cache, batch_shape=(1, 4096), past_len=0, last_tokens_only=1)
            y = model.forward(torch.tensor([prompt], dtype=torch.long), params)
            torch.cuda.synchronize()
            start = time.perf_counter()
            token = int(y[0, -1].argmax())
            tokens = [token]
            for pos in range(127):
                params['past_len'] = len(prompt) + pos
                y = model.forward(torch.tensor([[token]], dtype=torch.long), params)
                token = int(y[0, -1].argmax())
                tokens.append(token)
            torch.cuda.synchronize()
            return tokens, 127 / (time.perf_counter() - start)
        warm, _ = generate()
        tokens, rate = generate()
        if warm != tokens: raise ValueError('exl3 warm/timed tokens differ')
        rows.append(dict(prompt=prompt, tokens=tokens, tok_s=rate, forwards=127))
    out.write_text(json.dumps(rows) + '\n')


def compare(runs):
    rows = []
    for prompt in sorted({r['prompt'] for r in runs}):
        result = dict(prompt=prompt)
        for side in ['tensorfold', 'exllamav3']:
            tokens = [r['tokens'] for r in runs if r['side'] == side and r['prompt'] == prompt]
            if not tokens or any(t != tokens[0] for t in tokens): raise ValueError('timed tokens changed across repetitions')
            rates = [r['tok_s'] for r in runs if r['side'] == side and r['prompt'] == prompt]
            if len(rates) != 5 or any(not math.isfinite(rate) or rate <= 0 for rate in rates): raise ValueError('five positive runs required')
            result[side] = dict(median=statistics.median(rates), spread=[min(rates), max(rates)])
        result['ratio'] = result['tensorfold']['median'] / result['exllamav3']['median']
        result['speed_state'] = 'PASS' if result['ratio'] >= 1 else 'FAIL'
        rows.append(result)
    return dict(state='PASS', speed_state='PASS' if all(r['ratio'] >= 1 for r in rows) else 'FAIL',
                nonblocking=True, runs=runs, prompts=rows, timed_forwards=127, warmups=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'refs', 'out']: p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runner', type=Path)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--reference-worker', action='store_true')
    a = p.parse_args()
    if a.reference_worker: return reference(a.model, a.refs, a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    sequences = json.loads((a.refs / 'sequences.json').read_text())
    runs = []
    for rep in range(5):
        sides = ['tensorfold', 'exllamav3'] if rep % 2 == 0 else ['exllamav3', 'tensorfold']
        for side in sides:
            tasks = range(len(sequences)) if side == 'tensorfold' else [None]
            for index in tasks:
                label = f'{side}-{rep}-{index}'
                if side == 'tensorfold':
                    refs = a.out / f'prompt-{index}'
                    refs.mkdir(exist_ok=True)
                    (refs / 'sequences.json').write_text(json.dumps([sequences[index]]) + '\n')
                    command = [a.runner, 'llama-bench-once', a.model, refs]
                    need = 17000
                else:
                    command = [a.python, __file__, '--reference-worker', '--model', a.model,
                               '--refs', a.refs, '--out', a.out / (label + '.json')]
                    need = 20000
                cmd = [sys.executable, ROOT / 'tools/zig/llama_gpu_run.py', '--need-mib', need,
                       '--receipt', a.out / (label + '-vram.json'), '--timeout', 600, '--', *command]
                proc = subprocess.run(list(map(str, cmd)), capture_output=True, text=True,
                                      env=dict(os.environ, TF_LLAMA_KV_BITS='16', EXL3_INT8_GEMV='0'))
                (a.out / (label + '.log')).write_text(proc.stdout + proc.stderr)
                if proc.returncode: raise RuntimeError(f'{label} failed')
                if side == 'tensorfold':
                    text = proc.stdout + proc.stderr
                    tokens = json.loads(re.search(r'BENCH_TOKENS (\[.*\])', text)[1])
                    rows = [dict(tokens=tokens, tok_s=float(re.search(r'median_tok_s=([0-9.]+)', text)[1]))]
                else:
                    rows = json.loads((a.out / (label + '.json')).read_text())
                for i, row in enumerate(rows):
                    if len(row['tokens']) != 128: raise ValueError('short timed work')
                    runs.append(dict(row, side=side, rep=rep, prompt=index if index is not None else i))
                    # Retain both exact token lists; model-fidelity gates own cross-engine tie policy.
                    runs[-1]['prompt'] = index if index is not None else i
    result = compare(runs)
    (a.out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
