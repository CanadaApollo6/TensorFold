"""Measure exllamav3 cache drift with TensorFold's P4 tail-rewrite and greedy protocol."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from llama_kv_compare import distributions, floats

SCORED_ROWS = 512
GREEDY_TOKENS = 128
CHUNK = 2048


def config_pair(value):
    try:
        pair = tuple(map(int, value.split(',')))
        if len(pair) != 2 or any(n < 2 or n > 8 for n in pair):
            raise ValueError()
        return pair
    except ValueError:
        raise argparse.ArgumentTypeError('expected k_bits,v_bits, each from 2 to 8') from None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'prompts', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sequence', type=int, choices=(0, 1, 2), default=0)
    p.add_argument('--configs', type=config_pair, nargs='+', default=[(4, 4), (6, 6), (8, 8), (8, 4)])
    return p


class Exl3:
    def __init__(self, directory, capacity):
        os.environ['EXL3_INT8_GEMV'] = '0'
        import torch
        from exllamav3 import Cache, CacheLayer_fp16, CacheLayer_quant, Config, Model
        self.torch = torch
        self.Cache, self.fp16, self.quant = Cache, CacheLayer_fp16, CacheLayer_quant
        torch.set_grad_enabled(False)
        cfg = Config.from_directory(str(directory))
        self.model = Model.from_config(cfg)
        cfg.infer_params.no_reconstruct = True
        self.capacity = capacity
        self.loaded = False
        self.metadata = dict(exllamav3=importlib.metadata.version('exllamav3'), torch=torch.__version__,
                             int8_gemv=os.environ['EXL3_INT8_GEMV'], no_reconstruct=True,
                             attention='flash_attn', chunk=CHUNK, max_output_size=1, reserve_per_device=[2.0],
                             bc_attn=os.environ.get('EXL3_BC_ATTN', '1'),
                             qc_staging=os.environ.get('EXL3_QC_STAGING', 'default'))

    def build_cache(self, bits):
        kwargs = dict(k_bits=bits[0], v_bits=bits[1]) if bits else {}
        return self.Cache(self.model, max_num_tokens=self.capacity, max_batch_size=1,
                          layer_type=self.quant if bits else self.fp16, **kwargs)

    def activate(self, cache):
        if not self.loaded:
            self.model.load(progressbar=True, max_chunk_size=CHUNK, max_output_size=1, reserve_per_device=[2.0])
            self.loaded = True
        else:
            # Cache attachment after load does not allocate storage in exllamav3 1.6.5.
            if cache.recurrent_layers:
                raise ValueError('P4 reference requires a model with transformer KV only')
            for module in self.model.get_cache_layers():
                for instance in self.model.get_layer_instances(module.layer_idx):
                    cache.layers[instance].alloc(module.device)
            cache.initialized = True

    def release(self, cache):
        self.torch.cuda.synchronize()
        # Captured attention graphs retain the cache tensors even after CacheLayer.free().
        for module in self.model.get_cache_layers():
            for instance in self.model.get_layer_instances(module.layer_idx):
                module.bc_attn.pop(id(cache.layers[instance]), None)
        for layer in cache.layers.values():
            layer.free()
        cache.initialized = False
        cache.detach_from_model()
        gc.collect()
        self.torch.cuda.empty_cache()

    def params(self, cache, pos):
        return dict(attn_mode='flash_attn', cache=cache, batch_shape=(1, self.capacity), past_len=pos,
                    last_tokens_only=1)

    def prefill(self, cache, seq):
        ids = self.torch.tensor([seq])
        for pos in range(0, len(seq), CHUNK):
            self.model.prefill(ids[:, pos:pos + CHUNK], self.params(cache, pos))

    def step(self, cache, token, pos):
        logits = self.model.forward(self.torch.tensor([[token]]), self.params(cache, pos))
        return logits[0, -1].detach().float().cpu().contiguous().numpy().astype('<f4')

    def close(self):
        self.model.unload()
        gc.collect()
        self.torch.cuda.empty_cache()


def measure(backend, cache, seq, folder, vocab):
    folder.mkdir(parents=True, exist_ok=True)
    backend.activate(cache)
    backend.prefill(cache, seq)
    # Score original prompt tokens after a full prefill, rewriting the resident tail as llama-kv-pair does.
    with (folder / 'logits-0.f32').open('wb') as output:
        for pos in range(len(seq) - SCORED_ROWS, len(seq)):
            row = backend.step(cache, seq[pos], pos)
            if len(row) != vocab:
                raise ValueError('logit vocabulary mismatch')
            row.tofile(output)
    row = backend.step(cache, seq[-1], len(seq) - 1)
    tokens = [int(row.argmax())]
    for i in range(1, GREEDY_TOKENS):
        row = backend.step(cache, tokens[-1], len(seq) + i - 1)
        tokens.append(int(row.argmax()))
    (folder / 'greedy-0.json').write_text(json.dumps(tokens) + '\n')
    return tokens


def write_report(out, report):
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def run(args):
    seqs = json.loads(args.prompts.read_text())
    cfg = json.loads((args.model / 'config.json').read_text())
    if (not isinstance(seqs, list) or len(seqs) != 3 or any(not isinstance(s, list) for s in seqs)
            or len({tuple(s) for s in seqs}) != 3):
        raise ValueError('three distinct P4 histories required')
    seq = seqs[args.sequence]
    vocab = cfg['vocab_size']
    if len(seq) < SCORED_ROWS or seq[0] != cfg['bos_token_id']:
        raise ValueError('P4 history needs BOS and at least 512 positions')
    if any(type(t) is not int or not 0 <= t < vocab for t in seq):
        raise ValueError('invalid prompt token ID')
    capacity = (len(seq) + GREEDY_TOKENS + 255) // 256 * 256
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(state='RUNNING', diagnostic=True, sequence=args.sequence, history_tokens=len(seq), vocab=vocab,
                  capacity=capacity, reference='fp16', reference_bar=dict(mean_kl=.01, max_kl=.1), configs={},
                  prompts_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
                  sequence_sha256=hashlib.sha256(json.dumps(seq).encode()).hexdigest(),
                  model_config_sha256=hashlib.sha256((args.model / 'config.json').read_bytes()).hexdigest(),
                  protocol=dict(scored_positions=[len(seq) - SCORED_ROWS, len(seq) - 1], rows=SCORED_ROWS,
                                teacher_forced='original prompt tokens, one forward per position after full prefill',
                                greedy_tokens=GREEDY_TOKENS, greedy_tail_rewrite=True, stop_on_eos=False,
                                greedy_repetitions=1, prefill_passes_per_config=1))
    backend = Exl3(args.model, capacity)
    report['runtime'] = backend.metadata
    write_report(args.out, report)
    try:
        cache = backend.build_cache(None)
        try:
            reference_tokens = measure(backend, cache, seq, args.out / 'fp16', vocab)
        finally:
            backend.release(cache)
        del cache
        reference = floats(args.out / 'fp16/logits-0.f32')
        for bits in dict.fromkeys(args.configs):
            name = f'k{bits[0]}-v{bits[1]}'
            entry = dict(k_bits=bits[0], v_bits=bits[1], directory=name)
            report['configs'][name] = entry
            try:
                cache = backend.build_cache(bits)
            except (AssertionError, ValueError, TypeError, NotImplementedError) as error:
                entry.update(state='SKIPPED', reason=f'{type(error).__name__}: {error}')
                write_report(args.out, report)
                continue
            try:
                tokens = measure(backend, cache, seq, args.out / name, vocab)
            finally:
                backend.release(cache)
            del cache
            entry.update(distributions(reference, floats(args.out / name / 'logits-0.f32'), vocab))
            entry.update(state='MEASURED', greedy_matching=sum(a == b for a, b in zip(reference_tokens, tokens)),
                         greedy_first_divergence=next((i for i, (a, b) in enumerate(zip(reference_tokens, tokens))
                                                       if a != b), None))
            write_report(args.out, report)
        report['state'] = 'COMPLETE'
        write_report(args.out, report)
    except Exception as error:
        report.update(state='ERROR', error=f'{type(error).__name__}: {error}')
        write_report(args.out, report)
        raise
    finally:
        backend.close()
    return report


def main():
    report = run(parser().parse_args())
    measured = sum(c['state'] == 'MEASURED' for c in report['configs'].values())
    print(f'exl3 P4 diagnostic complete: {measured} measured, {len(report["configs"]) - measured} skipped')


if __name__ == '__main__':
    main()
