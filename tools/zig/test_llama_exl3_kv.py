"""CPU-only P4 reference protocol, artifact, skip and diagnostic-exit checks."""
import array
import contextlib
import io
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import exl3_kv_kl as ref
import llama_kv_compare


class Row(array.array):
    def __new__(cls, values):
        return super().__new__(cls, 'f', values)

    def argmax(self):
        return max(range(len(self)), key=self.__getitem__)


class Fake:
    def __init__(self, directory, capacity):
        self.capacity = capacity
        self.metadata = dict(exllamav3='CPU fake')
        self.active = None
        self.prefills, self.calls, self.released = [], {}, []
        self.closed = False

    def build_cache(self, bits):
        if self.active is not None:
            raise AssertionError('previous cache still resident')
        if bits == (6, 6):
            raise NotImplementedError('fake unsupported bit pair')
        return SimpleNamespace(bits=bits, tokens=[])

    def activate(self, cache):
        self.active = cache
        self.calls[cache.bits] = []

    def release(self, cache):
        assert self.active is cache
        self.released.append(cache.bits)
        self.active = None

    def prefill(self, cache, seq):
        cache.tokens = list(seq)
        self.prefills.append(cache.bits)

    def step(self, cache, token, pos):
        self.calls[cache.bits].append((token, pos))
        if pos == len(cache.tokens):
            cache.tokens.append(token)
        else:
            cache.tokens[pos] = token
        winner = (sum(cache.tokens[:pos + 1]) + (cache.bits == (4, 4))) % 3
        return Row([8 if i == winner else 0 for i in range(3)])

    def close(self):
        assert self.active is None
        self.closed = True


def cpu_distributions(a, b, vocab):
    # A small independent oracle keeps the fake usable on hosts without numpy or torch.
    divergences, flips = [], 0
    for pos in range(0, len(a), vocab):
        x, y = a[pos:pos + vocab], b[pos:pos + vocab]
        lp = [v - math.log(sum(map(math.exp, x))) for v in x]
        lq = [v - math.log(sum(map(math.exp, y))) for v in y]
        divergences.append(sum(math.exp(p) * (p - q) for p, q in zip(lp, lq)))
        flips += x.index(max(x)) != y.index(max(y))
    return dict(rows=len(divergences), mean_kl=sum(divergences) / len(divergences), max_kl=max(divergences),
                p99_kl=sorted(divergences)[math.ceil(.99 * len(divergences)) - 1],
                teacher_top1_agreement=1 - flips / len(divergences))


class Exl3KV(unittest.TestCase):
    def fixture(self, root):
        (root / 'config.json').write_text(json.dumps(dict(vocab_size=3, bos_token_id=0, eos_token_id=1)))
        seqs = [[0] + [i] * 514 for i in [0, 1, 2]]
        (root / 'prompts.json').write_text(json.dumps(seqs))
        return ['--model', str(root), '--prompts', str(root / 'prompts.json'), '--out', str(root / 'out')], seqs

    def test_cli_layout_late_rows_and_independent_128_token_continuations(self):
        self.assertIs(ref.distributions, llama_kv_compare.distributions)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv, seqs = self.fixture(root)
            instances = []
            def build(*args):
                backend = Fake(*args)
                instances.append(backend)
                return backend
            scorer = ref.distributions if importlib.util.find_spec('numpy') else cpu_distributions
            with patch.object(ref, 'Exl3', side_effect=build), patch.object(ref, 'distributions', side_effect=scorer), \
                    patch.object(sys, 'argv', ['exl3_kv_kl', *argv, '--sequence', '1']), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                ref.main()
            backend, = instances
            self.assertTrue(backend.closed)
            self.assertEqual(backend.prefills, [None, (4, 4), (8, 8), (8, 4)])
            self.assertEqual(backend.released, backend.prefills)
            self.assertEqual(backend.capacity, 768)
            seq = seqs[1]
            for bits in backend.prefills:
                calls = backend.calls[bits]
                expected_scores = [(seq[pos], pos) for pos in range(3, 515)]
                self.assertEqual(calls[:512], expected_scores)
                self.assertEqual(calls[512], (seq[-1], 514))
                generated, history = [], list(seq)
                for i in range(128):
                    generated.append((sum(history) + (bits == (4, 4))) % 3)
                    history.append(generated[-1])
                self.assertEqual(calls[513:], [(generated[i - 1], 515 + i - 1) for i in range(1, 128)])
                folder = root / 'out' / ('fp16' if bits is None else f'k{bits[0]}-v{bits[1]}')
                self.assertEqual(json.loads((folder / 'greedy-0.json').read_text()), generated)
                logits = llama_kv_compare.floats(folder / 'logits-0.f32')
                self.assertEqual(len(logits), 512 * 3)
                for row, pos in enumerate(range(3, 515)):
                    winner = (sum(seq[:pos + 1]) + (bits == (4, 4))) % 3
                    self.assertEqual(list(logits[row * 3:row * 3 + 3]), [8 if i == winner else 0 for i in range(3)])
            report = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(report['state'], 'COMPLETE')
            self.assertEqual(report['sequence'], 1)
            self.assertEqual(report['protocol']['scored_positions'], [3, 514])
            skipped = report['configs']['k6-v6']
            self.assertEqual(skipped['state'], 'SKIPPED')
            self.assertIn('NotImplementedError', skipped['reason'])
            self.assertFalse((root / 'out/k6-v6').exists())
            quant = report['configs']['k4-v4']
            self.assertGreater(quant['mean_kl'], .01)
            self.assertGreater(quant['max_kl'], .1)
            self.assertGreater(quant['p99_kl'], .1)
            self.assertEqual(quant['teacher_top1_agreement'], 0)
            self.assertLess(quant['greedy_matching'], 128)
            self.assertEqual(quant['greedy_first_divergence'], 0)
            for name in ['k8-v8', 'k8-v4']:
                self.assertAlmostEqual(report['configs'][name]['mean_kl'], 0)
                self.assertEqual(report['configs'][name]['greedy_matching'], 128)
            self.assertEqual(stdout.getvalue(), 'exl3 P4 diagnostic complete: 3 measured, 1 skipped\n')

    def test_argument_validation_and_explicit_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv, _ = self.fixture(root)
            args = ref.parser().parse_args([*argv, '--configs', '8,4', '2,3'])
            self.assertEqual(args.configs, [(8, 4), (2, 3)])
            for values in [['--configs', '4'], ['--configs', '9,4'], ['--sequence', '3']]:
                with self.subTest(values=values), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    ref.parser().parse_args([*argv, *values])
            (root / 'prompts.json').write_text(json.dumps([[0] * 512] * 3))
            with patch.object(ref, 'Exl3') as backend, self.assertRaisesRegex(ValueError, 'distinct'):
                ref.run(args)
            backend.assert_not_called()

    def test_measurement_error_is_not_hidden_as_unsupported_config(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            argv, _ = self.fixture(root)
            backend = Fake(root, 768)
            def fail(cache, token, pos):
                raise RuntimeError('inference failed')
            backend.step = fail
            with patch.object(ref, 'Exl3', return_value=backend), self.assertRaisesRegex(RuntimeError, 'inference failed'):
                ref.run(ref.parser().parse_args(argv))
            self.assertTrue(backend.closed)
            self.assertEqual(backend.released, [None])
            self.assertEqual(json.loads((root / 'out/summary.json').read_text())['state'], 'ERROR')

    def test_loaded_model_replaces_cache_and_releases_captured_graph_storage(self):
        backend = ref.Exl3.__new__(ref.Exl3)
        backend.loaded = False
        backend.capacity = 768
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock(), empty_cache=Mock()))
        module = SimpleNamespace(layer_idx=0, device='device-0', bc_attn={})
        backend.model = SimpleNamespace(load=Mock(), unload=Mock(), get_cache_layers=lambda: [module],
                                        get_layer_instances=lambda _: [(0, 0)])
        caches = []
        for _ in range(2):
            layer = SimpleNamespace(alloc=Mock(), free=Mock())
            cache = SimpleNamespace(layers={(0, 0): layer}, recurrent_layers={}, initialized=False,
                                    detach_from_model=Mock())
            caches.append(cache)
            backend.activate(cache)
            module.bc_attn[id(layer)] = SimpleNamespace(retained_storage=object())
            self.assertEqual(backend.params(cache, 123)['past_len'], 123)
            backend.release(cache)
            self.assertFalse(module.bc_attn)
            layer.free.assert_called_once()
            cache.detach_from_model.assert_called_once()
            self.assertFalse(cache.initialized)
        backend.model.load.assert_called_once_with(progressbar=True, max_chunk_size=2048, max_output_size=1,
                                                   reserve_per_device=[2.0])
        caches[1].layers[(0, 0)].alloc.assert_called_once_with('device-0')
        backend.close()
        backend.model.unload.assert_called_once()


if __name__ == '__main__':
    unittest.main()
