"""CPU checks for the deferred prompt receipts and job selection."""
import struct
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

from llama_full_gate import select
from llama_gate_budget import estimate
from llama_prefill_check import compare


class Prefill(unittest.TestCase):
    def test_fast_selection_keeps_reference_and_excludes_long_serial_jobs(self):
        jobs = [dict(label=s) for s in ['p1-reference', 'p1-prefill', 'p2', 'p3-affine4-kv4',
                'prefill-2000-none-kv16', 'prefill-20000-none-kv16', 'prefill-speed-60000-kv4', 'p4-60000-kl-0']]
        chosen = select(jobs, 'p1-*,prefill-2000-*,prefill-speed-*,p2,p3-*', None)
        self.assertIn('p1-reference', chosen)
        self.assertIn('prefill-speed-60000-kv4', chosen)
        self.assertNotIn('prefill-20000-none-kv16', chosen)
        self.assertNotIn('p4-60000-kl-0', chosen)
        with self.assertRaises(ValueError):
            select(jobs, 'p1-prefill', 'p1-reference')

    def test_batched_planning_scales_linear_projections_and_quadratic_attention(self):
        short = estimate(dict(label='p4-60000-kl-0'))
        long = estimate(dict(label='p4-100000-kl-0'))
        self.assertAlmostEqual(short, 75 + 180 + 2294 / 45)
        self.assertAlmostEqual(long, 75 + 2 * (50 + 100000 ** 2 / 60000000) + 2294 / 45)
        self.assertGreater(long, short)
        self.assertLess(long, 1237)

    def test_receipt_detects_bad_logits_taps_and_cache_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'serial', Path(temp) / 'batch'
            left.mkdir()
            right.mkdir()
            for d in [left, right]:
                (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
                (d / 'cache-0-0.bin').write_bytes(struct.pack('<3H', 0x3f80, 0x4000, 0x4040))
                (d / 'taps.bf16').write_bytes(struct.pack('<2H', 0x3f80, 0x4000))
            self.assertEqual(compare(left, right, '16')['state'], 'PASS')
            (right / 'cache-0-0.bin').write_bytes(struct.pack('<3H', 0x4040, 0x4000, 0x3f80))
            result = compare(left, right, '16')
            self.assertEqual(result['state'], 'FAIL')
            self.assertGreater(result['caches'][0]['changed_bytes'], 0)
            (right / 'cache-0-0.bin').write_bytes((left / 'cache-0-0.bin').read_bytes())
            (right / 'taps.bf16').write_bytes(bytes(4))
            self.assertEqual(compare(left, right, '16')['state'], 'FAIL')
            (right / 'logits.f32').write_bytes(struct.pack('<3f', 3., 2., 1.))
            self.assertFalse(compare(left, right, '16')['top1'])

    def test_p1_prompt_comparison_uses_final_reference_row_without_device_work(self):
        from llama_prefill_check import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = root / 'refs'
            refs.mkdir()
            (refs / 'sequences.json').write_text(json.dumps([[1, 2]]))
            (refs / 'ref-52-s0-logits.f32').write_bytes(struct.pack('<6f', 8, 4, 2, 1, 2, 3))
            def recorded(command, **kwargs):
                at = command.index('llama-prefill')
                folder = Path(command[at + 3])
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'logits.f32').write_bytes(struct.pack('<3f', 1, 2, 3))
            args = SimpleNamespace(out=root / 'out', prompts=None, model=root, runner=root / 'runner',
                                   refs=refs, phase='p1', timeout=10, kv='16', format='affine4', draft='none')
            with patch('llama_prefill_check.subprocess.run', side_effect=recorded):
                run(args)
            result = json.loads((args.out / 'summary.json').read_text())
            self.assertEqual(result['state'], 'PASS')
            self.assertEqual(result['sequences'][0]['kl'], 0)

    def test_kv4_receipt_compares_codes_and_half_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'serial', Path(temp) / 'batch'
            left.mkdir()
            right.mkdir()
            cache = bytearray((bytes([0x88]) * 64 + struct.pack('<4e', *[1.] * 4)) * 8)
            for d in [left, right]:
                (d / 'logits.f32').write_bytes(struct.pack('<2f', 1., 2.))
                (d / 'cache-0-0.bin').write_bytes(cache)
            self.assertEqual(compare(left, right, '4')['state'], 'PASS')
            cache[7 * 72:7 * 72 + 64] = bytes([0xff]) * 64
            (right / 'cache-0-0.bin').write_bytes(cache)
            self.assertEqual(compare(left, right, '4')['state'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
