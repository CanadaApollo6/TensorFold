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
from llama_prefill_check import cache_bar, compare, compare_host, compare_numpy, prefill_profile, PREFILL_KL_BAR


def capacity(command):
    """Allocator NEED lines as llama-capacity / llama-draft-capacity print them, all at 14000 MiB."""
    context = command[-1]
    lines = [f'NEED mode={m} context={context} bytes=1 need_mib=14000' for m in ['bf16', 'q4']]
    lines += [f'NEED draft={f} kv={m} context={context} bytes=1 need_mib=14000'
              for f in ['none', 'bf16', 'affine4'] for m in ['bf16', 'q4']]
    return SimpleNamespace(stdout='\n'.join(lines), stderr='')


class Prefill(unittest.TestCase):
    def test_prefill_kl_bar_keeps_top1_and_cache_requirements(self):
        self.assertEqual(PREFILL_KL_BAR, .01)
        comparators = [compare_host]
        try:
            import numpy
            comparators.append(compare_numpy)
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'serial', Path(temp) / 'batch'
            left.mkdir()
            right.mkdir()
            (left / 'logits.f32').write_bytes(struct.pack('<2f', 0, 1))
            for folder in (left, right):
                (folder / 'cache-0-0.bin').write_bytes(struct.pack('<128H', *[0x3f80] * 128))
            for logits, state in [((.2, 1), 'PASS'), ((.4, 1), 'FAIL'), ((1, 0), 'FAIL')]:
                (right / 'logits.f32').write_bytes(struct.pack('<2f', *logits))
                for comparator in comparators:
                    with self.subTest(comparator=comparator.__name__, logits=logits):
                        result = comparator(left, right, '16', cross_path=True)
                        self.assertEqual(result['state'], state)
                        self.assertEqual(result['kl_bar'], PREFILL_KL_BAR)
                        if state == 'PASS':
                            self.assertGreater(result['kl'], .001)
                            self.assertLess(result['kl'], PREFILL_KL_BAR)
            (right / 'logits.f32').write_bytes((left / 'logits.f32').read_bytes())
            (right / 'cache-0-0.bin').write_bytes(struct.pack('<128H', *[0x4000] * 128))
            for comparator in comparators:
                self.assertEqual(comparator(left, right, '16', cross_path=True)['state'], 'FAIL')

    def test_drafter_kv_corruption_fails_with_unchanged_source_taps(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'cold', Path(temp) / 'resume'
            left.mkdir()
            right.mkdir()
            for d in [left, right]:
                (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
                (d / 'taps.bf16').write_bytes(struct.pack('<2H', 0x3f80, 0x4000))
                (d / 'draft-kv.bf16').write_bytes(struct.pack('<2H', 0x3f80, 0x4000))
                (d / 'draft-context.json').write_text(json.dumps(dict(begin=17953, end=20000)))
            comparators = [compare, compare_host]
            try:
                import numpy
                comparators.append(compare_numpy)
            except ImportError:
                pass
            for comparator in comparators:
                self.assertEqual(comparator(left, right, '4')['state'], 'PASS')
            (right / 'draft-kv.bf16').write_bytes(struct.pack('<2H', 0x4000, 0x3f80))
            for comparator in comparators:
                self.assertEqual(comparator(left, right, '4')['state'], 'FAIL')
            (right / 'draft-kv.bf16').unlink()
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                compare(left, right, '4')
            (right / 'draft-kv.bf16').write_bytes((left / 'draft-kv.bf16').read_bytes())
            (right / 'draft-context.json').write_text(json.dumps(dict(begin=17954, end=20000)))
            with self.assertRaisesRegex(ValueError, 'drafter context mismatch'):
                compare(left, right, '4')

    def test_equivalence_runs_every_drafter_lifecycle_against_cold_kv(self):
        from llama_prefill_check import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompts = root / 'prompts.json'
            prompts.write_text(json.dumps([[1, 2, 3]]))
            kinds = []
            corrupt = False
            def recorded(command, **kwargs):
                if 'capacity' in command[1]:
                    return capacity(command)
                self.assertEqual(command[command.index('--need-mib') + 1], '15024')
                at = command.index('llama-prefill')
                folder = Path(command[at + 3])
                kinds.append(command[at + 5])
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'logits.f32').write_bytes(struct.pack('<3f', 1, 2, 3))
                for i in range(104):
                    (folder / f'cache-{i // 2}-{i % 2}.bin').write_bytes(struct.pack('<H', 0x3f80))
                (folder / 'taps.bf16').write_bytes(struct.pack('<1920H', *[0x3f80] * 1920))
                (folder / 'draft-kv.bf16').write_bytes(struct.pack('<H', 0x3f80))
                if corrupt and kinds[-1] == 'resume':
                    (folder / 'draft-kv.bf16').write_bytes(struct.pack('<H', 0x4000))
                (folder / 'draft-context.json').write_text(json.dumps(dict(begin=0, end=3)))
            cases = 'resume,shorten,resume-shorten,cancel-shorten,cancel-extend,rewrite-cancel'
            args = SimpleNamespace(out=root / 'out', prompts=prompts, model=root, runner=root / 'runner',
                                   phase='equivalence', timeout=10, kv='16', format='affine4', draft='draft',
                                   resume_cases=cases)
            with patch('llama_prefill_check.subprocess.run', side_effect=recorded):
                run(args)
            result = json.loads((args.out / 'summary.json').read_text())
            self.assertEqual(result['state'], 'PASS')
            self.assertEqual(kinds, ['serial', 'batch', *cases.split(',')])
            self.assertEqual(set(result['resumed']), set(cases.split(',')))
            corrupt = True
            with patch('llama_prefill_check.subprocess.run', side_effect=recorded):
                with self.assertRaisesRegex(RuntimeError, 'prefill comparison failed'):
                    run(args)
            result = json.loads((args.out / 'summary.json').read_text())
            self.assertEqual(result['resumed']['resume']['state'], 'FAIL')
            (args.out / 'serial/draft-kv.bf16').unlink()
            with patch('llama_prefill_check.subprocess.run', side_effect=lambda c, **k: capacity(c) if 'capacity' in c[1] else None), \
                    self.assertRaisesRegex(ValueError, 'missing drafter cache'):
                run(args)

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

    def test_profile_jobs_run_only_when_named_and_have_separate_budget(self):
        jobs = [dict(label=s) for s in ['prefill-speed-20000-kv4', 'profile-prefill-20000-kv4', 'draft-profile']]
        self.assertEqual(select(jobs, None, None), {'prefill-speed-20000-kv4', 'draft-profile'})
        self.assertEqual(select(jobs, 'profile-prefill-*', None), {'profile-prefill-20000-kv4'})
        self.assertGreater(estimate(dict(label='profile-prefill-20000-kv4')), estimate(dict(label='prefill-speed-20000-kv4')))

    def test_prefill_profile_averages_warm_runs_and_splits_host_time(self):
        phases = ['embed', 'norm', 'q', 'k', 'v', 'rope', 'attention', 'o', 'gate', 'up', 'swiglu', 'down', 'residual', 'head']
        lines = ['info: LLAMA_PREFILL_CHUNK run=1 position=0 rows=128 span_ms=9.000 gemm_ms=1 attention_ms=1 other_ms=1 idle_ms=0']
        for run in range(1, 5):
            scale = 10 if run == 1 else run - 1  # the cold run must not leak into the mean
            lines.append(f'info: LLAMA_PREFILL_RUN run={run} tokens=2000 chunks=16 span_ms={100 * scale:.3f} busy_ms={90 * scale:.3f} idle_ms={10 * scale:.3f}')
            lines += [f'info: LLAMA_PREFILL_PHASE run={run} phase={name} calls=52 ms={scale * (10 if name in ["gate", "attention"] else 1):.3f} us_per_token=0'
                      for name in phases]
        profile = prefill_profile('\n'.join(lines), 0.25)
        self.assertEqual(profile['runs'], [2, 3, 4])
        self.assertEqual(profile['chunks'], 16)
        self.assertAlmostEqual(profile['span_ms'], 200)
        self.assertAlmostEqual(profile['host_ms'], 50)
        self.assertAlmostEqual(profile['gemm_ms'], 2 * (7 + 10))
        self.assertAlmostEqual(profile['attention_ms'], 20)
        self.assertAlmostEqual(profile['other_ms'], 2 * 5)
        with self.assertRaisesRegex(ValueError, 'no warm'):
            prefill_profile(lines[1], 1)

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

    def test_cross_path_cache_drift_passes_only_serial_versus_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'serial', Path(temp) / 'batch'
            left.mkdir()
            right.mkdir()
            for d in [left, right]:
                (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
            (left / 'cache-0-0.bin').write_bytes(struct.pack('<4H', 0x3f80, 0x3f80, 0x3f80, 0x3f80))
            (right / 'cache-0-0.bin').write_bytes(struct.pack('<4H', 0x3f80, 0x3f80, 0x3f80, 0x3f88))
            drift = compare(left, right, '16', cross_path=True)
            self.assertTrue(.01 < drift['caches'][0]['relative_l2'] < .05)
            self.assertEqual(drift['state'], 'PASS')
            self.assertEqual(compare(left, right, '16')['state'], 'FAIL')
            (right / 'cache-0-0.bin').write_bytes(struct.pack('<4H', 0x3f80, 0x4000, 0x3f80, 0x3f80))
            self.assertEqual(compare(left, right, '16', cross_path=True)['state'], 'FAIL')

    def test_localized_cache_corruption_fails_cross_path_in_shallow_layers(self):
        positions, heads = 256, 8
        seed = [12345]
        def rand():
            seed[0] = (seed[0] * 1103515245 + 12345) % 2 ** 31
            return seed[0]
        bf16 = [0x3c00 + rand() % 0x700 | (rand() % 2) << 15 for _ in range(positions * heads * 128)]
        kv4 = bytearray()
        for _ in range(positions * heads):
            kv4 += bytes(rand() % 256 for _ in range(64)) + struct.pack('<4e', *[1.] * 4)
        drifted16 = [v ^ 1 if i % 3 == 0 else v for i, v in enumerate(bf16)]
        drifted4 = bytearray(kv4)
        for row in range(0, positions * heads, 8):
            drifted4[row * 72] ^= 1
        comparators = [compare_host]
        try:
            import numpy
            comparators.append(compare_numpy)
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as temp:
            # Zero just enough head rows to stay under the whole-plane bar: only the row bound can catch them.
            for kv, clean, drift, zeroed, row_bytes in [
                    ('16', struct.pack(f'<{len(bf16)}H', *bf16), struct.pack(f'<{len(bf16)}H', *drifted16), 4, 256),
                    ('4', bytes(kv4), bytes(drifted4), 16, 72)]:
                left, right = Path(temp) / f'serial{kv}', Path(temp) / f'batch{kv}'
                left.mkdir()
                right.mkdir()
                for d in [left, right]:
                    (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
                for name in ['cache-3-0.bin', 'cache-40-0.bin']:
                    (left / name).write_bytes(clean)
                    (right / name).write_bytes(drift)
                for comparator in comparators:
                    self.assertEqual(comparator(left, right, kv, cross_path=True)['state'], 'PASS')
                corrupt = bytearray(drift)
                corrupt[640 * row_bytes:(640 + zeroed) * row_bytes] = bytes(zeroed * row_bytes)
                (right / 'cache-40-0.bin').write_bytes(corrupt)
                for comparator in comparators:
                    self.assertEqual(comparator(left, right, kv, cross_path=True)['state'], 'PASS', 'deep rows drift past any row bar')
                    self.assertEqual(comparator(left, right, kv)['state'], 'FAIL', 'resume shares batched kernels at every depth')
                (right / 'cache-3-0.bin').write_bytes(corrupt)
                for comparator in comparators:
                    result = comparator(left, right, kv, cross_path=True)
                    shallow = next(r for r in result['caches'] if r['file'] == 'cache-3-0.bin')
                    self.assertLessEqual(shallow['relative_l2'], cache_bar(kv, 'cache-3-0.bin', True))
                    self.assertGreater(shallow['worst_row'], .9)
                    self.assertEqual(result['state'], 'FAIL')

    def test_low_norm_shallow_row_and_drafter_writes_fail_when_lost(self):
        from array import array
        seed = [7]
        def rand():
            seed[0] = (seed[0] * 1103515245 + 12345) % 2 ** 31
            return seed[0]
        pattern = [0x3c00 + rand() % 0x700 | (rand() % 2) << 15 for _ in range(1024)]
        def plane(rows, width):
            return array('H', [v ^ (r & 0x3f) for r in range(rows) for v in pattern[:width]])
        runs = [(64, [compare_host])]
        try:
            import numpy
            runs = [(64, [compare_host, compare_numpy]), (2047, [compare_numpy])]
        except ImportError:
            pass
        hidden = 256
        for positions, comparators in runs:
            with tempfile.TemporaryDirectory() as temp:
                left, right = Path(temp) / 'cold', Path(temp) / 'resume'
                left.mkdir()
                right.mkdir()
                cache = plane(256 * 8, 128)
                for i in range(5 * 128, 6 * 128):
                    cache[i] = (cache[i] & 0x8000) | (cache[i] & 0x7fff) - 0x200
                taps = plane(positions * 5, hidden)
                kv = plane(10 * positions, 1024)
                clean = {'cache-1-1.bin': cache, 'taps.bf16': taps, 'draft-kv.bf16': kv}
                for d in [left, right]:
                    (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
                    for name, values in clean.items():
                        (d / name).write_bytes(values.tobytes())
                    (d / 'draft-context.json').write_text(json.dumps(dict(begin=20000 - positions, end=20000)))
                def judged(name, values, cross_path, expected, why):
                    (right / name).write_bytes(values.tobytes())
                    for comparator in comparators:
                        state = comparator(left, right, '16', cross_path=cross_path)['state']
                        self.assertEqual(state, expected, f'{why} ({comparator.__name__}, {positions} positions)')
                    (right / name).write_bytes(clean[name].tobytes())
                for cross_path in [False, True]:
                    judged('cache-1-1.bin', cache, cross_path, 'PASS', 'unchanged')
                lost = array('H', cache)
                lost[5 * 128:6 * 128] = array('H', [0] * 128)
                judged('cache-1-1.bin', lost, True, 'FAIL', 'a zeroed low-norm shallow row')
                lost = array('H', kv)
                at = ((2 * 3 + 1) * positions + positions // 2) * 1024
                lost[at:at + 1024] = array('H', [0] * 1024)
                judged('draft-kv.bf16', lost, False, 'FAIL', 'one zeroed drafter V write after resume')
                at = positions // 2 * 5 * hidden
                lost = array('H', taps)
                lost[at:at + 5 * hidden] = array('H', [0] * 5 * hidden)
                judged('taps.bf16', lost, True, 'FAIL', 'a skipped tap row across all planes')
                lost = array('H', taps)
                lost[at + 2 * hidden:at + 5 * hidden] = array('H', [0] * 3 * hidden)
                (right / 'taps.bf16').write_bytes(lost.tobytes())
                for comparator in comparators:
                    record = next(r for r in comparator(left, right, '16', cross_path=True)['caches'] if r['file'] == 'taps.bf16')
                    self.assertEqual(record['worst_row'], 0, 'deep tap planes drift past any cross-path row bar')
                judged('taps.bf16', lost, False, 'FAIL', 'deep tap planes are exact after resume')

    def test_kv4_bos_copy_rows_partial_rows_and_empty_dumps(self):
        from array import array
        comparators = [compare_host]
        try:
            import numpy
            comparators.append(compare_numpy)
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as temp:
            left, right = Path(temp) / 'serial', Path(temp) / 'batch'
            left.mkdir()
            right.mkdir()
            # BOS rows grow with depth; layers past 9 are 16x larger, so a lost shallow copy barely moves the aggregate.
            bos = array('H', [0x3f80 + (i * 37) % 0x200 + (0x200 if i >= 10 * 2048 else 0) for i in range(52 * 2 * 1024)])
            for d in [left, right]:
                (d / 'logits.f32').write_bytes(struct.pack('<3f', 1., 2., 3.))
                (d / 'bos.bf16').write_bytes(bos.tobytes())
            for comparator in comparators:
                self.assertEqual(comparator(left, right, '4', cross_path=True)['state'], 'PASS')
            lost = array('H', bos)
            lost[(3 * 2 + 1) * 1024:(3 * 2 + 2) * 1024] = array('H', [0] * 1024)
            (right / 'bos.bf16').write_bytes(lost.tobytes())
            for cross_path in [True, False]:
                for comparator in comparators:
                    result = comparator(left, right, '4', cross_path=cross_path)
                    self.assertLess(result['caches'][0]['relative_l2'], .01)
                    self.assertEqual(result['state'], 'FAIL', 'a lost BOS V copy')
            (left / 'bos.bf16').write_bytes(struct.pack('<129H', *[0x3f80] * 129))
            (right / 'bos.bf16').write_bytes(struct.pack('<129H', *[0x3f80] * 128, 0x3fc0))
            verdicts = {c(left, right, '4')['caches'][0]['worst_row'] for c in comparators}
            self.assertEqual(verdicts, {.5})
            (right / 'bos.bf16').write_bytes(b'')
            (left / 'bos.bf16').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'empty cache evidence'):
                compare(left, right, '4')

    def test_p1_prompt_comparison_uses_final_reference_row_without_device_work(self):
        from llama_prefill_check import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = root / 'refs'
            refs.mkdir()
            (refs / 'sequences.json').write_text(json.dumps([[1, 2]]))
            (refs / 'ref-52-s0-logits.f32').write_bytes(struct.pack('<6f', 8, 4, 2, 1, 2, 3))
            def recorded(command, **kwargs):
                if 'capacity' in command[1]:
                    return capacity(command)
                self.assertEqual(command[command.index('--need-mib') + 1], '15024')
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
