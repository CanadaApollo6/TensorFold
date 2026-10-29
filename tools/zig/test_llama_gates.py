"""CPU regressions for paid-run correctness, crash recovery and quantitative gate decisions."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llama_draft_full
from llama_draft_stats import distribution, speed
import llama_full_gate as gate
from llama_gate_budget import estimate, calibrate
from llama_gate_dry import exercise
from llama_gate_state import atomic, identity, fingerprint, resume
import llama_gpu_run
import llama_kv_full
import llama_kv_prepare


def distributions():
    return [dict(prompt=p, row=r, variant=v, draws=8192, counts=[[1024] * 8, [1024] * 8])
            for p in range(3) for r in [1, 4, 7] for v in range(4)]


class Gates(unittest.TestCase):
    def invoke(self, module, args):
        with patch.object(sys, 'argv', [module.__name__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_whole_orchestration_resumes_real_crash_and_validates_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = exercise()
        self.assertEqual(report['jobs'], 16)
        self.assertEqual(report['resumed_jobs'], 14)
        self.assertEqual(report['gpu_calls'], 0)

    def test_selectors_close_dependencies_and_reject_typo(self):
        jobs = [dict(label=l) for l in ['p1-reference', 'p1-logits', 'p3-bf16', 'p4-60000-0']]
        self.assertEqual(gate.select(jobs, 'p1-logits', None), {'p1-reference', 'p1-logits'})
        self.assertEqual(gate.select(jobs, 'p3-*,p4-*', 'p3-*'), {'p4-60000-0'})
        with self.assertRaises(ValueError): gate.select(jobs, 'typo', None)

    def test_atomic_resume_refuses_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            artifact = d / 'result'
            artifact.write_text('good')
            job = dict(label='one', command=['cpu'], state='PASS', artifacts={str(artifact): identity(artifact)})
            atomic(d / 'summary.json', dict(fingerprint='key', jobs=[job]))
            self.assertEqual(resume(d / 'summary.json', 'key', [dict(job)])['jobs'][0]['state'], 'PASS')
            artifact.write_text('bad')
            with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
                resume(d / 'summary.json', 'key', [dict(job)])

    def test_all_content_and_args_contribute_to_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d / 'weight').write_bytes(b'first')
            before = identity(d)
            (d / 'weight').write_bytes(b'other')
            self.assertNotEqual(before, identity(d))
            self.assertNotEqual(fingerprint(dict(inputs=before, args=['a'])), fingerprint(dict(inputs=before, args=['b'])))

    def test_budget_stops_before_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            args = SimpleNamespace(out=d, work=d, budget_minutes=.01, only=None, skip=None, plan_only=False,
                                   sm='89', timeout=10)
            with self.assertRaisesRegex(RuntimeError, 'budget would be exceeded'), contextlib.redirect_stdout(io.StringIO()):
                gate.execute(args, [dict(label='p4-100000-0', command=[], state='NOT RUN')], 'key', {},
                             worker=lambda *a: self.fail('started over-budget job'))
            self.assertEqual(json.loads((d / 'summary.json').read_text())['state'], 'BUDGET')

    def test_wall_deadline_kills_child(self):
        with tempfile.TemporaryFile(mode='w+') as log, self.assertRaises(subprocess.TimeoutExpired):
            gate.bounded([sys.executable, '-c', 'import time; time.sleep(10)'], {}, log, .05)

    def test_projection_scales_attention_and_rejects_missing_calibration(self):
        self.assertGreater(estimate(dict(label='p4-100000-0')), estimate(dict(label='p4-60000-0')))
        with self.assertRaises(ValueError): calibrate({})

    def test_sampling_accepts_equivalent_histograms_and_rejects_bias(self):
        rows = distributions()
        self.assertEqual(distribution(rows)['alpha'], .01)
        rows[0]['counts'][1] = [8192] + [0] * 7
        with self.assertRaisesRegex(ValueError, 'equivalence'): distribution(rows)

    def test_sampling_refuses_missing_cells_or_small_sample(self):
        with self.assertRaises(ValueError): distribution(distributions()[:-1])
        rows = distributions()
        rows[0]['draws'] = 32
        with self.assertRaises(ValueError): distribution(rows)

    def test_speed_retains_spread_and_interval(self):
        result = speed([1.1, 1.2, 1.3, 1.4, 1.5])
        self.assertEqual(result['median'], 1.3)
        self.assertEqual(result['spread'], [1.1, 1.5])
        self.assertLessEqual(result['interval_95'][0], result['median'])
        with self.assertRaises(ValueError): speed([1])

    def test_kv_prefills_once_and_checks_every_sweep_length(self):
        for sweep in [False, True]:
            with self.subTest(sweep=sweep), tempfile.TemporaryDirectory() as temp:
                d = Path(temp)
                length = 100000 if sweep else 513
                prompts = d / 'prompts.json'
                prompts.write_text(json.dumps([[166100] + [i] * (length - 1) for i in [1, 2, 3]]))
                launches, comparisons = [], []
                lengths = [100000, 60000, 20000] if sweep else [length]
                def run(cmd, **kwargs):
                    cmd = list(map(str, cmd))
                    if 'llama-capacity' in cmd:
                        return SimpleNamespace(stdout=f'NEED mode=pair context={length + 128} bytes=1 need_mib=17000', stderr='', returncode=0)
                    if 'llama-kv-pair' in cmd or 'llama-kv-sweep' in cmd:
                        launches.append(cmd)
                        work = d / 'work'
                        records = [dict(mode=m, rep=None, prefill_seconds=1, prefill_tok_s=length) for m in ['bf16', 'q4']]
                        for n in lengths:
                            for rep in range(5):
                                for mode in (['bf16', 'q4'] if rep % 2 == 0 else ['q4', 'bf16']):
                                    base = work / str(n) if sweep else work
                                    folder = base / mode / str(rep)
                                    folder.mkdir(parents=True)
                                    (folder / 'greedy-0.json').write_text(json.dumps([7] * 128))
                                    records.append(dict(mode=mode, rep=rep, history_tokens=n, decode_tok_s=20))
                        (work / 'timings.json').write_text(json.dumps(records))
                    elif 'llama_kv_compare.py' in cmd[1]:
                        comparisons.append(cmd)
                        Path(cmd[-1]).write_text('{"passes_distribution_bar":true}')
                    else:
                        self.fail(cmd)
                    return SimpleNamespace(stdout='', stderr='', returncode=0)
                with patch.object(llama_kv_full.subprocess, 'run', side_effect=run):
                    self.invoke(llama_kv_full, ['--model', d, '--prompts', prompts, '--work', d / 'work',
                                              '--receipts', d / 'out', *(['--sweep'] if sweep else [])])
                self.assertEqual(len(launches), 1)
                self.assertEqual(len(comparisons), len(lengths))
                self.assertIn('llama_gpu_run.py', launches[0][1])
                result = json.loads((d / 'out/summary.json').read_text())
                self.assertEqual(result['prefill_passes'], 2)
                self.assertEqual(result['decode_forwards'], 1270 * len(lengths))

    def test_p3_rejects_no_acceptance_and_no_speedup(self):
        for accepted, ratio in [(0, 2), (8, 1), (8, .9)]:
            with self.subTest(accepted=accepted, ratio=ratio), tempfile.TemporaryDirectory() as temp:
                d = Path(temp)
                (d / 'prompts.json').write_text('[[1],[2],[3]]')
                def run(cmd, **kwargs):
                    if 'llama-draft-full' in list(map(str, cmd)):
                        rows = []
                        for p in range(3):
                            for rep in range(5):
                                plain = dict(tokens=[1] * 256, seconds=ratio, rounds=255, drafted=0, accepted=0, stopped=False)
                                drafted = dict(plain, seconds=1, rounds=100, drafted=700, accepted=accepted)
                                rows.append(dict(prompt=p, seed=None, variant=0, rep=rep, plain=plain, drafted=drafted))
                        (d / 'out/pairs.json').write_text(json.dumps(rows))
                        (d / 'out/pairs.json.distribution.jsonl').write_text('\n'.join(map(json.dumps, distributions())))
                        kwargs['stdout'].write('FORCED full_accept_then_reject\n' * 3)
                    return SimpleNamespace(returncode=0, stdout='NEED draft=bf16 kv=bf16 bytes=1 need_mib=17000', stderr='')
                with patch.object(llama_draft_full.subprocess, 'run', side_effect=run), patch.object(
                        llama_draft_full.subprocess, 'check_output', return_value='49152'):
                    with self.assertRaisesRegex(RuntimeError, 'mean acceptance'):
                        self.invoke(llama_draft_full, ['--model', d, '--draft', d, '--prompts', d / 'prompts.json',
                                                      '--out', d / 'out', '--format', 'bf16', '--kv', '16'])

    def test_prepare_refuses_identical_histories(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d / 'expected.jsonl').write_text('\n'.join(json.dumps(dict(name=str(i), prompt=[166100] + [1] * 2100)) for i in range(55)))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                self.invoke(llama_kv_prepare, ['--review', d, '--out', d / 'out', '--length', 2000])

    def test_guard_stops_only_its_child_when_reserve_is_breached(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            class Child:
                pid = 123
                returncode = None
                terminated = False
                def poll(self): return self.returncode
                def terminate(self): self.terminated, self.returncode = True, -15
                def wait(self, **kwargs): return self.returncode
                def kill(self): raise AssertionError('unexpected kill')
            child = Child()
            with patch.object(llama_gpu_run.subprocess, 'Popen', return_value=child), patch.object(
                    llama_gpu_run, 'smi', side_effect=['4000', '999, 2000\n123, 900', '1500']):
                with self.assertRaises(SystemExit):
                    self.invoke(llama_gpu_run, ['--need-mib', '1000', '--receipt', d / 'vram.json', '--', 'child'])
            self.assertTrue(child.terminated)
            self.assertTrue(json.loads((d / 'vram.json').read_text())['reserve_breached'])


if __name__ == '__main__':
    unittest.main()
