"""CPU regressions for paid-run correctness, crash recovery and quantitative gate decisions."""
import contextlib
import argparse
import runpy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llama_draft_full
from llama_draft_stats import speed
from llama_sample_oracle import evaluate, probabilities, DRAWS
from llama_sample_gate import fixtures, check
import llama_full_gate as gate
from llama_gate_budget import estimate, calibrate, job_class
from llama_gate_dry import exercise
from llama_gate_state import atomic, identity, fingerprint, resume
import llama_gpu_run
import llama_kv_full
import llama_kv_prepare
import llama_kv_compare
import llama_kv_parity as parity



class Gates(unittest.TestCase):
    def invoke(self, module, args):
        with patch.object(sys, 'argv', [module.__name__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_whole_orchestration_resumes_real_crash_and_validates_inputs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = exercise()
        self.assertEqual(report['jobs'], 66)
        self.assertEqual(report['selected_jobs'], 60)
        self.assertEqual(report['resumed_jobs'], 58)
        chosen = gate.select(report['plan'], 'exl3-r32-unit,profile-prefill-*', None)
        self.assertEqual(len(chosen), 6)
        self.assertIn('exl3-r32-unit', chosen)
        self.assertEqual(report['plan'][1]['label'], 'exl3-r32-unit')
        self.assertEqual(report['gpu_calls'], 0)
        self.assertTrue(report['changed_prefill_mode_refused'])
        class Parsed(Exception): pass
        original = argparse.ArgumentParser.parse_args
        def parse(parser, *a, **kw):
            original(parser, *a, **kw)
            raise Parsed()
        for job in report['plan']:
            command = job['command']
            if job['label'].startswith('p4ref-exl3-'):
                self.assertTrue(command[1].endswith('llama_gpu_run.py'))
                command = command[command.index('--') + 1:]
            with patch.object(sys, 'argv', command[1:]), patch.object(argparse.ArgumentParser, 'parse_args', parse):
                with self.assertRaises(Parsed): runpy.run_path(command[1], run_name='__main__')
            if job['label'].endswith('-wire'):
                self.assertTrue(job['command'][-1].endswith('wire-runner'))
            if job['label'].startswith('p3-') and not job['label'].endswith('-wire'):
                self.assertIn('--replay-runner', job['command'])
            if job['label'].startswith('prefill-') and not job['label'].startswith('prefill-speed-'):
                at = command.index('--resume-cases')
                expected = 'resume' if '-none-' in job['label'] else 'resume,shorten,resume-shorten,cancel-shorten,cancel-extend,rewrite-cancel'
                self.assertEqual(command[at + 1], expected)
        refs = [j for j in report['plan'] if j['label'].startswith('p4ref-exl3-')]
        self.assertEqual([j['label'] for j in refs], [f'p4ref-exl3-{n}-{i}' for n in [20000, 60000, 100000] for i in range(3)])
        for job in refs:
            cmd = job['command']
            length = int(job['label'].split('-')[2])
            self.assertEqual(Path(cmd[cmd.index('--prompts') + 1]).name, f'histories-{length}.json')
            self.assertEqual(cmd[cmd.index('--sequence') + 1], job['label'][-1])
            self.assertEqual(cmd[cmd.index('--') + 1], sys.executable)
            self.assertGreater(int(cmd[cmd.index('--need-mib') + 1]), 16384)
        labels = [j['label'] for j in report['plan']]
        self.assertGreater(labels.index(refs[0]['label']), labels.index('p3-affine4-kv4-wire'))
        self.assertEqual(labels[-1], 'p4-parity')
        self.assertEqual(len(gate.select(report['plan'], 'p4-*,p4ref-exl3-*', None)), 19)

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

    def test_keep_going_runs_later_jobs_and_ends_fail(self):
        for keep in [False, True]:
            with self.subTest(keep=keep), tempfile.TemporaryDirectory() as temp:
                d = Path(temp)
                args = SimpleNamespace(out=d, work=d, budget_minutes=60, only=None, skip=None, plan_only=False,
                                       sm='89', timeout=10, keep_going=keep)
                jobs = [dict(label=l, command=[], state='NOT RUN') for l in ['prefill-2000-bf16-kv16', 'p2']]
                ran = []
                def worker(job, log, limit):
                    ran.append(job['label'])
                    return 1 if job['label'] == 'prefill-2000-bf16-kv16' else 0
                with contextlib.redirect_stdout(io.StringIO()):
                    if keep:
                        self.assertEqual(gate.execute(args, jobs, 'key', {}, worker=worker)['state'], 'FAIL')
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'failed'):
                            gate.execute(args, jobs, 'key', {}, worker=worker)
                self.assertEqual(ran, ['prefill-2000-bf16-kv16', 'p2'] if keep else ['prefill-2000-bf16-kv16'])
                states = [j['state'] for j in json.loads((d / 'summary.json').read_text())['jobs']]
                self.assertEqual(states, ['FAIL', 'PASS'] if keep else ['FAIL', 'NOT RUN'])

    def test_keep_going_stops_on_reference_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            args = SimpleNamespace(out=d, work=d, budget_minutes=60, only=None, skip=None, plan_only=False,
                                   sm='89', timeout=10, keep_going=True)
            jobs = [dict(label=l, command=[], state='NOT RUN') for l in ['p1-reference', 'p1-logits']]
            with self.assertRaisesRegex(RuntimeError, 'p1-reference failed'), contextlib.redirect_stdout(io.StringIO()):
                gate.execute(args, jobs, 'key', {}, worker=lambda job, log, limit: 1)
            self.assertEqual([j['state'] for j in jobs], ['FAIL', 'NOT RUN'])

    def test_budget_stop_after_keep_going_failure_reports_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            args = SimpleNamespace(out=d, work=d, budget_minutes=60, only=None, skip=None, plan_only=False,
                                   sm='89', timeout=10, keep_going=True)
            jobs = [dict(label=l, command=[], state='NOT RUN') for l in ['prefill-2000-bf16-kv16', 'p2']]
            cost = lambda job, rates=None: 1e9 if job['label'] == 'p2' else 0
            with patch.object(gate, 'estimate', side_effect=cost), patch.object(gate, 'calibrate', return_value=0), \
                    self.assertRaisesRegex(RuntimeError, 'budget'), contextlib.redirect_stdout(io.StringIO()):
                gate.execute(args, jobs, 'key', {}, worker=lambda job, log, limit: 1)
            summary = json.loads((d / 'summary.json').read_text())
            self.assertEqual((summary['state'], summary['budget_exhausted_before']), ('FAIL', 'p2'))

    def test_bounded_kills_descendants_that_outlive_the_wrapper(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryFile(mode='w+') as log:
            marker = Path(temp) / 'pid'
            stubborn = ('import os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                        'open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(60)')
            wrapper = (f'import subprocess, sys, time, pathlib; subprocess.Popen([sys.executable, "-c", {stubborn!r}, {str(marker)!r}]); '
                       f'p = pathlib.Path({str(marker)!r})\nwhile not p.exists() or not p.read_text(): time.sleep(.05)')
            self.assertEqual(gate.bounded([sys.executable, '-c', wrapper], {}, log, 30, grace=.5), 0)
            stat = Path(f'/proc/{marker.read_text()}/stat')
            self.assertTrue(not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z')

    def test_cleanup_failure_stops_keep_going_on_every_path(self):
        with tempfile.TemporaryFile(mode='w+') as log, patch.object(gate, 'stop_group', return_value=False):
            for command, limit in [(['true'], 10), (['sleep', '10'], .05)]:
                with self.subTest(command=command), self.assertRaises(gate.CleanupFailed):
                    gate.bounded(command, {}, log, limit)
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            args = SimpleNamespace(out=d, work=d, budget_minutes=60, only=None, skip=None, plan_only=False,
                                   sm='89', timeout=10, keep_going=True)
            jobs = [dict(label=l, command=[], state='NOT RUN') for l in ['prefill-2000-bf16-kv16', 'p2']]
            def worker(job, log, limit): raise gate.CleanupFailed('survived')
            with self.assertRaises(gate.CleanupFailed), contextlib.redirect_stdout(io.StringIO()):
                gate.execute(args, jobs, 'key', {}, worker=worker)
            self.assertEqual([j['state'] for j in jobs], ['FAIL', 'NOT RUN'])

    def test_interrupt_during_cleanup_waits_for_cleanup(self):
        import os, signal
        finished = []
        def stop(process, grace):
            os.kill(os.getpid(), signal.SIGINT)
            finished.append(gate.group_alive(process.pid) or process.wait() is None)
            return True
        with tempfile.TemporaryFile(mode='w+') as log, patch.object(gate, 'stop_group', side_effect=stop):
            with self.assertRaises(KeyboardInterrupt):
                gate.bounded(['true'], {}, log, 10)
        self.assertEqual(finished, [False])

    def test_signals_during_wait_or_cleanup_stop_the_whole_group(self):
        import os, signal, threading
        class Deadline(Exception): pass
        def deadline(signum, frame): raise Deadline()
        saved, real = signal.signal(signal.SIGALRM, deadline), gate.stop_group
        def signalled(sig):
            def stop(process, grace):
                os.kill(os.getpid(), sig)
                return real(process, grace)
            return stop
        try:
            for sig in [signal.SIGINT, signal.SIGALRM]:
                for phase in ['wait', 'cleanup']:
                    with self.subTest(sig=sig, phase=phase), tempfile.TemporaryDirectory() as temp, \
                            tempfile.TemporaryFile(mode='w+') as log:
                        marker = Path(temp) / 'pid'
                        script = ('import os, subprocess, sys, time; subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
                                  f'open({str(marker)!r}, "w").write(str(os.getpid()))' + ('; time.sleep(60)' if phase == 'wait' else ''))
                        timer = threading.Timer(.5, os.kill, [os.getpid(), sig])
                        if phase == 'wait': timer.start()
                        started = time.monotonic()
                        with patch.object(gate, 'stop_group', side_effect=signalled(sig) if phase == 'cleanup' else real), \
                                self.assertRaises(KeyboardInterrupt if sig == signal.SIGINT else Deadline):
                            gate.bounded([sys.executable, '-c', script], {}, log, 30, grace=1)
                        timer.cancel()
                        self.assertLess(time.monotonic() - started, 10)
                        self.assertFalse(gate.group_alive(int(marker.read_text())))
        finally:
            signal.signal(signal.SIGALRM, saved)

    def test_wire_normalizer_matches_pre_normalized_pod_oracle(self):
        import uuid
        from llama_draft_wire import normalize
        from llama_serve_envelopes import normalize as oracle
        tool = f'chatcmpl-tool-{uuid.uuid4().hex[:16]}'
        body = (f'{{"id":"chatcmpl-{uuid.uuid4().hex}","created":1761800000,"choices":[{{"message":'
                f'{{"tool_calls":[{{"id":"{tool}"}}]}}}}]}}')
        self.assertEqual(normalize(body.encode()), oracle(body).encode())
        with self.assertRaises(ValueError): normalize(oracle(body).encode())

    def test_wall_deadline_kills_child(self):
        with tempfile.TemporaryFile(mode='w+') as log, self.assertRaises(subprocess.TimeoutExpired):
            gate.bounded([sys.executable, '-c', 'import time; time.sleep(10)'], {}, log, .05)

    def test_projection_scales_attention_and_rejects_missing_calibration(self):
        self.assertGreater(estimate(dict(label='p4-100000-0')), estimate(dict(label='p4-60000-0')))
        with self.assertRaises(ValueError): calibrate(dict(label='p4-60000-0'), float('nan'))
        self.assertEqual(job_class(dict(label='p4-parity')), 'kv-parity')
        self.assertEqual(estimate(dict(label='p4-parity'), {'kv': 999}), 5)
        short, long = [dict(label=f'p4ref-exl3-{n}-0') for n in [20000, 60000]]
        self.assertEqual(job_class(short), 'kv-reference')
        self.assertEqual((estimate(short), estimate(long)), (600, 1800))
        self.assertEqual(estimate(long, {'kv-reference': calibrate(short, 800), 'kv': 1}), 3000)

    def test_independent_oracle_rejects_token_transfer(self):
        case = fixtures()[0]
        p = probabilities(case)
        counts = [int(v * DRAWS) for v in p]
        counts[0] += DRAWS - sum(counts)
        record = dict(draws=DRAWS, counts=counts, accepted=1, proposed=2)
        self.assertEqual(evaluate(case, record)['state'], 'PASS')
        transfer = min(counts[0], int(.08 * DRAWS))
        counts[0] -= transfer
        counts[8] += transfer
        self.assertEqual(evaluate(case, record)['state'], 'FAIL')

    def test_sampling_refuses_small_draw_count(self):
        with self.assertRaises(ValueError): evaluate(fixtures()[0], dict(draws=1, counts=[1]))

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
                        Path(cmd[-1]).write_text('{"state":"PASS","results":[{"mean_kl":0.9}],"greedy_equal":false}')
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


class P4Parity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        means = [(.0345, .0293), (.0129, .0119), (.9178, .8982),
                 (.0383, .0388), (.0644, .0683), (.0934, .1001),
                 (.0414, .0415), (1.6353, 1.8748), (.1574, .1464)]
        self.paths = []
        for (length, sequence), (tf_mean, ref_mean) in zip(parity.CASES, means):
            tf_label, ref_label = parity.labels(length, sequence)
            tf = self.root / tf_label / f'quality-{length}.json'
            ref = self.root / ref_label / 'summary.json'
            tf.parent.mkdir()
            ref.parent.mkdir()
            metric = dict(rows=512, mean_kl=tf_mean, max_kl=4, greedy_matching=3)
            atomic(tf, dict(state='PASS', results=[metric], greedy_equal=False))
            atomic(ref, dict(state='COMPLETE', history_tokens=length, sequence=sequence,
                             configs={'k4-v4': dict(metric, mean_kl=ref_mean, state='MEASURED')}))
            self.paths.append((tf, ref))

    def scale(self, index, factor):
        tf, ref = self.paths[index]
        data = json.loads(tf.read_text())
        data['results'][0]['mean_kl'] = factor * json.loads(ref.read_text())['configs']['k4-v4']['mean_kl']
        atomic(tf, data)

    def test_todays_numbers_pass_despite_max_kl_and_greedy(self):
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'PASS')
        self.assertEqual(len(report['results']), 9)
        self.assertAlmostEqual(report['pooled']['ratio'], 2.9954 / 3.2093)
        self.assertTrue(all(row['tensorfold']['max_kl'] == 4 and not row['tensorfold']['greedy_equal']
                            for row in report['results']))

    def test_one_sequence_at_1_3_fails_with_pooled_inside_bar(self):
        self.scale(0, 1.3)
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'FAIL')
        self.assertEqual(report['results'][0]['state'], 'FAIL')
        self.assertEqual(report['pooled']['state'], 'PASS')

    def test_pooled_1_06_fails_even_when_all_sequences_pass(self):
        for i in range(9):
            self.scale(i, 1.06)
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'FAIL')
        self.assertTrue(all(row['state'] == 'PASS' for row in report['results']))
        self.assertAlmostEqual(report['pooled']['ratio'], 1.06)
        self.assertEqual(report['pooled']['state'], 'FAIL')

    def test_missing_counterpart_fails(self):
        self.paths[-1][1].unlink()
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'FAIL')
        self.assertIn('FileNotFoundError', report['results'][-1]['error'])
        self.assertIsNone(report['pooled']['ratio'])

    def test_zero_reference_requires_zero_tensorfold_kl(self):
        for tf, ref in self.paths:
            a, b = json.loads(tf.read_text()), json.loads(ref.read_text())
            a['results'][0]['mean_kl'] = b['configs']['k4-v4']['mean_kl'] = 0
            atomic(tf, a)
            atomic(ref, b)
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'PASS')
        self.assertEqual(report['pooled']['ratio'], 1)
        tf = self.paths[0][0]
        a = json.loads(tf.read_text())
        a['results'][0]['mean_kl'] = .001
        atomic(tf, a)
        report = parity.evaluate(self.root)
        self.assertEqual(report['state'], 'FAIL')
        self.assertEqual(report['results'][0]['state'], 'FAIL')
        self.assertIsNone(report['results'][0]['ratio'])

    def test_tf_error_and_mismatched_sequence_fail(self):
        tf, ref = self.paths[0]
        for measurement in [dict(state='FAIL'), dict(state='PASS', sequence=1),
                            dict(state='PASS', sequence=0, history_tokens=100000)]:
            atomic(tf.parent / 'summary.json', measurement)
            self.assertEqual(parity.evaluate(self.root)['state'], 'FAIL')
        (tf.parent / 'summary.json').unlink()
        data = json.loads(ref.read_text())
        data['sequence'] = 1
        atomic(ref, data)
        self.assertEqual(parity.evaluate(self.root)['state'], 'FAIL')

    def test_errored_and_nonfinite_counterparts_fail(self):
        ref = self.paths[0][1]
        original = json.loads(ref.read_text())
        for state, config_state, value in [('ERROR', 'MEASURED', .02), ('COMPLETE', 'SKIPPED', .02),
                                            ('COMPLETE', 'MEASURED', float('nan'))]:
            data = dict(original, state=state)
            data['configs']['k4-v4'].update(state=config_state, mean_kl=value)
            atomic(ref, data)
            self.assertEqual(parity.evaluate(self.root)['state'], 'FAIL')
        tf = self.paths[0][0]
        data = json.loads(tf.read_text())
        data['results'][0]['mean_kl'] = float('inf')
        atomic(tf, data)
        atomic(ref, dict(original, state='COMPLETE', configs={'k4-v4': dict(rows=512, mean_kl=.02,
                                                                 max_kl=4, greedy_matching=3, state='MEASURED')}))
        self.assertEqual(parity.evaluate(self.root)['state'], 'FAIL')

    def test_gate_rejects_stale_receipts_after_failed_input(self):
        jobs = [dict(label=label, state='PASS') for case in parity.CASES for label in parity.labels(*case)]
        for state in ('FAIL', 'RUNNING', 'NOT RUN'):
            jobs[-1]['state'] = state
            report = parity.evaluate(self.root, dict(jobs=jobs))
            self.assertEqual(report['state'], 'FAIL')
            self.assertIn(f'is {state}; expected PASS', report['results'][-1]['error'])

    def test_parity_only_missing_inputs_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = dict(label='p4-parity', state='NOT RUN', command=[sys.executable, parity.__file__,
                       '--receipts', str(root), '--gate-summary', str(root / 'summary.json')])
            args = SimpleNamespace(out=root, work=root, budget_minutes=10, only='p4-parity', skip=None,
                                   plan_only=False, sm='89', timeout=10)
            def worker(job, log, limit):
                return subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, timeout=limit).returncode
            with self.assertRaisesRegex(RuntimeError, 'p4-parity failed'), contextlib.redirect_stdout(io.StringIO()):
                gate.execute(args, [job], 'key', {}, worker=worker)
            self.assertEqual(json.loads((root / 'summary.json').read_text())['state'], 'FAIL')
            report = json.loads((root / 'p4-parity/summary.json').read_text())
            self.assertEqual(report['state'], 'FAIL')
            self.assertIn('input job p4-20000-timing-0 is missing', report['results'][0]['error'])

    def test_parity_failure_is_a_gate_failure_after_successful_measurements(self):
        self.scale(0, 1.3)
        jobs = [dict(label=label, state='PASS', command=[])
                for case in parity.CASES for label in parity.labels(*case)]
        jobs.append(dict(label='p4-parity', state='NOT RUN', command=[sys.executable, parity.__file__,
                         '--receipts', str(self.root), '--gate-summary', str(self.root / 'summary.json')]))
        args = SimpleNamespace(out=self.root, work=self.root, budget_minutes=10, only='p4-*,p4ref-exl3-*',
                               skip=None, plan_only=False, sm='89', timeout=10)
        def worker(job, log, limit):
            return subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, timeout=limit).returncode
        with self.assertRaisesRegex(RuntimeError, 'p4-parity failed'), contextlib.redirect_stdout(io.StringIO()):
            gate.execute(args, jobs, 'key', {}, worker=worker)
        report = json.loads((self.root / 'summary.json').read_text())
        self.assertEqual(report['state'], 'FAIL')
        self.assertTrue(all(job['state'] == 'PASS' for job in report['jobs'][:-1]))
        self.assertEqual(report['jobs'][-1]['state'], 'FAIL')

    def test_comparator_measures_high_drift_and_rejects_nonfinite(self):
        bf16, q4 = self.root / 'bf16', self.root / 'q4'
        for folder in (bf16, q4):
            folder.mkdir()
            (folder / 'logits-0.f32').write_bytes(b'')
            (folder / 'greedy-0.json').write_text(json.dumps([1 if folder == bf16 else 2] * 128))
        output = self.root / 'quality.json'
        argv = ['compare', str(bf16), str(q4), '--out', str(output)]
        for value in (.9, float('nan'), float('inf')):
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(llama_kv_compare, 'distributions', return_value=dict(mean_kl=value, max_kl=5)):
                if value == .9:
                    llama_kv_compare.main()
                    report = json.loads(output.read_text())
                    self.assertEqual(report['state'], 'PASS')
                    self.assertFalse(report['greedy_equal'])
                else:
                    with self.assertRaises(ValueError): llama_kv_compare.main()


if __name__ == '__main__':
    unittest.main()
