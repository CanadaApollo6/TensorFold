"""CPU regressions for shell failure propagation, deadlines, calibration and speed verdicts."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from llama_decode_speed import compare
from llama_gate_budget import calibrate, estimate, job_class
import llama_full_gate as gate

ROOT = Path(__file__).resolve().parents[2]


class PaidJob(unittest.TestCase):
    def test_cpu_dry_records_work_and_sync_failures(self):
        for flags, expected in [({}, 0), ({'FAIL_WORK': '1'}, 23), ({'FAIL_SYNC': 'copy'}, 1), ({'FAIL_SYNC': 'upload'}, 1)]:
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                proc = subprocess.run(['bash', ROOT / 'tools/zig/llama_hf_job.sh'],
                                      env=dict(os.environ, DRY='1', LOCAL_DRY='1', W=tmp,
                                               OUT=str(root / 'out'), WORK=str(root / 'work'), **flags))
                self.assertEqual(proc.returncode, expected)
                self.assertEqual((root / 'out/status.txt').read_text().strip(), 'PASS' if not expected else 'FAIL')
                self.assertIn('CONTAINER_START', (root / 'out/container.log').read_text())

    def test_exact_sync_body_refuses_work_failure_even_if_receipts_succeed(self):
        source = (ROOT / 'tools/zig/llama_hf_job.sh').read_text()
        body = source[source.index('run_sync()'):source.index('finish()')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'out').mkdir()
            mock = root / 'hf'
            mock.write_text('#!/bin/bash\n[[ $3 != */work ]]\n')
            mock.chmod(0o755)
            script = f'W={tmp}; OUT={tmp}/out; WORK={tmp}/work; HF=({mock}); OUT_BUCKET=mock; GATE_DEADLINE=$(( $(date +%s) + 200 ));\n'
            proc = subprocess.run(['bash', '-euc', script + body + '\nsync_out || exit 17\n'])
            self.assertEqual(proc.returncode, 17)
            self.assertEqual((root / 'out/status.txt').read_text().strip(), 'FAIL')

    def test_expired_container_deadline_refuses_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = SimpleNamespace(out=root, work=root, budget_minutes=210, deadline=time.time() - 1,
                                   only=None, skip=None, plan_only=False, sm='89', timeout=10)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'budget'):
                gate.execute(args, [dict(label='p1-reference', command=[], state='NOT RUN')], 'key', {},
                             worker=lambda *a: self.fail('expired deadline ran work'))
            self.assertEqual(json.loads((root / 'summary.json').read_text())['state'], 'BUDGET')

    def test_each_job_class_uses_its_own_measurement(self):
        wire = dict(label='p2')
        kv = dict(label='p4-60000-kl-0')
        draft_wire = dict(label='p3-bf16-kv16-wire')
        self.assertNotEqual(job_class(wire), job_class(draft_wire))
        rates = {job_class(wire): calibrate(wire, 80)}
        self.assertEqual(estimate(wire, rates), 100)
        self.assertEqual(estimate(kv, rates), estimate(kv))
        self.assertEqual(estimate(draft_wire, rates), estimate(draft_wire))

    def test_speed_ratio_and_spread_are_nonblocking(self):
        runs = [dict(prompt=0, rep=i, side=side, tok_s=rate, tokens=[1] * 128)
                for i in range(5) for side, rate in [('tensorfold', 20 + i), ('exllamav3', 40 + i)]]
        result = compare(runs)
        self.assertEqual(result['state'], 'PASS')
        self.assertEqual(result['speed_state'], 'FAIL')
        self.assertEqual(result['prompts'][0]['tensorfold']['spread'], [20, 24])
        self.assertEqual(result['prompts'][0]['ratio'], 22 / 42)


if __name__ == '__main__':
    unittest.main()
