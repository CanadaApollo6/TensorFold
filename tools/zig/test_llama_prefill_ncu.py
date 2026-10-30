"""CPU checks for bounded Nsight capture selection and incomplete receipt rejection."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import llama_prefill_ncu as ncu
from llama_gate_budget import estimate, job_class


def raw_csv(count=9, duration='1234', bad_grid=False):
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(['ID', 'Kernel Name', 'Grid Size', 'Block Size', 'Metric Name', 'Metric Unit', 'Metric Value'])
    for i, (phase, grid, threads) in enumerate(zip(ncu.PHASES[:count], ncu.GRIDS, ncu.THREADS)):
        name = {'attention_tiles': 'tf_kv4_tiles128', 'attention_merge': 'tf_kv4_tile_merge'}.get(
            phase, 'tf_exl3_linear32_7_2_' + ('8' if threads == 256 else '4'))
        writer.writerow([i, name, str((1, 1, 1) if bad_grid else grid), str((threads, 1, 1)),
                         'gpu__time_duration.sum', 'nsecond', duration])
        for metric in ['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
                       'smsp__warps_eligible.avg.per_cycle_active',
                       'dram__throughput.avg.pct_of_peak_sustained_elapsed']:
            writer.writerow([i, name, str(grid), str((threads, 1, 1)), metric, '%', '1.23'])
    return output.getvalue()


class PrefillNcu(unittest.TestCase):
    def test_capture_requires_all_kernels_valid_durations_and_production_geometry(self):
        self.assertEqual([r['phase'] for r in ncu.counters(raw_csv())], ncu.PHASES)
        for raw in [raw_csv(count=8), raw_csv(duration='nan'), raw_csv(duration='0'),
                    raw_csv(bad_grid=True), '==ERROR== ERR_NVGPUCTRPERM',
                    raw_csv().replace('warps_eligible', 'unknown_counter'),
                    raw_csv().replace('tf_kv4_tile_merge', 'tf_kv4_merge')]:
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                ncu.counters(raw)

    def test_profilers_do_not_calibrate_plain_speed_budget(self):
        for label, kind, seconds in [('profile-kernels-60000-kv4', 'prefill-ncu', 1800),
                                     ('profile-prefill-60000-kv4', 'prefill-profile', 1080)]:
            job = dict(label=label)
            self.assertEqual(job_class(job), kind)
            self.assertEqual(estimate(job, {'prefill-speed': 1}), seconds)

    def test_production_wrapper_writes_bound_receipt_and_refuses_partial_captures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(model=root, runner=root / 'runner', prompts=root / 'prompts.json',
                                      out=root / 'out', timeout=1800)
            args.runner.write_bytes(b'CPU runner')
            args.prompts.write_text(json.dumps([[166100] + [1] * 59999]))
            captured = []
            def run(command, **kwargs):
                captured.append((command, kwargs['env']))
                (args.out / 'counters.csv').write_text(raw_csv())
                for state in ['START', 'STOP']:
                    kwargs['stdout'].write(f'LLAMA_PREFILL_NCU run=2 position=59904 rows=96 state={state}\n')
                return subprocess.CompletedProcess(command, 0)
            with patch.object(ncu, 'find_ncu', return_value='/ncu'), \
                    patch.object(ncu, 'engine_need', return_value=22000), \
                    patch.object(ncu.subprocess, 'check_output', return_value='CPU metadata'), \
                    patch.object(ncu.subprocess, 'run', side_effect=run):
                ncu.run(args)
                report = json.loads((args.out / 'summary.json').read_text())
                self.assertEqual(report['state'], 'PASS')
                self.assertEqual(len(report['kernels']), 9)
                command, env = captured[0]
                self.assertEqual(command[command.index('--launch-count') + 1], '9')
                self.assertEqual(command[command.index('--profile-from-start') + 1], 'off')
                self.assertEqual(command[command.index('--target-processes') + 1], 'all')
                self.assertIn('llama_gpu_run.py', ' '.join(command))
                self.assertEqual(env['TF_LLAMA_PREFILL_ROWS'], '128')
                self.assertEqual(env['TF_LLAMA_PREFILL_NCU'], '1')
                def incomplete(command, **kwargs):
                    run(command, **kwargs)
                    (args.out / 'counters.csv').write_text(raw_csv(count=8))
                with patch.object(ncu.subprocess, 'run', side_effect=incomplete), self.assertRaises(ValueError):
                    ncu.run(args)
                self.assertEqual(json.loads((args.out / 'summary.json').read_text())['state'], 'FAIL')
                with patch.object(ncu, 'find_ncu', side_effect=FileNotFoundError('ncu missing')), \
                        self.assertRaises(FileNotFoundError):
                    ncu.run(args)
                self.assertEqual(json.loads((args.out / 'summary.json').read_text())['state'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
