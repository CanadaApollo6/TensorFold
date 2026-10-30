"""Exercise the production orchestration, fingerprints, and durable crash recovery using CPU workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from llama_gate_state import atomic
from llama_gate_budget import SETUP_SECONDS, SYNC_SECONDS


def phase(root, crash):
    import llama_full_gate as gate
    execute = gate.execute
    calls = []
    def run(args, jobs, binding, inputs):
        def worker(job, log, limit):
            calls.append(job['label'])
            if crash and len(calls) == 3:
                os._exit(73)
            command = [sys.executable, __file__, '--stub', job['label'], str(args.out)]
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=limit).returncode
        return execute(args, jobs, binding, inputs, worker=worker)
    argv = ['gate', '--sm', '89', '--work', str(root / 'work'), '--out', str(root / 'out')]
    for name in ['model', 'draft', 'pod', 'smoke', 'baseline-runner', 'runner', 'wire-runner', 'replay-runner', 'binary', 'histories']:
        argv += ['--' + name, str(root / ('histories.json' if name == 'histories' else name))]
    with patch.object(gate, 'execute', side_effect=run), patch.object(sys, 'argv', argv):
        gate.main()
    atomic(root / ('resume-calls.json' if not crash else 'crash-calls.json'), calls)


def exercise(out=None):
    with tempfile.TemporaryDirectory(prefix='llama-gate-cpu-') as temp:
        root = Path(temp)
        for name in ['model', 'draft']:
            (root / name).mkdir()
            (root / name / 'config.json').write_text('{}')
            (root / name / 'weights.safetensors').write_text('CPU stub weights')
        for name in ['pod', 'smoke', 'baseline-runner', 'runner', 'wire-runner', 'replay-runner', 'binary']:
            (root / name).write_text('CPU stub ' + name)
        (root / 'histories.json').write_text(json.dumps([[166100, i] + [i] * 99998 for i in [1, 2, 3]]))
        commands = []
        def launch(kind):
            cmd = [sys.executable, __file__, '--phase', kind, str(root)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            commands.append(dict(command=cmd, exit=result.returncode, log=result.stdout + result.stderr))
            return result
        assert launch('crash').returncode == 73
        before = json.loads((root / 'out/summary.json').read_text())
        assert [j['state'] for j in before['jobs'][:3]] == ['PASS', 'PASS', 'RUNNING']
        assert launch('resume').returncode == 0, commands[-1]['log']
        after = json.loads((root / 'out/summary.json').read_text())
        assert after['state'] == 'PASS' and len(after['jobs']) == 46
        assert after['jobs'][2]['attempts'] == 2
        assert all(j['attempts'] == 1 for j in after['jobs'][:2])
        assert len(json.loads((root / 'resume-calls.json').read_text())) == 44
        assert launch('resume').returncode == 0
        assert json.loads((root / 'resume-calls.json').read_text()) == []
        (root / 'model/weights.safetensors').write_text('changed input')
        assert launch('resume').returncode != 0
        assert 'fingerprint mismatch' in commands[-1]['log']
        report = dict(state='PASS', jobs=46, crash_exit=73, resumed_jobs=44, completed_reruns=0,
                      changed_model_refused=True, gpu_calls=0, commands=commands,
                      projection=after['projection'], plan=[dict(label=j['label'], command=j['command']) for j in after['jobs']])
        first_projection = next(line.removeprefix('PROJECTED ') for line in commands[0]['log'].splitlines() if line.startswith('PROJECTED '))
        report['planning_projection'] = json.loads(first_projection)
        report['planning_projection']['setup_seconds'] = SETUP_SECONDS
        report['planning_projection']['whole_job_seconds'] = SETUP_SECONDS + report['planning_projection']['total_seconds'] + SYNC_SECONDS
        if out:
            out.mkdir(parents=True, exist_ok=True)
            atomic(out / 'dry-run.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'commands'}, indent=2))
        return report


def main():
    if sys.argv[1] == '--stub':
        label, out = sys.argv[2], Path(sys.argv[3])
        folder = out / label
        folder.mkdir(exist_ok=True)
        result = dict(state='PASS', prefill=[dict(prefill_tok_s=600)] * 2,
                      runs=[dict(decode_tok_s=50)] * 10)
        atomic(folder / ('comparison.json' if label == 'p1-speed' else 'summary.json'), result)
        print('CPU STUB ' + label)
    elif sys.argv[1] == '--phase':
        phase(Path(sys.argv[3]), sys.argv[2] == 'crash')
    else:
        exercise()


if __name__ == '__main__':
    main()
