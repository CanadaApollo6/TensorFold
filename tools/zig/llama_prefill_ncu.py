"""Capture first-layer kernel counters in the final chunk of the second 60K KV4 prefill."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from llama_gate_state import atomic, digest, identity
from llama_prefill_check import engine_need

ROOT = Path(__file__).resolve().parents[2]
FILTER = r'regex:^(tf_kv4_tiles128|tf_kv4_tile_merge|tf_exl3_linear(32)?_[0-9]+_[0-2]_[48])$'
PHASES = ['q', 'k', 'v', 'attention_tiles', 'attention_merge', 'o', 'gate', 'up', 'down']
GRIDS = [(224, 1, 1), (64, 1, 1), (64, 1, 1), (6, 8, 30), (56, 96, 1),
         (224, 1, 1), (172, 1, 1), (172, 1, 1), (224, 1, 1)]
THREADS = [128, 128, 128, 224, 128, 128, 256, 256, 128]
SECTIONS = ['SpeedOfLight', 'MemoryWorkloadAnalysis', 'MemoryWorkloadAnalysis_Tables', 'SchedulerStats', 'WarpStateStats',
            'Occupancy', 'SourceCounters', 'LaunchStats']
ENVIRONMENT = ['TF_EXL3_PROMPT_R16', 'TF_EXL3_PROMPT_R32', 'EXL3_INT8_GEMV']


def find_ncu():
    found = shutil.which('ncu')
    if found:
        return found
    paths = []
    for root in [Path('/opt/nvidia'), Path('/usr/local/cuda')]:
        for pattern in ['nsight-compute*/ncu', 'nsight-compute*/*/ncu']:
            paths += sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError('ncu missing; use a CUDA devel image with cuda-nsight-compute-12-8 installed')
    return str(paths[-1])


def counters(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError('no Nsight raw CSV header; check counters permissions and ncu.log')
    launches = {}
    for row in csv.DictReader(io.StringIO('\n'.join(lines[start:]))):
        if not (row.get('ID') or '').isdigit():
            continue
        item = launches.setdefault(int(row['ID']), dict(kernel=row['Kernel Name'], metrics={},
                                   grid=tuple(map(int, re.findall(r'\d+', row['Grid Size']))),
                                   block=tuple(map(int, re.findall(r'\d+', row['Block Size'])))))
        item['metrics'][row['Metric Name']] = dict(unit=row['Metric Unit'], value=row['Metric Value'])
    if len(launches) != len(PHASES):
        raise ValueError(f'expected nine first-layer kernels, captured {len(launches)}')
    result = []
    for phase, grid, threads, (_, item) in zip(PHASES, GRIDS, THREADS, sorted(launches.items())):
        kernel = item['kernel']
        expected = {'attention_tiles': 'tf_kv4_tiles128', 'attention_merge': 'tf_kv4_tile_merge'}
        if phase in expected:
            valid = kernel == expected[phase]
        else:
            valid = re.fullmatch(r'tf_exl3_linear(32)?_[0-9]+_[0-2]_[48]', kernel)
        if not valid:
            raise ValueError(f'unexpected kernel for {phase}: {kernel}')
        if item['grid'] != grid or item['block'] != (threads, 1, 1):
            raise ValueError(f'unexpected launch geometry for {phase}: {item}')
        duration = item['metrics'].get('gpu__time_duration.sum', {})
        value = float(duration.get('value', 'nan').replace(',', ''))
        if not math.isfinite(value) or value <= 0 or duration.get('unit') not in ['nsecond', 'usecond', 'msecond']:
            raise ValueError(f'missing or invalid duration for {phase}')
        for group in ['warp_issue_stalled', 'warps_eligible', 'throughput']:
            values = [m['value'] for name, m in item['metrics'].items() if group in name]
            if not values or not any(re.fullmatch(r'[0-9.,]+', v) for v in values):
                raise ValueError(f'missing hardware counters for {phase}: {group}')
        selected = {k: v for k, v in item['metrics'].items() if any(part in k for part in
                    ['time_duration', 'throughput', 'warps_active', 'warps_eligible', 'issue_active',
                     'warp_issue_stalled', 'bank_conflicts', 'hit_rate', 'registers_per_thread', 'occupancy_limit'])}
        result.append(dict(phase=phase, kernel=kernel, grid=grid, threads=threads, metrics=selected))
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = dict(state='RUNNING', diagnostic_only=True, tokens=60000, position=59904, rows=96,
                   run=2, layer=0, cache_control='none', clock_control='none', replay='kernel',
                   environment={name: os.environ.get(name) for name in ENVIRONMENT})
    path = args.out / 'summary.json'
    atomic(path, receipt)
    try:
        histories = json.loads(args.prompts.read_text())
        if len(histories[0]) != 60000 or histories[0][0] != 166100:
            raise ValueError('requires a BOS-prefixed 60000-token history')
        prompt = args.out / 'prompt.json'
        prompt.write_text(json.dumps([histories[0]]) + '\n')
        ncu = find_ncu()
        receipt['ncu_version'] = subprocess.check_output([ncu, '--version'], text=True)
        receipt['runner'] = identity(args.runner)
        receipt['fatbins'] = identity(ROOT / 'zig-out/fatbin') if (ROOT / 'zig-out/fatbin').exists() else None
        receipt['source_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        receipt['prompt_sha256'] = digest(prompt)
        raw = args.out / 'counters.csv'
        raw.unlink(missing_ok=True)
        (args.out / 'kernels.ncu-rep').unlink(missing_ok=True)
        need = engine_need(argparse.Namespace(kv='4', draft='none', runner=args.runner, model=args.model), 60000)
        command = [ncu, '--profile-from-start', 'off', '--target-processes', 'all',
                   '--kernel-name-base', 'function', '--kernel-name', FILTER, '--launch-count', '9',
                   '--replay-mode', 'kernel', '--cache-control', 'none', '--clock-control', 'none',
                   '--csv', '--page', 'raw', '--print-kernel-base', 'function', '--log-file', str(raw), '--export', str(args.out / 'kernels'),
                   '--force-overwrite']
        for section in SECTIONS:
            command += ['--section', section]
        command += [sys.executable, str(ROOT / 'tools/zig/llama_gpu_run.py'), '--need-mib', str(need),
                    '--timeout', str(args.timeout), '--receipt', str(args.out / 'vram.json'), '--',
                    str(args.runner), 'llama-prefill', str(args.model), str(prompt),
                    str(args.out / 'timing'), 'none', 'timing']
        receipt['command'] = command
        atomic(path, receipt)
        env = dict(os.environ, TF_LLAMA_KV_BITS='4', TF_LLAMA_PREFILL_ROWS='128',
                   TF_LLAMA_PREFILL_PROFILE='1', TF_LLAMA_PREFILL_NCU='1')
        with (args.out / 'ncu.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
        log = (args.out / 'ncu.log').read_text()
        for state in ['START', 'STOP']:
            marker = f'LLAMA_PREFILL_NCU run=2 position=59904 rows=96 state={state}'
            if log.count(marker) != 1:
                raise ValueError(f'expected one capture marker: {marker}')
        receipt['kernels'] = counters(raw.read_text())
        receipt['state'] = 'PASS'
    except BaseException as error:
        receipt.update(state='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic(path, receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'runner', 'prompts', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    run(p.parse_args())


if __name__ == '__main__':
    main()
