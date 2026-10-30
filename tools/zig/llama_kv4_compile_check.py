"""Offline KV4 SASS preservation and new-kernel spill audit using the build's nvcc flags."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ['-O3', '--fmad=false', '-std=c++20', '-D__CUDA_NO_HALF_OPERATORS__',
         '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
         '-D__CUDA_NO_HALF2_OPERATORS__', '--expt-relaxed-constexpr']
NEW = {'tf_kv4_prepare128', 'tf_kv4_prompt128'}


def sass(tool, image):
    raw = subprocess.check_output([tool, '--dump-sass', str(image)], text=True)
    parts = re.split(r'Function : (\w+)\n', raw)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def resources(log):
    pattern = (r"Function properties for (\w+)\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
               r"(\d+) bytes spill loads\n.*?Used (\d+) registers([^\n]*)")
    out = {}
    for name, stack, stores, loads, regs, rest in re.findall(pattern, log):
        shared = re.search(r'(\d+) bytes smem', rest)
        out[name] = dict(registers=int(regs), static_shared=int(shared[1]) if shared else 0,
                         stack=int(stack), spill_stores=int(stores), spill_loads=int(loads))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--nvcc', default='/opt/cuda/bin/nvcc')
    p.add_argument('--cuobjdump', default='/opt/cuda/bin/cuobjdump')
    p.add_argument('--sms', default='80,86,89,120,121')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    (args.out / 'compile.json').unlink(missing_ok=True)
    metadata = dict(base=subprocess.check_output(['git', 'rev-parse', args.base], cwd=ROOT, text=True).strip(),
                    nvcc=subprocess.check_output([args.nvcc, '--version'], text=True), flags=FLAGS,
                    sources={name: hashlib.sha256((ROOT / 'zig/kernels/cuda' / name).read_bytes()).hexdigest()
                             for name in ['kv4.cu', 'kv4.cuh', 'kv4_prompt.cuh']})
    with tempfile.TemporaryDirectory(prefix='kv4-sass-') as temp:
        source = Path(temp)
        for name in ['kv4.cu', 'kv4.cuh']:
            raw = subprocess.check_output(['git', 'show', f'{args.base}:zig/kernels/cuda/{name}'], cwd=ROOT)
            (source / name).write_bytes(raw)
        for sm in args.sms.split(','):
            variants = {}
            for arm, path in [('old', source / 'kv4.cu'), ('new', ROOT / 'zig/kernels/cuda/kv4.cu')]:
                cubin = args.out / f'{arm}-{sm}.cubin'
                command = [args.nvcc, '-cubin', f'-arch=sm_{sm}', *FLAGS, '-Xptxas=-v', '-o', str(cubin), str(path)]
                result = subprocess.run(command, capture_output=True, text=True)
                log = result.stdout + result.stderr
                (args.out / f'{arm}-{sm}.log').write_text(log)
                result.check_returncode()
                variants[arm] = sass(args.cuobjdump, cubin)
            old, new = variants['old'], variants['new']
            assert old and old.keys() | NEW == new.keys(), f'symbol set changed on sm_{sm}'
            assert all(old[name] == new[name] for name in old), f'old SASS changed on sm_{sm}'
            usage = resources((args.out / f'new-{sm}.log').read_text())
            for name in NEW:
                assert name in usage
                assert all(usage[name][field] == 0 for field in ['stack', 'spill_stores', 'spill_loads']), (sm, name, usage[name])
            report[sm] = dict(existing_sass_identical=len(old), new={name: usage[name] for name in sorted(NEW)})
            print(json.dumps({sm: report[sm]}), flush=True)
    (args.out / 'compile.json').write_text(json.dumps(dict(metadata=metadata, targets=report), indent=2) + '\n')


if __name__ == '__main__':
    main()
