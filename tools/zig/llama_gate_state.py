"""Durable paid-gate checkpoints and content identities."""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    if path.is_file():
        return digest(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {str(f.relative_to(path)): digest(f) for f in sorted(path.rglob('*')) if f.is_file()}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resume(path, binding, jobs):
    if not path.exists():
        return dict(state='NOT RUN', fingerprint=binding, jobs=jobs)
    saved = json.loads(path.read_text())
    if saved['fingerprint'] != binding:
        raise ValueError('resume fingerprint mismatch; use a new --work and --out for changed inputs')
    prior = {job['label']: job for job in saved['jobs']}
    for job in jobs:
        old = prior.get(job['label'])
        if old:
            job["attempts"] = old.get("attempts", 0)
        if old and old['command'] == job['command'] and old['state'] == 'PASS':
            if not old.get('artifacts') or any(not Path(p).is_file() or digest(p) != h
                                              for p, h in old['artifacts'].items()):
                raise ValueError(f"resume artifact mismatch for {job['label']}")
            job.update(old)
    saved.update(jobs=jobs, state='NOT RUN')
    return saved
