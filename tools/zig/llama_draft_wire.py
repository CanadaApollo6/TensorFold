"""Check greedy JSON and SSE bytes using P2 envelope normalization."""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def normalize(data):
    def identity(match):
        prefix, value = match.group(2), match.group(3)
        n = 16 if prefix.endswith(b"tool-") else 32
        if len(value) != n or value[12:13] != b"4" or (n == 32 and value[16:17] not in [b"8", b"9", b"a", b"b"]):
            raise ValueError("invalid P2 UUID envelope")
        return match.group(1) + prefix + b"0" * n + match.group(4)
    data = re.sub(rb'("id"\s*:\s*")(chatcmpl-(?:tool-)?)([0-9a-f]+)(")', identity, data)
    return re.sub(rb'("created"\s*:\s*)[0-9]+', rb'\g<1>0', data)


def replies(args, drafting):
    label = "drafted" if drafting else "plain"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    binary = (ROOT / "zig-out/native/bin/tensorfold-native").resolve()
    server = [str(binary), "serve", args.model, "--port", str(port), "--name", "gate", "--parallel", "1",
              "--context", str(args.context), "--prompt-cache-gib", "0", "--keep-warm", "0", "--no-update-check"]
    server += ["--drafter", args.draft] if drafting else ["--no-drafts"]
    command = [sys.executable, str(ROOT / "tools/zig/llama_gpu_run.py"), "--need-mib", str(args.need_mib),
               "--timeout", str(args.timeout), "--receipt", str(args.out / f"{label}-vram.json"), "--", *server]
    url = f"http://127.0.0.1:{port}"
    responses = []
    with (args.out / f"{label}.log").open("w") as log:
        guard = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        child = None
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if guard.poll() is not None:
                    raise RuntimeError((args.out / f"{label}.log").read_text())
                children = Path(f"/proc/{guard.pid}/task/{guard.pid}/children").read_text().split()
                for pid in children:
                    try:
                        if Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode() == str(binary):
                            child = int(pid)
                    except FileNotFoundError:
                        continue
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("server readiness timeout")
            for prompt in ["Count to five.", "Write a hello world program.", "Explain why the sky is blue."]:
                for stream in [False, True]:
                    body = dict(model="gate", messages=[dict(role="user", content=prompt)], temperature=0,
                                max_tokens=args.tokens, ignore_eos=True, stream=stream)
                    if stream:
                        body["stream_options"] = dict(include_usage=True)
                    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                     {"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        data = response.read()
                    if b'"error"' in data:
                        raise RuntimeError(data.decode())
                    (args.out / f"{label}-{len(responses)}.txt").write_bytes(data)
                    responses.append(data)
        finally:
            if child is None and guard.poll() is None:
                for pid in Path(f"/proc/{guard.pid}/task/{guard.pid}/children").read_text().split():
                    try:
                        if Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode() == str(binary):
                            child = int(pid)
                    except FileNotFoundError:
                        continue
            if child is not None and Path(f"/proc/{child}").exists():
                os.kill(child, signal.SIGTERM)
            guard.wait(timeout=20)
    return responses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--need-mib", type=int)
    p.add_argument("--format", choices=["bf16", "affine4"], default="affine4")
    p.add_argument("--kv", choices=["16", "4"], default="16")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["TF_LLAMA_DRAFT_TRACE"] = "1"
    os.environ["TF_LLAMA_DRAFT_FORMAT"] = args.format
    os.environ["TF_LLAMA_KV_BITS"] = args.kv
    if args.need_mib is None:
        total = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"], text=True).strip()
        budget = subprocess.run([ROOT / "zig-out/bin/tf-llama-test", "llama-draft-capacity", args.model,
                                 args.draft, total, "512", str(args.context)], capture_output=True, text=True, check=True)
        text = budget.stdout + budget.stderr
        (args.out / "capacity.log").write_text(text)
        kv = "q4" if args.kv == "4" else "bf16"
        need = re.search(rf"NEED draft={args.format} kv={kv} .* need_mib=(\d+)", text)
        if not need:
            raise RuntimeError("no matching allocator requirement")
        args.need_mib = int(need.group(1)) + 512
    plain, drafted = replies(args, False), replies(args, True)
    if not re.search(r"Llama draft offered=[1-7]", (args.out / "drafted.log").read_text()):
        raise RuntimeError("server loaded the drafter but offered no proposals")
    for i, (left, right) in enumerate(zip(plain, drafted, strict=True)):
        if normalize(left) != normalize(right):
            (args.out / f"plain-{i}.txt").write_bytes(left)
            (args.out / f"drafted-{i}.txt").write_bytes(right)
            raise SystemExit(f"wire bytes differ in case {i}")
    result = dict(prompts=3, json_equal=3, sse_equal=3, tokens=args.tokens, normalized=["id", "created"], offered=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
