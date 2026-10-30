"""Check greedy JSON and SSE bytes using P2 envelope normalization."""
import argparse
import asyncio
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


def completion_bytes(data):
    def clean(raw):
        value = json.loads(raw)
        for key in ["id", "created", "tensorfold", "speculative"]:
            value.pop(key, None)
        return json.dumps(value, ensure_ascii=False).encode()
    if not data.startswith(b"data: "):
        return clean(data)
    return b"\n\n".join(b"data: " + (b"[DONE]" if frame[6:] == b"[DONE]" else clean(frame[6:]))
                            for frame in data.split(b"\n\n") if frame) + b"\n\n"


def replies(args, drafting):
    label = "drafted" if drafting else "plain"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    binary = args.binary.resolve()
    server = [str(binary), "serve", args.model, "--port", str(port), "--name", "nanbeige", "--parallel", "1",
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
            requests = [dict(model="nanbeige", messages=[dict(role="user", content=prompt)], temperature=0,
                             max_tokens=args.tokens) for prompt in
                        ["Count to five.", "Write a hello world program.", "Explain why the sky is blue."]]
            requests += [dict(requests[0], max_tokens=n) for n in range(1, 9)]
            requests += [dict(requests[0], stop=stop) for stop in [".", "five", "</think>"]]
            requests += [dict(model="nanbeige", prompt="Count to five.", temperature=0, max_tokens=args.tokens,
                              stop=stop, _route="/v1/completions") for stop in [".", "five", "</think>"]]
            if args.reference:
                requests.append(args.reference[0])
            for original in requests:
                for stream in [False, True]:
                    body = dict(original, stream=stream)
                    route = body.pop("_route", "/v1/chat/completions")
                    if stream:
                        body["stream_options"] = dict(include_usage=True)
                    request = urllib.request.Request(url + route, json.dumps(body).encode(),
                                                     {"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        data = response.read()
                    if b'"error"' in data:
                        raise RuntimeError(data.decode())
                    (args.out / f"{label}-{len(responses)}.txt").write_bytes(data)
                    if route == "/v1/completions":
                        data = completion_bytes(data)
                    responses.append(data)
            if args.reference:
                for actual, expected in zip(responses[-2:], args.reference[1], strict=True):
                    # The pod oracle already zeroes ids and created times (llama_serve_envelopes.normalize).
                    if normalize(actual) != expected:
                        raise RuntimeError("P2 application bytes differ from pod with P1 reference tokens")
            if args.smoke and not drafting:
                smoke = subprocess.run([sys.executable, args.smoke, "--url", url, "--concurrency", "1", "--tokens", "128"],
                                       capture_output=True, text=True, timeout=args.timeout)
                (args.out / "smoke.log").write_text(smoke.stdout + smoke.stderr)
                if smoke.returncode or "TOOL_TURN OK" not in smoke.stdout or "LOAD concurrency=1" not in smoke.stdout:
                    raise RuntimeError("P2 server smoke failed")
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
    p.add_argument("--draft")
    p.add_argument("--binary", type=Path, default=ROOT / "zig-out/native/bin/tensorfold-native")
    p.add_argument("--runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-test")
    p.add_argument("--wire-runner", type=Path, default=ROOT / "zig-out/bin/tf-llama-wire-test")
    p.add_argument("--pod", type=Path)
    p.add_argument("--reference-ids", type=Path)
    p.add_argument("--smoke", type=Path)
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
    args.reference = None
    if args.reference_ids:
        if not args.pod:
            p.error("--pod required with --reference-ids")
        from transformers import AutoTokenizer
        from llama_serve_reference import pod_functions
        from llama_serve_envelopes import oracle
        hf = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        fixture = next(c for c in json.loads((ROOT / "zig/src/server/fixtures/llama/prompts.json").read_text())
                       if c["name"] == "p1-chat")
        ids = json.loads(args.reference_ids.read_text())
        case = dict(raw=hf.decode(ids, skip_special_tokens=False), ids=ids, tools=[],
                    extra=dict(fixture["request"], model="nanbeige", temperature=0, max_tokens=len(ids)))
        asyncio.run(oracle(args.pod, pod_functions(args.pod), hf)(case))
        args.reference = (case["request"], [case["json_bytes"].encode(), case["sse_bytes"].encode()])
    if not args.draft and args.need_mib is None:
        p.error("--need-mib required for plain P2 server")
    if args.need_mib is None:
        total = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "--id=0"], text=True).strip()
        budget = subprocess.run([args.runner, "llama-draft-capacity", args.model,
                                 args.draft, total, "512", str(args.context)], capture_output=True, text=True, check=True)
        text = budget.stdout + budget.stderr
        (args.out / "capacity.log").write_text(text)
        kv = "q4" if args.kv == "4" else "bf16"
        need = re.search(rf"NEED draft={args.format} kv={kv} .* need_mib=(\d+)", text)
        if not need:
            raise RuntimeError("no matching allocator requirement")
        args.need_mib = int(need.group(1)) + 512
    if args.draft:
        forced = [sys.executable, ROOT / "tools/zig/llama_gpu_run.py", "--need-mib", str(args.need_mib),
                  "--timeout", str(args.timeout), "--receipt", args.out / "forced-vram.json", "--",
                  args.wire_runner, args.model, args.draft]
        result = subprocess.run(list(map(str, forced)), capture_output=True, text=True, timeout=args.timeout)
        (args.out / "forced.log").write_text(result.stdout + result.stderr)
        if result.returncode or "FORCED_WIRE cases=42" not in result.stdout + result.stderr:
            raise RuntimeError("forced GPU stopping coverage failed; see forced.log")
    plain = replies(args, False)
    if not args.draft:
        result = dict(state="PASS", json_cases=len(plain) // 2, sse_cases=len(plain) // 2,
                      pod_reference_equal=bool(args.reference), smoke=bool(args.smoke))
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    drafted = replies(args, True)
    if not re.search(r"Llama draft offered=[1-7]", (args.out / "drafted.log").read_text()):
        raise RuntimeError("server loaded the drafter but offered no proposals")
    for i, (left, right) in enumerate(zip(plain, drafted, strict=True)):
        if normalize(left) != normalize(right):
            (args.out / f"plain-{i}.txt").write_bytes(left)
            (args.out / f"drafted-{i}.txt").write_bytes(right)
            raise SystemExit(f"wire bytes differ in case {i}")
    result = dict(requests=len(plain) // 2, json_equal=len(plain) // 2, sse_equal=len(plain) // 2, tokens=args.tokens,
                  normalized_chat=["id", "created"],
                  completion_comparison="frames, content, finish_reason and usage; omit identity and native timing/speculation metadata",
                  offered=True, forced_gpu_wire_cases=42, forced_eos_offsets=list(range(2, 9)),
                  forced_stop_offsets=list(range(2, 9)), forced_draws="test-only backend wrapper; real CUDA forwards")
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
