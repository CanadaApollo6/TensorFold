"""Run guarded Nanbeige HTTP checks or construct a read-only-source model slice."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "tools/zig/llama_gpu_run.py"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def slice_model(source, dest, layers):
    dest.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    assert 0 < layers < config["num_hidden_layers"]
    config["num_hidden_layers"] = layers
    save(dest / "config.json", config)
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "generation_config.json",
                 "special_tokens_map.json", "added_tokens.json"):
        if not (dest / name).exists():
            (dest / name).symlink_to(source / name)
    header, selected = {}, []
    length = 0
    for shard in sorted(source.glob("model-*.safetensors")):
        with shard.open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            tensors = json.loads(f.read(size))
        for name, info in tensors.items():
            if name == "__metadata__":
                continue
            if name.startswith("model.layers.") and int(name.split(".")[2]) >= layers:
                continue
            lo, hi = info["data_offsets"]
            header[name] = dict(info, data_offsets=[length, length + hi - lo])
            selected.append((shard, size + 8 + lo, hi - lo))
            length += hi - lo
    data = json.dumps(header, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    digest = hashlib.sha256()
    with (dest / "model.safetensors").open("wb") as out:
        out.write(struct.pack("<Q", len(data)))
        out.write(data)
        for shard, offset, count in selected:
            with shard.open("rb") as f:
                f.seek(offset)
                while count:
                    chunk = f.read(min(count, 8 << 20))
                    assert chunk
                    out.write(chunk)
                    digest.update(chunk)
                    count -= len(chunk)
    save(dest / "slice.json", dict(source=str(source), layers=layers, tensors=len(header),
                                   tensor_bytes=length, tensor_sha256=digest.hexdigest()))
    print(f"slice layers={layers} tensors={len(header)} bytes={length}")


def post(url, body):
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=240) as response:
        value = json.load(response)
    assert "error" not in value, value
    return dict(seconds=time.monotonic() - start, body=body, response=value)


def stream(url, body, cancel=False):
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(dict(body, stream=True,
                                     stream_options=dict(include_usage=True))).encode(),
                                     {"Content-Type": "application/json"})
    events, done = [], False
    with urllib.request.urlopen(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            assert "error" not in event, event
            events.append(event)
            if cancel:
                return events
    assert done
    return events


def tokens(reply):
    return reply["response"]["tensorfold"]["token_ids"]


def guard_command(out, label, need, command, timeout):
    return [sys.executable, str(GUARD), "--need-mib", str(need), "--timeout", str(timeout),
            "--receipt", str(out / f"{label}-vram.json"), "--", *map(str, command)]


def check_receipt(out, label, need):
    receipt = json.loads((out / f"{label}-vram.json").read_text())
    assert not receipt["timeout"], receipt
    assert receipt["minimum_free_mib"] >= 2048, receipt
    assert 0 < receipt["peak_process_vram_mib"] <= need, receipt
    return receipt


def server_checks(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    binary = args.binary.resolve()
    command = [binary, "serve", args.model, "--port", str(port), "--name", "nanbeige-exl3-a1",
               "--context", str(args.context), "--parallel", "1", "--no-drafts", "--temperature", "0",
               "--prompt-cache-gib", "0", "--keep-warm", "0", "--max-tokens", "512", "--no-update-check"]
    guarded = guard_command(args.out, "server", args.need_mib, command, args.timeout)
    save(args.out / "commands.json", dict(server=guarded))
    with (args.out / "server.log").open("w") as log:
        guard = subprocess.Popen(guarded, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        child = None
        result = {}
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if guard.poll() is not None:
                    text = (args.out / "server.log").read_text()
                    if "BLOCKED need" in text:
                        result = dict(state="BLOCKED", admission=text.strip(), commands=guarded)
                        save(args.out / "summary.json", result)
                        print(json.dumps(result), flush=True)
                        raise SystemExit(2)
                    raise RuntimeError(text)
                children = Path(f"/proc/{guard.pid}/task/{guard.pid}/children").read_text().split()
                for pid in children:
                    try:
                        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    except FileNotFoundError:
                        continue
                    if cmdline[0].decode() == str(binary):
                        child = int(pid)
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        assert response.status == 200
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("server did not become healthy")
            base = dict(model="nanbeige-exl3-a1", max_tokens=16, temperature=0, return_token_ids=True)
            bodies = [dict(base, messages=[dict(role="user", content=text)]) for text in
                      ["Count to five.", "Write a hello world program."]]
            solo = [post(url, body) for body in bodies]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                parallel = list(pool.map(lambda b: post(url, b), bodies))
            assert all(tokens(s) == tokens(p) for s, p in zip(solo, parallel))
            assert all(len(tokens(s)) == 16 for s in solo)
            save(args.out / "concurrency.json", dict(solo=solo, concurrent=parallel))
            events = stream(url, bodies[0])
            stream_ids = next(e["tensorfold"]["token_ids"] for e in events if "tensorfold" in e)
            assert stream_ids == tokens(solo[0])
            save(args.out / "stream.json", events)
            stream(url, dict(bodies[1], max_tokens=128), cancel=True)
            after_cancel = post(url, bodies[0])
            assert tokens(after_cancel) == tokens(solo[0])
            save(args.out / "after-cancel.json", after_cancel)
            rejected = []
            for fields in [dict(temperature=-1), dict(top_p=2), dict(repetition_penalty=1.1)]:
                try:
                    post(url, dict(bodies[0], **dict(fields, stream=True)))
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                    assert error.headers["Content-Type"].startswith("application/json")
                    rejected.append(dict(fields=fields, status=error.code, error=json.load(error)))
                else:
                    raise AssertionError(f"request accepted unsupported fields {fields}")
            save(args.out / "rejections.json", rejected)
            result = dict(state="PASS", context=args.context, layers=json.loads((args.model / "config.json").read_text())["num_hidden_layers"],
                          solo_tokens=[len(tokens(s)) for s in solo], concurrent_equal=True, stream_equal=True,
                          cancel_reuse_equal=True, pre_stream_rejections=len(rejected))
            if args.full:
                smoke = subprocess.run([sys.executable, str(args.smoke), "--url", url,
                                        "--concurrency", "1", "--tokens", "128"],
                                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
                (args.out / "smoke.log").write_text(smoke.stdout)
                assert smoke.returncode == 0 and "TOOL_TURN OK" in smoke.stdout and "LOAD concurrency=1" in smoke.stdout, smoke.stdout
                result["tool_smoke"] = "PASS"
                perf_body = dict(base, max_tokens=128, messages=[dict(role="user", content=
                                 "Write a detailed essay on compiler design with code examples.")])
                perf = [post(url, perf_body) for _ in range(5)]
                assert all(len(tokens(p)) == 128 for p in perf)
                assert all(tokens(p) == tokens(perf[0]) for p in perf)
                rates = [p["response"]["tensorfold"]["tokens_per_second"] for p in perf]
                assert all(rate > 0 for rate in rates)
                save(args.out / "throughput.json", dict(requests=perf, decode_tok_s=rates,
                     median=statistics.median(rates), minimum=min(rates), maximum=max(rates)))
                result["decode_tok_s"] = dict(median=statistics.median(rates), minimum=min(rates),
                                              maximum=max(rates), repetitions=5, completion_tokens=128)
            else:
                result["full_model_checks"] = "NOT RUN on a slice"
            if args.oracle:
                oracle = args.out / "oracle-tokens.json"
                seqs = json.loads((args.refs / "sequences.json").read_text())
                prompt = seqs[-1]
                req = urllib.request.Request(url + "/v1/completions", json.dumps(dict(
                    model=base["model"], prompt=prompt, max_tokens=128, temperature=0,
                    return_token_ids=True, ignore_eos=True)).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=240) as response:
                    got = json.load(response)
                save(args.out / "greedy-server.json", got)
                assert got["tensorfold"]["token_ids"] == json.loads(oracle.read_text())
                p1 = next(c for c in json.loads((ROOT / "zig/src/server/fixtures/llama/prompts.json").read_text())
                          if c["name"] == "p1-chat")
                assert prompt == p1["ids"]
                chat = post(url, dict(base, **p1["request"], max_tokens=128, ignore_eos=True))
                assert tokens(chat) == json.loads(oracle.read_text())
                assert chat["response"]["usage"]["prompt_tokens"] == len(prompt)
                save(args.out / "greedy-chat.json", chat)
                result["greedy_equal"] = 128
                result["greedy_paths"] = ["chat", "completions"]
            save(args.out / "summary.json", result)
        finally:
            if child is not None and Path(f"/proc/{child}").exists():
                os.kill(child, signal.SIGTERM)
            guard.wait(timeout=20)
    receipt = check_receipt(args.out, "server", args.need_mib)
    result["peak_mib"] = receipt["peak_process_vram_mib"]
    result["minimum_free_mib"] = receipt["minimum_free_mib"]
    save(args.out / "summary.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cut = sub.add_parser("slice")
    cut.add_argument("model", type=Path)
    cut.add_argument("out", type=Path)
    cut.add_argument("--layers", type=int, default=2)
    serve = sub.add_parser("serve")
    serve.add_argument("binary", type=Path)
    serve.add_argument("model", type=Path)
    serve.add_argument("out", type=Path)
    serve.add_argument("--need-mib", type=int, required=True)
    serve.add_argument("--context", type=int, default=1024)
    serve.add_argument("--timeout", type=int, default=180)
    serve.add_argument("--full", action="store_true")
    serve.add_argument("--smoke", type=Path)
    serve.add_argument("--oracle", type=Path)
    serve.add_argument("--refs", type=Path)
    args = parser.parse_args()
    if args.mode == "slice":
        slice_model(args.model.resolve(), args.out.resolve(), args.layers)
    else:
        if args.oracle:
            args.out.mkdir(parents=True, exist_ok=True)
            command = guard_command(args.out, "oracle", args.need_mib, [args.oracle, "llama-serve-greedy", args.model,
                                    args.refs, "128", args.out / "oracle-tokens.json"], args.timeout)
            save(args.out / "oracle-command.json", command)
            value = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (args.out / "oracle.log").write_text(value.stdout)
            if "BLOCKED need" not in value.stdout:
                assert value.returncode == 0, value.stdout
                check_receipt(args.out, "oracle", args.need_mib)
        server_checks(args)


if __name__ == "__main__":
    main()
