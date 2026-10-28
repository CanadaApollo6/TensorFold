"""Check independent sampling distributions and trace-derived Nanbeige HTTP requests."""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import llama_serve_check as serving


def stats(args):
    results = []
    for line in args.input.read_text().splitlines():
        case = json.loads(line)
        rule, values = case["sampling"], case["logits"]
        if case["name"] == "penalties":
            for token, count in {7: 3, 6: 1}.items():
                values[token] -= case["penalties"]["presence"] + count * case["penalties"]["frequency"]
        order = sorted(range(len(values)), key=lambda i: (-values[i], i))
        if rule["top_k"]:
            order = order[:rule["top_k"]]
        weights = {i: math.exp((values[i] - max(values)) / rule["temperature"]) for i in order}
        norm = sum(weights.values())
        keep, cumulative = [], 0
        for i in order:
            keep.append(i)
            cumulative += weights[i] / norm
            if 0 < rule["top_p"] < 1 and cumulative >= rule["top_p"]:
                break
        keep = [i for i in keep if weights[i] >= max(weights.values()) * rule["min_p"]]
        norm = sum(weights[i] for i in keep)
        expected = [weights[i] / norm if i in keep else 0 for i in range(len(values))]
        counts = case["counts"]
        assert sum(counts) == case["draws"]
        assert all(c == 0 for c, p in zip(counts, expected) if p == 0), case["name"]
        if len(values) > 8:
            counts = [sum(counts[i:i + 256]) for i in range(0, len(values), 256)]
            expected = [sum(expected[i:i + 256]) for i in range(0, len(values), 256)]
        if args.control and case["name"] == "pilot":
            expected = [1 / len(counts)] * len(counts)
        cells = [(c, p * case["draws"]) for c, p in zip(counts, expected) if p > 0]
        assert min(e for _, e in cells) >= 5
        chi_square = sum((c - e) ** 2 / e for c, e in cells)
        result = dict(name=case["name"], draws=case["draws"], degrees_of_freedom=len(cells) - 1,
                      chi_square=chi_square, threshold=40, support=len(keep),
                      minimum_expected=min(e for _, e in cells), observed=counts, expected=expected)
        results.append(result)
        print(json.dumps(result), flush=True)
        assert chi_square < 40, result
    serving.save(args.output, results)


def trace(args):
    shapes = collections.Counter()
    first = None
    traces = 0
    for line in args.input.open():
        for item in json.loads(line)["traces"]:
            traces += 1
            for call in item["calls"]:
                shapes[json.dumps(call["sampling"], sort_keys=True)] += 1
                if first is None:
                    parents = []
                    node = item["nodes"][call["node"]].get("parent")
                    while node is not None:
                        parents.append(item["nodes"][node]["message"])
                        node = item["nodes"][node].get("parent")
                    first = dict(model=call["model"], messages=list(reversed(parents)),
                                 tools=[dict(type="function", function={k: v for k, v in t.items() if k != "type"})
                                        for t in item["nodes"][0]["tools"]],
                                 **call["sampling"], stream=True)
    assert first and first["messages"]
    args.out.mkdir(parents=True, exist_ok=True)
    serving.save(args.out / "pi-request.json", first)
    serving.save(args.out / "pi-fields.json", dict(source=str(args.input), traces=traces,
                 source_sha256=hashlib.file_digest(args.input.open("rb"), "sha256").hexdigest(),
                 calls=sum(shapes.values()), sampling_shapes=dict(shapes), request_fields=sorted(first),
                 provenance="Reconstructed messages, tools, model and sampling from the first call; stream=true added. No raw HTTP body was retained."))


def request(url, route, body, cancel=False):
    req = urllib.request.Request(url + "/v1/" + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        assert response.status == 200
        if not body.get("stream"):
            return json.load(response)
        assert response.headers["Content-Type"].startswith("text/event-stream")
        events = []
        for line in response:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                return dict(events=events, done=True)
            event = json.loads(line[6:])
            assert "error" not in event, event
            events.append(event)
            if cancel and len(events) >= 3:
                return dict(events=events, cancelled=True)
    raise AssertionError("SSE ended without DONE")


def ids(reply):
    if "events" in reply:
        return next(e["tensorfold"]["token_ids"] for e in reply["events"] if "tensorfold" in e)
    return reply["tensorfold"]["token_ids"]


def serve(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    binary = args.binary.resolve()
    command = [binary, "serve", args.model, "--port", str(port), "--name", "nanbeige-exl3-a1",
               "--context", "32768", "--parallel", "1", "--no-drafts", "--prompt-cache-gib", "0",
               "--keep-warm", "0", "--max-tokens", "512", "--no-update-check"]
    guarded = serving.guard_command(args.out, "server", args.need_mib, command, 180)
    serving.save(args.out / "commands.json", guarded)
    child = None
    replies = {}
    with (args.out / "server.log").open("w") as log:
        guard = subprocess.Popen(guarded, cwd=serving.ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if guard.poll() is not None:
                    raise RuntimeError((args.out / "server.log").read_text())
                children = Path(f"/proc/{guard.pid}/task/{guard.pid}/children").read_text().split()
                for pid in children:
                    try:
                        if Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode() == str(binary):
                            child = int(pid)
                    except FileNotFoundError:
                        pass
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise RuntimeError("server did not become healthy")
            log_text = (args.out / "server.log").read_text()
            assert "temperature 1" in log_text and "top_p 0.95" in log_text, log_text
            p1 = next(c for c in json.loads((serving.ROOT / "zig/src/server/fixtures/llama/prompts.json").read_text())
                      if c["name"] == "p1-chat")
            serial = [json.loads(line) for line in args.serial.read_text().splitlines()]
            for route in ("chat/completions", "completions"):
                label = route.replace("/", "-")
                common = dict(model="nanbeige-exl3-a1", max_tokens=32, return_token_ids=True, ignore_eos=True)
                prompt = p1["request"] if route.startswith("chat") else dict(prompt=p1["ids"])
                body = dict(common, **prompt)
                replies[label + "-no-fields"] = request(url, route, body)
                default = request(url, route, dict(body, seed=1234))
                explicit = request(url, route, dict(body, seed=1234, temperature=1, top_p=0.95, top_k=0, min_p=0))
                assert ids(default) == ids(explicit) == serial[0]["tokens"]
                streamed = request(url, route, dict(body, seed=1234, stream=True))
                assert ids(streamed) == ids(default)
                other = request(url, route, dict(body, seed=1235))
                assert ids(other) != ids(default)
                penalized = request(url, route, dict(body, seed=1234, presence_penalty=1, frequency_penalty=0.5))
                assert ids(penalized) == serial[1]["tokens"]
                override = request(url, route, dict(body, seed=-1, temperature=0.7, top_p=0.8, top_k=7, min_p=0.1, max_tokens=8))
                assert len(ids(override)) == 8
                replies[label] = dict(default=default, explicit=explicit, stream=streamed, other_seed=other,
                                     penalties=penalized, overrides=override)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    queued = list(pool.map(lambda _: request(url, route, dict(body, seed=1234)), range(2)))
                assert all(ids(q) == ids(default) for q in queued)
            pi = json.loads(args.pi.read_text())
            replies["pi"] = request(url, "chat/completions", pi, cancel=True)
            after = request(url, "chat/completions", dict(common, **p1["request"], seed=1234))
            assert ids(after) == serial[0]["tokens"]
            rejected = []
            for field in (dict(repetition_penalty=1.1), dict(temperature=-1), dict(top_p=2)):
                try:
                    request(url, "chat/completions", dict(common, **p1["request"], **field, stream=True))
                except urllib.error.HTTPError as error:
                    assert error.code == 400 and error.headers["Content-Type"].startswith("application/json")
                    rejected.append(dict(field=field, error=json.load(error)))
                else:
                    raise AssertionError(field)
            replies["rejections"] = rejected
            serving.save(args.out / "responses.json", replies)
        finally:
            if child is not None and Path(f"/proc/{child}").exists():
                os.kill(child, signal.SIGTERM)
            guard.wait(timeout=20)
    receipt = serving.check_receipt(args.out, "server", args.need_mib)
    summary = dict(state="PASS", endpoints=2, serial_equal_tokens=64, defaults=dict(temperature=1, top_p=0.95, top_k=0, min_p=0),
                   same_seed_equal=True, different_seed_differs=True, penalties_equal_serial=True,
                   concurrent_equal=True, pi_sse_status=200, pi_max_tokens=pi["max_tokens"],
                   pi_cancelled_after_events=len(replies["pi"]["events"]), cancel_reuse_equal=True,
                   pre_stream_rejections=len(rejected), peak_mib=receipt["peak_process_vram_mib"],
                   minimum_free_mib=receipt["minimum_free_mib"])
    serving.save(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    distribution = sub.add_parser("stats")
    distribution.add_argument("input", type=Path)
    distribution.add_argument("output", type=Path)
    distribution.add_argument("--control", action="store_true")
    capture = sub.add_parser("trace")
    capture.add_argument("input", type=Path)
    capture.add_argument("out", type=Path)
    server = sub.add_parser("serve")
    for name in ("binary", "model", "out", "serial", "pi"):
        server.add_argument(name, type=Path)
    server.add_argument("--need-mib", type=int, default=3500)
    args = parser.parse_args()
    {"stats": stats, "trace": trace, "serve": serve}[args.mode](args)


if __name__ == "__main__":
    main()
