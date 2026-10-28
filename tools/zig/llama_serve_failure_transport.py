"""Check the pod's real HTTP/1.1 aborts with uvicorn and a scripted CPU generator."""
import argparse
import asyncio
import json
import socket
import types
from pathlib import Path

from llama_serve_failures import AppOracle
from llama_serve_envelopes import normalize


def decode(data):
    head, _, body = data.partition(b'\r\n\r\n')
    headers = {k.decode().lower(): v.decode().strip() for line in head.split(b'\r\n')[1:]
               for k, v in [line.split(b':', 1)]}
    complete = True
    if headers.get('transfer-encoding') == 'chunked':
        at, parts, complete = 0, [], False
        while at < len(body):
            end = body.index(b'\r\n', at)
            size = int(body[at:end], 16)
            at = end + 2
            if size == 0:
                assert body[at:] == b'\r\n'
                complete = True
                break
            parts.append(body[at:at + size])
            assert body[at + size:at + size + 2] == b'\r\n'
            at += size + 2
        body = b''.join(parts)
    return dict(status=int(head.split(b' ')[1]), headers=headers, body=normalize(body.decode('utf8')),
                complete=complete, connection_end='EOF', raw=normalize(data.decode('utf8')))


async def verify(model, pod, fixtures, destination):
    import uvicorn
    from transformers import AutoTokenizer
    rows = json.loads(fixtures.read_text())
    rows += json.loads(fixtures.with_name("failure_faults.json").read_text())
    hf = AutoTokenizer.from_pretrained(model, local_files_only=True)
    oracle = AppOracle(pod, hf)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    sock.setblocking(False)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(oracle.app, log_level='critical', lifespan='off', http='h11'))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    results = []
    try:
        for _ in range(1000):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(.01)
        assert server.started
        names = {'bytes', 'lone-high', 'second-call-serialization', 'second-call-encoding',
                 'literal-parser-memory', 'json-recursion', 'frozenset-fallback', 'model-surrogate',
                 'memory-reasoning', 'memory-producer-runtime'}
        for row in rows:
            if row['name'] not in names:
                continue
            real_json, real_job = oracle.env['json'], oracle.env['AsyncJob']
            class FailedJob(real_job):
                async def run(self):
                    yield dict(text='Plan.</think>')
                    raise RuntimeError('generation failed')

            def dumps(value, **kwargs):
                choices = value.get('choices') or [{}] if isinstance(value, dict) else [{}]
                if 'reasoning' in choices[0].get('delta', {}):
                    raise MemoryError()
                return json.dumps(value, **kwargs)

            if row['name'] == 'memory-producer-runtime':
                oracle.env['AsyncJob'] = FailedJob
            if row['name'] == 'memory-reasoning':
                oracle.env['json'] = types.SimpleNamespace(dumps=dumps, loads=json.loads, JSONDecodeError=json.JSONDecodeError)
            oracle.chunks(hf(row['raw'], add_special_tokens=False)['input_ids'])
            for key, stream in [('json', False), ('sse', True)]:
                expected = row[key]
                connection = b'' if expected['status'] == 500 or not expected['complete'] else b'Connection: close\r\n'
                body = json.dumps(dict(row['request'], stream=stream)).encode()
                reader, writer = await asyncio.open_connection('127.0.0.1', port)
                writer.write(b'POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n' + connection +
                             b'Content-Type: application/json\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
                await writer.drain()
                wire = await asyncio.wait_for(reader.read(), 10)
                writer.close()
                await writer.wait_closed()
                got = decode(wire)
                for field in ['status', 'body', 'complete']:
                    assert got[field] == expected[field], (row['name'], key, field)
                assert got['headers']['content-type'] == expected['headers']['content-type']
                results.append(dict(name=row['name'], mode=key, **got))
            oracle.env['json'], oracle.env['AsyncJob'] = real_json, real_job
    finally:
        server.should_exit = True
        await task
        sock.close()
    destination.write_text(json.dumps(dict(uvicorn=uvicorn.__version__, cases=results), indent=1) + '\n')
    print(f'{len(results)} real TCP responses match ASGI bytes and completion; aborted SSE omits the zero chunk')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'pod', 'fixtures', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    asyncio.run(verify(args.model, args.pod, args.fixtures, args.output))
