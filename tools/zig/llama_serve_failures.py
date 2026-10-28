"""Label output failures through the pod's FastAPI app without importing torch or EXL3."""
import argparse
import ast
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import types
import uuid
from pathlib import Path

from llama_serve_envelopes import normalize, Torch
from llama_serve_reference import pod_functions, FIXTURES


def call(value, key='payload'):
    return '<tool_call><function=run><parameter=' + key + '>' + value + '</parameter></function></tool_call>'


def cases():
    properties = dict(payload={'type': 'object'}, integer={'type': 'integer'}, number={'type': 'number'})
    tools = [dict(type='function', function=dict(name='run', parameters=dict(type='object', properties=properties)))]
    literals = [
        ('bytes', "b'a'"), ('empty-bytes', "b''"), ('raw-bytes', r"br'\x41'"),
        ('nested-bytes', "{'x':b'a'}"), ('set', '{1,2}'), ('empty-set', 'set()'),
        ('nested-set', "{'x':{1,2}}"), ('complex', '1+2j'), ('imaginary', '-2j'),
        ('complex-overflow', '9' * 400 + '+1j'),
        ('complex-overflow-before-invalid', '[' + '9' * 400 + '+1j,bad]'),
        ('invalid-before-complex-overflow', '[bad,' + '9' * 400 + '+1j]'),
        ('complex-key-collides', '{0:1,0j:2}'),
        ('complex-large-key-collides', '{9007199254740992:1,(9007199254740992+0j):2}'),
        ('complex-large-key-distinct', '{9007199254740993:1,(9007199254740992+0j):2}'),
        ('ellipsis', '...'), ('nested-ellipsis', '[...]'), ('tuple-key', '{(1,):1}'),
        ('bytes-key', "{b'a':1}"), ('complex-key', '{1j:1}'), ('ellipsis-key', '{...:1}'),
        ('list-key', '{[]:1}'), ('dict-key', '{{}:1}'), ('set-key', '{{1}:1}'),
        ('tuple-list-key', '{([],):1}'), ('unhashable-set-item', '{[1]}'),
        ('conversion-before-invalid', '{[]:1,bad:2}'), ('invalid-before-conversion', '{bad:2,[]:1}'),
        ('invalid-value-before-hash', '{[]:bad}'), ('invalid-after-bytes', "[b'a',bad]"),
        ('overwritten-bytes', "{'x':b'a','x':1}"), ('overwritten-set', "{'x':set(),'x':1}"),
        ('lone-high', r"'\ud800'"), ('lone-low', r"'\udfff'"), ('literal-surrogate-pair', r"'\ud83d\ude00'"),
        ('json-high', r'"\ud800"'), ('json-surrogate-pair', r'"\ud83d\ude00"'),
        ('surrogate-key', r"{'\ud800':1}"), ('nested-surrogate', r"{'x':'\ud800'}"),
        ('overwritten-surrogate', r"{'x':'\ud800','x':1}"),
        ('frozenset-fallback', 'frozenset()'), ('ellipsis-name-fallback', 'Ellipsis'),
        ('invalid-bytes', r"b'\x+1'"), ('non-ascii-bytes', "b'雪'"), ('invalid-set', '{1,2'),
        ('decimal-limit', '9' * 4301), ('decimal-limit-json', '[' + '9' * 4301 + ']'),
        ('decimal-limit-literal', "{'x':" + '9' * 4301 + '}'),
        ('hex-limit', '0x' + 'f' * 3572), ('binary-limit', '0b' + '1' * 14285),
        ('octal-limit', '0o' + '7' * 4762), ('hex-at-limit', '0x' + 'f' * 3571),
        ('hex-leading-zeroes', '0x' + '0' * 5000 + '1'),
        ('literal-nesting', '(' * 201 + '1' + ')' * 201),
        ('json-depth-901', '[' * 901 + '0' + ']' * 901),
        ('json-recursion', '[' * 10000 + '0' + ']' * 10000),
        ('json-object-recursion', '{"x":' * 10000 + '0' + '}' * 10000),
        ('json-recursion-invalid-tail', '[' * 10000 + '!' + ']' * 10000),
        ('literal-recursion', '0+' * 10000 + '0'),
        ('literal-parser-memory', '+' * 6000 + '1'),
        ('literal-invert-memory', '~' * 6000 + '1'),
        ('literal-not-memory', 'not ' * 6000 + 'True'),
        ('quoted-long-operators', repr('+' * 6000)),
    ]
    for op in ['*', '/', '//', '%', '<<', '>>', '|', '^', '&', '@']:
        literals.append(('ast-chain-' + op, ('1' + op) * 10000 + '1'))
    literals += [('power-parser-memory', '1**' * 3500 + '1'),
                 ('conditional-parser-memory', '1 if 1 else ' * 6000 + '1'),
                 ('lambda-parser-memory', 'lambda: ' * 3500 + '1')]
    for depth in [9980, 9990, 9991, 9992, 9993, 9994]:
        literals.append((f'json-depth-{depth}', '[' * depth + '0' + ']' * depth))
    out = [dict(name=name, raw='Plan.</think>prefix' + call(value), tools=tools) for name, value in literals]
    for key in ['integer', 'number']:
        out.append(dict(name=key + '-decimal-limit', raw='</think>' + call('9' * 4301, key), tools=tools))
    for value in ["b'a'", r"'\ud800'", '0x' + 'f' * 3572]:
        label = 'bytes' if value[0] == 'b' else 'surrogate' if value[0] == "'" else 'huge-int'
        out.append(dict(name='duplicate-parameter-' + label, raw='</think>' + call(value).replace(
            '</function>', '<parameter=payload>1</parameter></function>'), tools=tools))
    out += [
        dict(name='second-call-serialization', raw='Plan.</think>prefix' + call('1') + call("b'a'"), tools=tools),
        dict(name='second-call-encoding', raw='Plan.</think>prefix' + call('1') + call(r"'\ud800'"), tools=tools),
        dict(name='no-think-close', raw=call("b'a'"), tools=tools),
        dict(name='no-tools', raw='Plan.</think>' + call("b'a'"), tools=[]),
        dict(name='model-surrogate', raw='</think>answer', tools=[], extra=dict(model='bad\ud800')),
    ]
    for schema in [1, 'object', [1], True, 0, False, [], '']:
        offered = copy.deepcopy(tools)
        offered[0]['function']['parameters']['properties']['payload'] = schema
        out.append(dict(name='schema-' + repr(schema), raw='</think>' + call('1'), tools=offered))
    malformed_tools = [dict(type='function'), dict(type='function', function={}),
                       dict(type='function', function=dict(name=[])), 1]
    for slot in ['parameters', 'properties']:
        for value in [None, 1, [], False]:
            offered = copy.deepcopy(tools)
            f = offered[0]['function']
            if slot == 'parameters':
                f['parameters'] = value
            else:
                f['parameters']['properties'] = value
            malformed_tools.append(offered[0])
    for i, offered in enumerate(malformed_tools):
        for suffix, raw in [('call', '</think>' + call('1')), ('no-call', '</think>answer'),
                            ('no-close', 'reasoning only')]:
            out.append(dict(name=f'tool-shape-{i}-{suffix}', raw=raw, tools=[offered]))
    for arguments in ['42', 'null', '[]', '"hello"', [1], None]:
        history = [dict(role='user', content='x'), dict(role='assistant', tool_calls=[dict(type='function',
                   function=dict(name='run', arguments=arguments))]), dict(role='user', content='continue')]
        for fmt in ['xml', 'json']:
            out.append(dict(name='history-' + repr(arguments) + '-' + fmt, raw='</think>answer', tools=tools,
                            extra=dict(messages=history, chat_template_kwargs=dict(tool_call_format=fmt))))
    return out


class AppOracle:
    def __init__(self, pod, hf):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse, StreamingResponse
        self.state = {}
        self.hf = hf
        state = self.state

        class Job:
            def __init__(self, *args, **kwargs):
                pass

            def __aiter__(self):
                return self.run()

            async def run(self):
                for text in state['chunks']:
                    yield dict(text=text)
                yield dict(eos=True, new_tokens=state['count'], eos_reason='stop')

            async def cancel(self):
                pass

        env = pod_functions(pod)
        env.update(dict(asyncio=asyncio, time=time, uuid=uuid, Request=Request,
                   JSONResponse=JSONResponse, StreamingResponse=StreamingResponse, torch=Torch, AsyncJob=Job,
                   ComboSampler=lambda **kw: kw, S=dict(names=['nanbeige'], hf=hf, gen=None),
                   CONTEXT=262144, STOP_IDS=[166101, 166102]))
        nodes = [n for n in ast.parse(pod.read_text()).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name in {'chat', 'error'}]
        for node in nodes:
            node.decorator_list = []
        exec(compile(ast.Module(nodes, type_ignores=[]), str(pod), 'exec'), env)
        self.env = env
        self.app = FastAPI()
        self.app.add_api_route('/v1/chat/completions', env['chat'], methods=['POST'])

    async def request(self, body):
        data = json.dumps(body).encode()
        events = []
        used = False

        async def receive():
            nonlocal used
            if not used:
                used = True
                return dict(type='http.request', body=data, more_body=False)
            await asyncio.Event().wait()

        async def send(event):
            events.append(event)

        scope = dict(type='http', asgi=dict(version='3.0', spec_version='2.4'), http_version='1.1',
                     scheme='http', method='POST', path='/v1/chat/completions', raw_path=b'/v1/chat/completions',
                     root_path='', query_string=b'', server=('test', 80), client=('local', 1),
                     headers=[(b'content-type', b'application/json')])
        exception = None
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            exception = type(exc).__name__ + ': ' + str(exc)
        head = next(e for e in events if e['type'] == 'http.response.start')
        data = b''.join(e.get('body', b'') for e in events)
        return dict(status=head['status'], headers={k.decode(): v.decode() for k, v in head['headers']},
                    body=normalize(data.decode('utf8')), exception=exception,
                    complete=any(e['type'] == 'http.response.body' and not e.get('more_body', False) for e in events))

    def chunks(self, ids):
        sent = ''
        chunks = []
        for i in range(1, len(ids) + 1):
            text = self.hf.decode(ids[:i], skip_special_tokens=False)
            if text.endswith('\ufffd'):
                continue
            assert text.startswith(sent)
            chunks.append(text[len(sent):])
            sent = text
            if '</think>' in text:
                rest = self.hf.decode(ids, skip_special_tokens=False)[len(sent):]
                if rest:
                    chunks.append(rest)
                break
        self.state.update(chunks=chunks, count=len(ids))

    async def label(self, case):
        ids = self.hf(case['raw'], add_special_tokens=False)['input_ids']
        self.chunks(ids)
        body = dict(messages=[dict(role='user', content='x')], tools=case['tools'], max_tokens=65000,
                    stream_options=dict(include_usage=True))
        body.update(case.get('extra', {}))
        return dict(case, request=body, json=await self.request(body),
                    sse=await self.request(dict(body, stream=True)))


async def fault_labels(oracle):
    from starlette.responses import JSONResponse
    rows = []
    real_json, real_ast = oracle.env['json'], oracle.env['ast']
    real_render = JSONResponse.render
    real_job = oracle.env['AsyncJob']
    raw = 'Plan.</think>prefix' + call("{'x':1}")
    for seam in ['loads', 'literal', 'arguments', 'role', 'reasoning', 'whole', 'producer-runtime']:
        def loads(value, **kwargs):
            if seam == 'loads':
                raise MemoryError()
            return json.loads(value, **kwargs)

        def literal(value):
            if seam == 'literal':
                raise MemoryError()
            return ast.literal_eval(value)

        def dumps(value, **kwargs):
            if isinstance(value, dict):
                delta = (value.get('choices') or [{}])[0].get('delta', {})
                if (seam == 'arguments' and 'payload' in value or
                        seam == 'role' and 'role' in delta or seam == 'reasoning' and 'reasoning' in delta):
                    raise MemoryError()
            return json.dumps(value, **kwargs)

        def render(self, content):
            raise MemoryError()

        class FailedJob(real_job):
            async def run(self):
                yield dict(text='Plan.</think>')
                raise RuntimeError('generation failed')

        if seam == 'producer-runtime':
            oracle.env['AsyncJob'] = FailedJob
        oracle.env['json'] = types.SimpleNamespace(loads=loads, dumps=dumps, JSONDecodeError=json.JSONDecodeError)
        oracle.env['ast'] = types.SimpleNamespace(literal_eval=literal)
        if seam == 'whole':
            JSONResponse.render = render
        try:
            row = await oracle.label(dict(name='memory-' + seam, raw=raw, tools=cases()[0]['tools']))
            rows.append(row)
        finally:
            oracle.env['json'], oracle.env['ast'] = real_json, real_ast
            JSONResponse.render = real_render
            oracle.env['AsyncJob'] = real_job
    return rows


async def generate(model, pod, destination):
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(model, local_files_only=True)
    oracle = AppOracle(pod, hf)
    rows = [await oracle.label(case) for case in cases()]
    destination.write_text('[\n' + ',\n'.join(json.dumps(row, ensure_ascii=True, separators=(',', ':')) for row in rows) + '\n]\n')
    faults = await fault_labels(oracle)
    destination.with_name('failure_faults.json').write_text('[\n' + ',\n'.join(json.dumps(row, ensure_ascii=True, separators=(',', ':')) for row in faults) + '\n]\n')
    sources = dict(python=sys.version, pod_sha256=hashlib.sha256(pod.read_bytes()).hexdigest(),
                   versions={name: importlib.metadata.version(name) for name in ['fastapi', 'starlette', 'transformers']})
    destination.with_suffix('.sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), json_failures=sum(r['json']['status'] == 500 for r in rows),
                         sse_aborts=sum(not r['sse']['complete'] for r in rows)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('pod', type=Path)
    parser.add_argument('--output', type=Path, default=FIXTURES / 'failures.json')
    args = parser.parse_args()
    asyncio.run(generate(args.model, args.pod, args.output))
