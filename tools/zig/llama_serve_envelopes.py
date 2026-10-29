"""Execute the pod's actual chat coroutine with scripted token text and capture its response bytes."""
import ast
import asyncio
import json
import re
import time
import uuid


class Request:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body

    async def is_disconnected(self):
        return False


class Stream:
    def __init__(self, events, **kwargs):
        self.events = events


class Torch:
    long = None

    @staticmethod
    def tensor(ids, **kwargs):
        return ids


def normalize(text):
    text = re.sub(r'chatcmpl-tool-[0-9a-f]{16}', 'chatcmpl-tool-' + '0' * 16, text)
    text = re.sub(r'chatcmpl-[0-9a-f]{32}', 'chatcmpl-' + '0' * 32, text)
    return re.sub(r'("created": ?)[0-9]+', r'\g<1>0', text)


def oracle(path, scope, hf):
    module = ast.parse(path.read_text())
    chat = next(n for n in module.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'chat')
    chat.decorator_list = []
    state = dict(chunks=[], count=0, eos='stop')

    class HF:
        def apply_chat_template(self, *args, **kwargs):
            return hf.apply_chat_template(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return hf(*args, **kwargs)

    class Job:
        def __init__(self, *args, **kwargs):
            state['params'] = kwargs

        def __aiter__(self):
            return self.run()

        async def run(self):
            for chunk in state['chunks']:
                yield dict(text=chunk)
            yield dict(eos=True, new_tokens=state['count'], eos_reason=state['eos'])

        async def cancel(self):
            pass

    def error(status, message, kind='BadRequestError'):
        return dict(error=dict(message=message, type=kind, param=None, code=status))

    namespace = dict(scope, asyncio=asyncio, time=time, Request=Request,
                     S=dict(names=['nanbeige'], hf=HF(), gen=None), torch=Torch, AsyncJob=Job,
                     ComboSampler=lambda **kwargs: kwargs, CONTEXT=4096, STOP_IDS=[166101, 166102],
                     StreamingResponse=Stream, error=error)
    exec(compile(ast.Module([chat], type_ignores=[]), str(path), 'exec'), namespace)

    async def label(case):
        body = dict(messages=[dict(role='user', content='x')], tools=case['tools'], max_tokens=512)
        body.update(case.get('extra', {}))
        raw_ids = case.get('ids')
        if raw_ids is None:
            raw_ids = hf(case['raw'], add_special_tokens=False)['input_ids']
        want = body.get('max_completion_tokens') or body.get('max_tokens') or len(raw_ids) + 1
        ids = raw_ids[:want]
        state.update(chunks=[], count=len(ids), eos='max_new_tokens' if len(raw_ids) >= want else 'stop')
        sent = ''
        for i in range(1, len(ids) + 1):
            decoded = hf.decode(ids[:i], skip_special_tokens=False)
            if decoded.endswith('\ufffd'):
                continue
            assert decoded.startswith(sent)
            state['chunks'].append(decoded[len(sent):])
            sent = decoded
        state['params'] = None
        whole = await namespace['chat'](Request(body))
        case['request'] = body
        case['status'] = whole.get('error', {}).get('code', 200)
        case['json_bytes'] = normalize(json.dumps(whole, ensure_ascii=False, separators=(',', ':')))
        streamed = await namespace['chat'](Request(dict(body, stream=True)))
        if isinstance(streamed, Stream):
            frames = [data async for data in streamed.events]
            case['sse_bytes'] = normalize(''.join(frames))
        else:
            case['sse_bytes'] = normalize(json.dumps(streamed, ensure_ascii=False, separators=(',', ':')))
        case['request_params'] = None if state['params'] is None else dict(max_tokens=state['params']['max_new_tokens'],
                                      sampler=state['params']['sampler'], seed=state['params']['seed'])
        return case

    return label


def cases(typed, tools):
    out = [dict(name=name, raw=raw, tools=offered, extra=extra) for name, raw, offered, extra in [
        ('media', '', [], dict(model='image-alias', messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url='image'))])])),
        ('context', '', [], dict(max_tokens=4096)),
        ('empty', '', [], {}), ('no-answer', '<think>\nPlan.\n', [], {}),
        ('empty-answer', 'Plan.</think> \n', [], {}), ('answer', '</think> answer ', [], {}),
        ('unicode', '<think>\n计划💡。\n</think> 答复 ', [], {}),
        ('tools-empty', '', tools, {}), ('tools-prose', 'Plan.</think> answer ', tools, {}),
        ('calls', '</think><tool_call><function=run></tool_call><tool_call><function=run></tool_call>', tools, {}),
        ('parallel-false', '</think><tool_call><function=run></tool_call><tool_call><function=run></tool_call>',
         tools, dict(parallel_tool_calls=False)),
        ('usage', 'Plan.</think>answer', [], dict(stream_options=dict(include_usage=True))),
        ('length', 'thinking continues at length', [], dict(max_tokens=1)),
        ('ignored-settings', 'Plan.</think>answerSTOPtail', tools,
         dict(tool_choice='none', stop='STOP', presence_penalty=2, repetition_penalty=2, ignore_eos=True)),
        ('required-ignored', '</think>answer', tools, dict(tool_choice='required', response_format=dict(type='json_object'))),
        ('model-echo', '</think>answer', [], dict(model='requested-alias')),
        ('null-params', '</think>answer', [], dict(temperature=None, top_p=None, top_k=None, min_p=None, seed=None)),
        ('false-kwargs', '</think>answer', [], dict(chat_template_kwargs=[])),
        ('zero-limits', '</think>answer', [], dict(max_completion_tokens=0, max_tokens=0)),
        ('priority-limit', 'thinking continues at length', [], dict(max_completion_tokens=1, max_tokens=512)),
    ]]
    out += [dict(c) for c in typed if c['name'] in {'lfm-prose', 'lfm-argument', 'duplicate-parameters'}]
    for kind in ['i', 'n', 'b', 'o', 'a', 'e', 'u', 'z', 'v', 'w', 'x']:
        out += [dict(c) for c in typed if c['name'].startswith('typed-' + kind + '-')][:4]
    return out
