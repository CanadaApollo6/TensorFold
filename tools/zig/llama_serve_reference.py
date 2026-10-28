"""Regenerate Nanbeige prompt and output goldens from HF and the scored pod server."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "zig/src/server/fixtures/llama"


def pod_functions(path):
    names = {"text_of", "prepare", "typed", "split_output"}
    module = ast.parse(path.read_text())
    constants = {"THINK_END", "TOOL_CALL", "FUNCTION", "PARAMETER"}
    nodes = [n for n in module.body if
             isinstance(n, ast.FunctionDef) and n.name in names or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    scope = dict(ast=ast, json=json, uuid=uuid, re=re)
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def tool():
    return {"type": "function", "function": {"name": "bash", "description": "Execute a bash command. 中文 <&>",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}, "timeout": {"type": "number"},
                "count": {"type": "integer"}, "enabled": {"type": "boolean"},
                "options": {"type": "object"}, "maybe": {"type": ["null", "integer"]}}}}}


def prompt_cases():
    base = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Hello 世界."}]
    history = base + [{"role": "assistant", "content": "Done.", "reasoning_content": "Keep this plan."},
                      {"role": "user", "content": "Continue."}]
    call = {"id": "call_1", "type": "function", "function": {"name": "bash",
            "arguments": json.dumps({"command": "printf '中文\\n'", "timeout": 2.5, "enabled": True,
                                     "options": {"b": 2, "a": [1, False]}})}}
    bad_call = json.loads(json.dumps(call))
    bad_call["function"]["arguments"] = "{"
    tools = [tool()]
    calls = base + [{"role": "assistant", "content": None, "reasoning_content": "Inspect it.", "tool_calls": [call]}]
    results = calls + [{"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                       {"role": "tool", "tool_call_id": "call_2", "content": "second"}]
    source = ast.parse((ROOT / "tools/zig/llama_reference.py").read_text())
    make = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "prompts")
    p1 = next(n.value for n in make.body if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == "messages" for t in n.targets))
    return [dict(name=name, request=dict(messages=messages, **extra)) for name, messages, extra in [
        ("p1-chat", ast.literal_eval(p1), {}),
        ("system-user", base, {}), ("default-system", base[1:], {}),
        ("reasoning-history", history, {}), ("reasoning-alias", base + [
            {"role": "assistant", "content": "Answer.", "reasoning": "Alias plan."}, base[1]], {}),
        ("embedded-thinking", base + [{"role": "assistant", "content": "<think>\nPlan.\n</think>\nAnswer."}, base[1]], {}),
        ("tools", base, dict(tools=tools)), ("calls-with-tools", calls, dict(tools=tools)),
        ("calls-without-tools", calls, {}), ("tool-results", results, dict(tools=tools)),
        ("preserve-false", history, dict(chat_template_kwargs=dict(preserve_thinking=False))),
        ("thinking-off", base, dict(chat_template_kwargs=dict(enable_thinking=False))),
        ("json-calls", results, dict(tools=tools, chat_template_kwargs=dict(tool_call_format="json"))),
        ("bad-json-arguments", base + [{"role": "assistant", "content": None, "tool_calls": [bad_call]}], dict(tools=tools)),
        ("text-parts", [{"role": "user", "content": ["first",
                                                       {"type": "text", "text": "second"}]}], {}),
    ]]


def output_cases():
    call = "<tool_call>\n<function=bash>\n<parameter=command>\n  echo hi  \n</parameter>\n" + \
           "<parameter=timeout>2.5</parameter><parameter=count>3</parameter>" + \
           "<parameter=enabled>true</parameter><parameter=options>{\"a\": [1, false]}</parameter>" + \
           "<parameter=maybe>null</parameter></function>\n</tool_call>"
    return [dict(name=name, raw=raw, tools=tools) for name, raw, tools in [
        ("unicode", "计划💡。\n</think>\n\n答复。", []),
        ("plain", "Plan.\n</think>\n\n  Answer.  \n", []),
        ("no-close", "<think>\nPlan.\n", []),
        ("no-close-call", "Plan.\n" + call, [tool()]),
        ("typed", "<think>\nPlan.\n</think>\nChecking.\n" + call, [tool()]),
        ("multiple", "Plan.</think>\n" + call + "\n" + call, [tool()]),
        ("no-tools", "Plan.</think>" + call, []),
        ("unclosed-call", "Plan.</think>" + call.removesuffix("</tool_call>"), [tool()]),
        ("trailing-prose", "Plan.</think>before\n" + call + "\nafter", [tool()]),
        ("reasoning-spaces", "<think>\n  plan  \n</think>answer", []),
        ("typed-edge", "Plan.</think><tool_call><function=bash><parameter=command>null</parameter>"
         "<parameter=count>0007</parameter><parameter=enabled>not true</parameter>"
         "<parameter=options>{'a': True}</parameter></function></tool_call>", [tool()]),
        ("truncated-params", "Plan.</think><tool_call><function=bash><parameter=command>  echo hi  "
         "<parameter=timeout>3.5", [tool()]),
        ("unknown-function", "Plan.</think><tool_call><function=other><parameter=x>3</parameter>"
         "</function></tool_call>", [tool()]),
        ("malformed-first", "Plan.</think>before<tool_call>bad</tool_call>" + call, [tool()]),
        ("no-function", "Plan.</think><tool_call>bad</tool_call>", [tool()]),
    ]]


def generate(model, pod):
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(model, local_files_only=True)
    scope = pod_functions(pod)
    prompts = prompt_cases()
    for case in prompts:
        request = case["request"]
        messages = scope["prepare"](request["messages"])
        kwargs = request.get("chat_template_kwargs", {})
        case["prompt"] = hf.apply_chat_template(messages, tools=request.get("tools"),
                                                add_generation_prompt=True, tokenize=False, **kwargs)
        case["ids"] = hf.apply_chat_template(messages, tools=request.get("tools"),
                                             add_generation_prompt=True, tokenize=True, return_dict=False, **kwargs)
        assert case["ids"] == hf(case["prompt"], add_special_tokens=False)["input_ids"]
        assert case["ids"][0] == 166100 and case["ids"][1] != 166100
    outputs = output_cases()
    for case in outputs:
        reasoning, content, calls = scope["split_output"](case["raw"], case["tools"])
        case["expected"] = dict(reasoning=reasoning, content=content, calls=[
            dict(name=c["function"]["name"], arguments=json.loads(c["function"]["arguments"])) for c in calls])
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, cases in [("prompts", prompts), ("outputs", outputs)]:
        (FIXTURES / f"{name}.json").write_text("[\n" + ",\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n]" + "\n")
    import importlib.metadata
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in [
        ("pod", pod), ("tokenizer", model / "tokenizer.json"), ("tokenizer_config", model / "tokenizer_config.json")]}
    hashes["versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers", "jinja2")}
    (FIXTURES / "sources.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"HF prompts={len(prompts)} tokens={sum(len(c['ids']) for c in prompts)} pod outputs={len(outputs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("pod", type=Path)
    ap.add_argument("--zig", type=Path)
    ap.add_argument("--results", type=Path)
    args = ap.parse_args()
    generate(args.model, args.pod)
    if args.zig:
        import os
        env = dict(os.environ, TF_LLAMA_MODEL=str(args.model), ZIG_GLOBAL_CACHE_DIR=str(args.results / "global-cache"))
        subprocess.run([str(args.zig), "build", "test-llama-server-host", "-j4",
                        "--cache-dir", str(args.results / "host-cache"),
                        "--summary", "all"],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
