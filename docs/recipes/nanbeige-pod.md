# Nanbeige pod compatibility

Nanbeige chat uses the scored EXL3 pod's request and response contract. Its parsers do not use the shared Qwen, LFM2, DSML, or tool-parameter decoders. Other families keep their existing paths.

`nanbeige_output.zig` owns `split_output`, the three XML regex scans, schema lookup, and `typed`. `nanbeige_numbers.zig` implements Python numeric conversion. `nanbeige_literal.zig` implements the JSON-compatible subset of `ast.literal_eval`. `nanbeige_response.zig` owns ordered response envelopes, UUID4 IDs, and JSON and SSE encoding. `nanbeige_request.zig` owns request defaults, context limits, and error envelopes. `nanbeige.zig` retains message preparation and the streaming reasoning holdback.

The parser preserves the pod's first `</think>` partition, case-sensitive names, unclosed spans, final-newline `$` behavior, and parameter newline removal. Tool arguments use `json.dumps(args, ensure_ascii=False)`, including Python separators, float text, nonfinite numbers, exact integers, and dictionary insertion order. Streaming reasoning retains its raw `<think>` prefix and newlines, while whole responses normalize them.

The pod ignores `parallel_tool_calls`, `tool_choice`, stop strings, penalties, top-level thinking controls, logprobs, and structured-output settings. Nanbeige chat ignores those fields too. A null sampler field uses the pod default. Zero or absent completion limits use the remaining context. The requested model string is echoed even when it is not a configured alias. Assistant argument strings decode before template rendering, including valid JSON scalars and arrays.

## Regenerate and check

Run from the engine root with the pod's Python environment and tokenizer directory. The generator only loads the tokenizer.

```sh
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_reference.py MODEL_DIR SERVE_EXL3_PY
TF_LLAMA_MODEL=MODEL_DIR CUDA_VISIBLE_DEVICES='' zig build test-llama-server-host -j4 --summary all
```

The generator extracts the pod's functions and `chat` coroutine through AST. It labels the existing prompt and tag-soup fixtures, a seeded model-like grammar, typed schemas, prepared messages, and whole HTTP bodies. Response tests drive the production HTTP writer with a scripted CPU engine and the real tokenizer. They compare exact bytes after replacing only IDs and creation times. SSE labels receive the same token-text chunks as the engine; different generation chunk boundaries can produce different reasoning frames.

## Deliberate limits

- Sets, bytes, complex numbers, Ellipsis, and unsupported dictionary keys return the raw parameter string. The pod can fail with an uncaught `TypeError` while serializing those values. Raw text keeps the tool response usable.
- Named Unicode escapes such as `\N{LATIN CAPITAL LETTER A}` return the raw parameter string. The small literal parser has no Unicode-name database. Hexadecimal, Unicode code-point, octal, raw, and ordinary string escapes are supported.
- Malformed request types, invalid sampler ranges, nonpositive completion counts, and engine admission or runtime failures follow native validation and failure handling. They do not reproduce every exception raised by HF, FastAPI, or EXL3. The engine uses its admitted context window, which may be smaller than the pod's 262144 tokens.
- This is CPU parser, request, and wire-format evidence. It does not establish GPU generation equivalence or TB2 scores.
