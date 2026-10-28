# Nanbeige pod compatibility

Nanbeige chat uses the scored EXL3 pod's request and response contract. Its parsers do not use the shared Qwen, LFM2, DSML, or tool-parameter decoders. Other families keep their existing paths.

`nanbeige_output.zig` owns `split_output`, the three XML regex scans, schema lookup, and `typed`. `nanbeige_numbers.zig` implements Python numeric conversion. `nanbeige_literal.zig` parses Python literal values. `nanbeige_literal_value.zig` retains their types through validation and serialization. `nanbeige_failure.zig` owns the explicit `PodFailure` error and output resource guards. `nanbeige_response.zig` owns ordered response envelopes, UUID4 IDs, and JSON and SSE encoding. `nanbeige_request.zig` owns request defaults, context limits, and error envelopes. `nanbeige.zig` retains message preparation and the streaming reasoning holdback.

The parser preserves the pod's first `</think>` partition, case-sensitive names, unclosed spans, final-newline `$` behavior, and parameter newline removal. Tool arguments use `json.dumps(args, ensure_ascii=False)`, including Python separators, float text, nonfinite numbers, exact integers, and dictionary insertion order. Streaming reasoning retains its raw `<think>` prefix and newlines, while whole responses normalize them.

The pod ignores `parallel_tool_calls`, `tool_choice`, stop strings, penalties, top-level thinking controls, logprobs, and structured-output settings. Nanbeige chat ignores those fields too. A null sampler field uses the pod default. Zero or absent completion limits use the remaining context. The requested model string is echoed even when it is not a configured alias. Assistant argument strings decode before template rendering, including valid JSON scalars and arrays.

## Regenerate and check

Run from the engine root with the pod's Python environment and tokenizer directory. The generator only loads the tokenizer.

```sh
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_reference.py MODEL_DIR SERVE_EXL3_PY
TF_LLAMA_MODEL=MODEL_DIR CUDA_VISIBLE_DEVICES='' zig build test-llama-server-host -j4 --summary all
```

The generator extracts the pod's functions and `chat` coroutine through AST. It labels the existing prompt and tag-soup fixtures, a seeded model-like grammar, typed schemas, escape repros, prepared messages, and whole HTTP bodies. Response tests drive the production HTTP writer with a scripted CPU engine and the real tokenizer. They compare exact bytes after replacing only IDs and creation times. SSE labels receive the same token-text chunks as the engine; different generation chunk boundaries can produce different reasoning frames.

`nanbeige_unicode.zig` resolves literal names using a generated Unicode 15.0 database. The request allocator inflates the embedded gzip table on its first explicit-name lookup. CJK unified, Khitan, and Nushu names use generated ranges. Hangul uses syllable composition. Python 3.12 accepts lowercase ordinary names and aliases, Khitan, Nushu, and CJK compatibility names. CJK unified names and Hangul require Python's case-sensitive spelling. Tangut names and named sequences are rejected by that interpreter.

Regenerate the table and check every named code point and alias in both cases against Python 3.12 and Zig:

```sh
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_unicode.py --zig /path/to/zig
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_unicode.py --check --zig /path/to/zig
```

The generator pins the Unicode version and hashes of the official NameAliases.txt and NamedSequences.txt inputs. `--ucd-dir DIR` uses local copies. The gzip table stores a little-endian count, sorted-name offsets, and records containing a three-byte code point and a NUL-terminated ASCII name. Its adjacent license covers the Unicode data. Source NUL bytes invalidate only the Python-literal fallback. JSON and literal escapes that decode to NUL remain valid.

## Deliberate limits

Output typing and serialization propagate `PodFailure`. Bytes, sets, complex values, `...`, unsupported dictionary keys, unhashable literal keys and set members, invalid output schemas, integer-to-complex overflow, recursion overflow, and literal parser stack exhaustion produce the pod's plain HTTP 500 body. Whole replies use `text/plain; charset=utf-8` and `Internal Server Error`. A JSON or literal decimal conversion exceeding 4300 digits falls back to the raw string. A base-prefixed literal can parse successfully and then exceed the decimal serialization limit, which fails instead. `frozenset()` and the name `Ellipsis` are rejected by `literal_eval` and remain raw strings.

An unserializable parameter stays pending until duplicate parameter names have settled. Literal evaluation errors fail immediately. Serialization validates the complete output before any answer or tool frames. UTF-8 validation happens on each encoded response or frame, so a surrogate can leave earlier content and tool frames on the wire. No invalid UTF-8 is sent. SSE uses HTTP/1.1 chunked framing. Normal completion includes `[DONE]` and the zero chunk. An output failure closes without either, matching uvicorn's incomplete-chunked-response behavior. Reasoning frames already sent remain intact.

Regenerate the failure fixtures through the actual FastAPI app, with its generator replaced by CPU token text:

```sh
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_failures.py MODEL_DIR SERVE_EXL3_PY
CUDA_VISIBLE_DEVICES='' python tools/zig/llama_serve_failure_transport.py MODEL_DIR SERVE_EXL3_PY zig/src/server/fixtures/llama/failures.json TRANSPORT_JSON
```

The second command requires uvicorn and briefly opens a loopback listener. It captures the real bytes and EOF for whole replies, healthy streams, and aborted streams. The first requires FastAPI and Starlette, rather than a substitute response encoder. Fixture metadata records the Python version, dependency versions, and pod source hash. Six additional fixtures inject `MemoryError` into the actual pod app at JSON loading, literal evaluation, argument serialization, role and reasoning chunk serialization, and whole-response rendering. The wire tests use a zero-capacity allocator to verify those allocation boundaries without exhausting host memory. A seventh fixture raises a generator `RuntimeError`. A producer failure emits an error frame, ends the HTTP body, and omits `[DONE]`. An output failure outside that producer aborts the HTTP body.

Resource guards match the supplied Python 3.12.14 interpreter's observed app behavior. C recursion and parser-stack limits depend on the Python build and active call stack. JSON decoding and argument serialization have separate whole and stream limits. The literal guard checks long unary, binary, power, conditional, and lambda chains outside strings and comments. It does not implement every CPython parser resource counter, such as every combination of deeply nested operators and invalid grammar. Exact thresholds for other builds and genuine process-wide memory exhaustion remain a documented boundary. Allocation failure during output returns 500 or aborts SSE when enough memory remains to write the fixed error header. Process termination, an OS OOM kill, and socket reset details cannot be mirrored across runtimes.

The Unicode table remains request-owned. Its allocation and inflation strategy is unchanged.

- Malformed history arguments in the template's XML mode return the pod's plain 500 before SSE opens. JSON tool-call mode retains its accepted scalar arguments.
- Malformed request types, invalid sampler ranges, nonpositive completion counts, and engine admission or runtime failures follow native validation and failure handling. They do not reproduce every exception raised by HF, FastAPI, or EXL3. The engine uses its admitted context window, which may be smaller than the pod's 262144 tokens.
- This is CPU parser, request, and wire-format evidence. It does not establish GPU generation equivalence or TB2 scores.
