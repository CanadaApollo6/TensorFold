# Llama chunked prefill

This change targets Nanbeige4.3-33B EXL3 on the local `nanbeige` branch, starting at `98884be08`.
All verification in this receipt is CPU or compilation only. No GPU was used.

## Execution contract

`TF_LLAMA_PREFILL_ROWS` accepts 1 through 1024 rows and defaults to 128.
The target allocates bounded row buffers once. EXL3 prompt projections use its existing 128-row blocks.
Embeddings upload from pinned staging as a chunk, followed by row RMS norms, QKV, RoPE, cache writes,
attention, output projection, MLP and final norm. Only the final prompt row passes through the LM head.
The existing decode and speculative verification paths retain their row arithmetic.

BF16 attention uses `prefill_attention` with head dimension 128, eight warps, one head per block and
128 query rows per block. Its causal mask admits keys through each query's absolute position.
The cache contains the retained prefix plus the new chunk. GQA seven maps 56 query heads to eight KV heads.
The new kernel variant needs 65,536 dynamic shared bytes.

KV4 packs all new K/V rows at their absolute cache offsets using the existing H32 midpoint codec.
Attention runs eight query rows per wave with the serial codec's existing partial/merge arithmetic.
Each query uses quantized preceding keys, the retained BF16 BOS and its own unquantized current K/V.
A wave's causal lengths differ by row. Unused partial groups have zero weight.
Scratch is bounded by eight serial attention workspaces rather than one workspace per prompt row.
This vector attention path is a performance risk relative to tensor-core BF16 attention.

Taps after target layers `[4,15,26,38,49]` use the same BF16 conversion and row-major five-plane layout
as serial capture. Only absolute positions in the drafter's final context window are ingested.
Both drafter formats use the shared DFlash2 runtime. A bounded tap ring lets a continued prompt rebuild
the drafter context without rerunning the retained target prefix.
When a rewritten or shortened prompt needs taps older than that ring, reuse stops before those taps.
The final prompt token is excluded from reuse so its logits are refreshed.

The adapter retains token IDs for the valid cache prefix, including consumed decode/verification rows.
A new stream compares its rendered prompt with those IDs, sets `s.cached` and starts at the first new row.
Rejected verification suffixes are truncated. Cancelled chunks retain only completed rows.
Each prompt step synchronizes one chunk, publishes `fill_at`, and yields to the shared scheduler.
Sampling, penalties and stop handling remain in their existing code paths.

## Capacity projections

These CPU planner results use 24 GiB total, a 2 GiB reserve, and a 512 MiB driver allowance.
They are allocation estimates, not measured physical fits.

| Drafter | BF16 target KV | KV4 target KV |
| --- | ---: | ---: |
| None, supplied previous ceiling | 40,700 | 144,424 |
| None, chunk 128 | 40,438 | 139,784 |
| BF16, chunk 128 | 21,171 | 73,293 |
| Affine-4, chunk 128 | 34,279 | 118,535 |

The no-drafter changes are -262 and -4,640 tokens.
Runtime planning includes prompt rows, eight logits rows and eight KV4 attention workspaces.
Draft planning additionally includes chunk captures and the retained context-window tap ring.
The target's configured room callback and physical 2 GiB reserve also cover these allocations.

CPU-only planner commands use `tf-llama-test llama-capacity MODEL 24576 20000` and
`tf-llama-test llama-draft-capacity MODEL DRAFT 24576 512 20000`.
Both commands return before opening the CUDA driver.

## Deferred GPU checks

`tf-llama-test llama-prefill MODEL PROMPTS OUT DRAFT_OR_none serial|batch|resume|timing`
loads all 52 layers. Equivalence dumps every populated K/V cache row, final logits, BF16 BOS and retained taps.
Resume first fills half the history, releases the stream and continues with the complete prompt.
Timing performs one warmup and averages three synchronized cold prefills, excluding loading and downloads.

`llama_prefill_check.py` checks final-logit KL <= 0.001 and identical argmax, BF16 cache/tap relative L2 <= 0.01,
and decoded KV4 relative L2 <= 0.05. It also reports changed cache bytes.
KV4 comparisons use rotated values, preserving L2 because H32 is orthonormal.
These are declared numerical gates, not a claim of bit-identical BF16 flash attention.
P1 compares the final position of every existing exllamav3 reference sequence on the prompt path,
using its existing KL <= 0.01 threshold.

The full plan has 45 jobs, including these additions.

- `p1-prefill` checks existing P1 references through batched prefill.
- `prefill-{2000,20000}-{none,bf16,affine4}-kv{16,4}` compares serial, batched and resumed execution.
- `prefill-speed-{2000,20000,60000}-kv{16,4}` measures TensorFold and exllamav3 on the same card.

Timing uses exllamav3 cached `flash_attn`, 2,048-row chunks and matching 16-bit or 4-bit KV.
Timing receipts record tok/s, times, repetitions, ratio and a separate speed verdict against the 2x target.
A missed speed target is retained in the report. It does not discard the numerical receipts.
All child model commands go through `llama_gpu_run.py`.

A fast paid subset is available through the existing container entry point.

```bash
ENGINE_COMMIT=<commit> OUT_BUCKET=<bucket> SM=80 JOB_SECONDS=12600 \
  bash tools/zig/llama_hf_job.sh --fast
```

The container also needs its existing pinned input mount, image digest and output mount.
`--fast` selects `p1-*,prefill-2000-*,prefill-speed-*,p2,p3-*`.
It excludes the 20K serial matrix and P4 long jobs. It retains the P3 format/KV matrix.
`--only` supports custom subsets and `--skip` further narrows the fast set.
The selector automatically includes `p1-reference` for `p1-prefill`.
A subset remains `PARTIAL`, so it cannot claim whole-gate qualification.

P4 planning now assumes 2,000 projection tokens/s plus a quadratic attention term per KV format.
A check pair costs `75 + 2 * (N/2000 + N*N/60000000) + 2294/45` seconds before live calibration.
That gives 306 seconds at 60K and 559 seconds at 100K, replacing 526 and 1,237 seconds.
P4 timeouts are three times the current per-class projection, floored at 600 seconds and capped by
the caller's timeout and remaining paid-job deadline. Initial limits are 918 and 1,678 seconds.
The container's default per-job cap is 1,800 seconds, overridable with `GATE_JOB_TIMEOUT`.
These are unmeasured planning assumptions. Per-class wall-time calibration retains its 25% reserve.
The 20K P4 jobs run before the long P4 jobs to calibrate that class first.
The full 45-job matrix may need multiple paid subsets under the existing 210-minute container cap.

## Risks awaiting a rented card

Numerical equality, first-token equality and the 2x speed target are NOT RUN.
Flash attention changes the reduction order. KV4 code boundaries can amplify small upstream differences.
Eight-row KV4 waves preserve arithmetic but may remain bandwidth or launch limited at long contexts.
SM80/86/89 compilation does not qualify the new symbols' live resource usage or numerical behavior.
The full-model gate must establish those receipts on the intended hardware.
Long serial baselines and cache dumps can dominate the paid equivalence jobs.
Context-window tap reconstruction must be checked through both drafter formats and rewritten prefixes.
The local ATC executable was absent. Work proceeded under the explicit local implementation assignment.

## CPU and build verification

Run from the engine root with Zig 0.17.0. `ZIG` below is
`$HOME/.local/share/zig-x86_64-linux-0.17.0/zig`. `MODEL` is the supplied 3.5-bpw model directory.
`BASE` is the supplied `tensorfold-p2-fix1/lean-base.txt`.
`TMPDIR` was set to `$HOME/.cache/llama-prefill-tmp` for compilation.

```bash
"$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
  -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j8 --summary all
"$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
  -Dnvcc=/opt/cuda/bin/nvcc -Dsm=80 -j8 --summary all
"$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
  -Dnvcc=/opt/cuda/bin/nvcc -Dsm=89 -j8 --summary all
TF_LLAMA_MODEL="$MODEL" "$ZIG" build test-llama-server-host test-server-cpu \
  test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host \
  test-llama-host test-dflash2-host test-qwen27-speed-host test-qwen27-plans-host \
  test-qwen27-equality-host -j4 --summary all
"$ZIG" build test-cuda-host -j4 --summary all
python3 -m unittest discover -s tools/zig -p 'test_llama*.py'
python3 tools/zig/llama_full_gate.py --dry-run --out /tmp/llama-prefill-dry
python3 tools/zig/lean_check.py > /tmp/h.txt
python3 tools/zig/lean_new.py "$BASE" /tmp/h.txt
bash -n tools/zig/llama_hf_job.sh
```

The standalone `test-qwen27` step is unavailable in this Linux build graph.
The available Qwen27 speed, plans and equality host targets were run instead.
The Llama tests include recorded KV4 writes at position 127 across 8/8/1 query waves,
chunk progress and retained-prefix continuation, drafter context-window seams and budget boundaries.
Python receipt controls reject changed cache positions, altered taps and changed final logits.
The orchestration dry-run exercises a real child-process crash, durable resume and changed-input refusal.

The requested host command passed all 38 build steps. Its final run executed 156 tests successfully;
other unchanged test artifacts were cached. The standalone Llama run passed 70 tests.
Python passed 23 tests. The dry-run passed all 45 CPU stub jobs and resumed 43 after its planted crash.
Lean reported 426 inherited findings and zero new findings. Changed files stay within 600 lines.
The CUDA host suite reported only the named inherited `exl3_load` failure, `NondeterministicMemoryUsage`.
Its earlier run executed 1,958 passing tests with eight skips; the final rerun executed 662 passing tests
with one skip and reused unchanged artifacts. Neither run added another failure.
SM80, SM86 and SM89 each compiled all 130 build steps successfully.
