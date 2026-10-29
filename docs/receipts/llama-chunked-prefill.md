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
Resume fills a prefix overlapping the final tap window, releases the stream and continues with the complete prompt.
For 20K tokens the prefix is 19K, so 1,047 retained tap rows must be restored.
Drafter dumps include every valid reconstructed K/V row in layer, K/V-plane and absolute-position order,
plus exact context bounds. Source taps alone do not establish correct cache reconstruction.
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
Live context-window tap reconstruction remains NOT RUN through both drafter formats and rewritten prefixes.
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


## Review corrections after 5a5044d9

The rejected implementation treated `history_end - window` as the oldest initialized tap.
Prefill only captures the final requested tail, so cancellation can leave a much shorter interval.
Moving that endpoint backwards during restore also invented validity for older slots, including overwritten slots.
The target retained the right token prefix while the drafter could ingest unrelated or uninitialized data.

`tap_history.zig` now owns an optional nonempty interval of initialized absolute tap positions.
A successful ingest appends one position and evicts positions overwritten by ring wrap.
Restore intersects that interval with the retained common prefix and never moves its lower bound backwards.
A discontinuous append starts a new interval. An uncovered restore fails before reading any slot.

Reuse requires complete coverage of `[requested_tail_start, common_prefix_end)`.
When coverage is missing, target reuse ends at the earlier of the common prefix and requested tail start.
The normal chunked target path then rebuilds the missing tail with the retained earlier target KV.
This conservative policy can replay at most 2,047 already cached target rows for this checkpoint.
It needs no new device buffers or special target replay path. Valid overlapping continuation keeps full reuse.
The extra target work is unmeasured. No GPU was used for these corrections.

The production attachment is parameterized by target and runtime types for host tests.
Tests execute the real capture, ingest, reuse, restore and backend lifecycle methods.
Recording host dependencies check each tap's absolute position and token identity and every live reconstructed row.
They never open the CUDA driver. The tests cover:

- Fresh 20K prefill shortened by one token. The old policy reads the uncaptured row at 17,952.
- Rewritten 20K prompt cancelled after one chunk at 10K, then shortened to 11K.
  The old restore endpoint manufactures coverage starting at 8,953.
- Initial 20K fill cancelled at 18,048, then submitted as a 19K prefix.
  The old policy restores 1,000 uncaptured rows starting at 16,953.
- Cancellation followed by an 18,047-token prompt. The old policy invents valid rows starting at 16,000.
- An unrelated prompt sharing 32 tokens, cancelled and shortened. Old slots retain another prompt's identity.
- Eight cycles of wrap, rewrite, cancellation and truncation. The old policy eventually reads invalid older slots.
- A 19K to 20K overlapping continuation and cancellation followed by extension.
  These are positive controls for preserving valid reuse, and were already valid under the old implementation.
- Direct uncovered restore. The new guard refuses it before the recording runtime sees a row.

The full-model equivalence jobs now compare serial and batch, then compare five lifecycle cases with cold batch.
The cases are `resume`, `shorten`, `cancel-shorten`, `cancel-extend` and `rewrite-cancel`.
Both drafter formats and both target KV formats run these cases at 2K and 20K.
No-drafter jobs retain one overlapping resume case. The top-level plan remains 45 jobs.
The new lifecycle cases add unmeasured runtime and disk cost to each drafted equivalence job.
Existing deadline admission and per-class calibration still bound paid runs.

`draft-kv.bf16` is required and compared with relative L2 <= 0.01, including when target KV is four-bit.
`draft-context.json` must match exactly. Missing or asymmetric artifacts fail the comparison.
Python controls corrupt only the reconstructed ring while keeping source taps and target logits identical.
Both numeric comparator implementations reject that corruption, as does the complete equivalence orchestration.
The controls also reject missing drafter evidence and changed context bounds.

All GPU/model numerical comparisons, proposal quality, device resource behavior and performance remain NOT RUN.
Compilation and recording host dependencies do not qualify the live drafter runtime.


## Correction validation commands

Run these from the engine root. `ZIG`, `MODEL` and `BASE` have the meanings in the earlier command block.
Set `TMPDIR` to an owned directory under `$HOME/.cache` if the system temporary quota is full.

```bash
"$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
  -Dnvcc=/opt/cuda/bin/nvcc -Dsm=86 -j8 --summary all
"$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
  -Dnvcc=/opt/cuda/bin/nvcc -Dsm=80 -j8 --summary all
TF_LLAMA_MODEL="$MODEL" "$ZIG" build test-llama-server-host test-server-cpu \
  test-tool-parse-host test-server-recovery-host test-tool-visibility-http-host \
  test-dflash2-host test-llama-host -j4 --summary all
"$ZIG" build test-llama-host -j4 --summary all
"$ZIG" build test-cuda-host -j4 --summary all
python3 -m unittest discover -s tools/zig -p 'test_llama*.py'
python3 tools/zig/llama_full_gate.py --dry-run --out "$TMPDIR/dry-run"
python3 tools/zig/lean_check.py > /tmp/h.txt
python3 tools/zig/lean_new.py "$BASE" /tmp/h.txt
git diff --check
```

The initial SM86 attempt exposed a `usize` to `u32` assignment in the new gate helper.
It was corrected with an explicit token-width cast before the final build runs.


SM86 and SM80 each passed all 130 build steps. The combined requested host command passed all 28 steps.
That command executed 59 passing tests and reused the other unchanged artifacts, including DFlash2 and Llama.
The standalone corrected Llama artifact passed all 79 tests, including nine new tap-history tests.
System Python passed all 25 tests using the standard-library comparator.
The existing project virtual environment also passed all 25 tests with NumPy 2.5.3, exercising both comparators.
The supplemental command was `"$VENV_PYTHON" -m unittest discover -s tools/zig -p 'test_llama*.py'`.
The CPU dry-run passed all 45 stub jobs, resumed 43 after exit 73,
and recorded zero GPU calls. Lean reported 426 inherited findings and zero new findings.
The broader CUDA host suite had 662 passing tests, one skip and only the permitted `exl3_load` failure,
`NondeterministicMemoryUsage` from `checkAllAllocationFailures`. It passed 268 of 270 build steps.
The failed run step and its dependent aggregate step account for the two incomplete steps.

A temporary production-method mutation reinstated the original ingest endpoint and reuse/restore policy from 5a5044d9.
It compiled, then failed seven of the 79 host tests. Six detected invalid tap rows or prompt identities.
The seventh detected the missing uncovered-restore guard. Both positive reuse controls still passed.
Restoring the corrected source passed all 79 tests again. The independent reviewer's extracted reproducer
also confirmed its three expected invalid restores, but the permanent regressions use production methods.

Zig formatting and whitespace checks passed. Every changed file remains below 600 lines.
No source changes were made under the protected job directories, and no GPU command was executed.
