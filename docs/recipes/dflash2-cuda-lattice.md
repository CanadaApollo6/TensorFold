# DFlash2 CUDA compact lattice

The Llama CUDA DFlash2 runtime reduces the BF16 head logits on the GPU before the host conditional selector.
The target head and sampling arithmetic are unchanged.
Qwen's separate CUDA and Metal drafter implementations do not call this runtime.

## Candidate contract

The input is the same BF16-narrowed head output used by the former CPU lattice.
The order is descending value, then ascending token ID, including equal signed zeros.
Any NaN or either infinity anywhere in a row returns `DraftLatticeFinite`.

The first launch reuses `tf_topk_slices` from `sample.cu` and `logprob_rows.cuh`.
Each 256-thread block selects up to 16 entries from a 4096-token slice.
The second launch uses one 256-thread block per row and reuses `topk_best` to merge those candidates.
It then computes a max-shifted FP64 exponential sum in 256 strided chains and a fixed binary reduction tree.
The subtraction before each exponential is FP32, matching the previous CPU expression, including overflow to -infinity.
Compilation disables contraction and flush-to-zero through the existing sample image flags.

Each downloaded row is 144 bytes, with 16 uint32 IDs, 16 FP32 values, an FP64 log of shifted mass, and two uint32 words.
One word rejects nonfinite input; the other is padding initialized to zero.
Seven rows transfer 1008 bytes instead of 2336768 bytes of logits.
The separate seven-by-rank BF16 selector projection still transfers 3584 bytes at rank 256.
The slice scratch and compact result allocations participate in the existing pool and memory admission plan.

## Normalizer exactness

The old normalizer is `maximum + f32(log(sum))`, with the final addition in FP32.
Subtracting this rounded value from each retained logit can change selector ties.
A generic FP32 reduction, or removing the row constant algebraically, cannot preserve that contract.

The new path bounds the difference between the GPU log of shifted mass and the old CPU result by
`e = 8 * (vocab + 1024) * epsilon64` in absolute log units.
For 166912 entries, e is about 2.983e-10. For 248320 entries, e is about 4.429e-10.
This is a rounding certificate, not a tolerance applied to selector decisions.

All terms are nonnegative and at least one is 1, so the exact sum is in [1, vocab].
The CPU has at most vocab sequential sum roundings, and the GPU has at most ceil(vocab / 256) + 8.
For vocab within uint32, their combined gamma bounds are below `2 * vocab * epsilon64`.
FP64 exp and log errors, log conditioning, bound calculation rounding, and underflowed terms fit in the remaining margin.
The supported Linux host uses FP64 libm exp/log; the bound allows several ulps per transcendental.
CUDA's double exp/log accuracy is documented in the
[NVIDIA floating point appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html).
No fast math intrinsic is used.

The host applies the old FP32 cast and addition to both `log_mass - e` and `log_mass + e`.
If their bits agree, monotonic rounding proves that the old normalizer has those same bits.
The resulting unary values and therefore every conditional selector score and chain are bit-identical.
If the endpoint bits differ, only that row is downloaded and the old CPU lattice calculation supplies its normalizer.
Invalid rows fail before selection. A nonfinite reduction result also requests fallback.
The fallback has no probability or tie tolerance and is visible in `lattice_fallback` profiling.

## Sampling and profiling

The lane engine samples the target at absolute positions and accepts drafts by target-token equality.
`llama/backend.zig` verifies through the target sampler, and `lanes/engine.zig` calls `acceptPath`.
It does not perform a probability-ratio acceptance test using DFlash2 unary values or tree scores.
There are no draft probabilities passed into acceptance that could disagree with proposal probabilities.
The cached-logit sampling oracle and coupled-seed model gates remain required.

`TF_LLAMA_DRAFT_PROFILE` reports the two launches together as `lattice_gpu` with a CUDA event span.
`download` includes compact rows and projected hidden rows, plus stream completion wait.
`topk` and `logsumexp` report zero host calls and time; any exceptional old-row calculation is `lattice_fallback`.
Do not add stream spans to host wall times because waits overlap device execution.

## Verification

`zig build test-dflash2-host` exercises the old scalar lattice against a host emulation of the GPU layout and sum tree.
It covers real vocabulary widths, slice tails, ties, signed zero, near-boundary BF16 values, extreme rows and nonfinite input.
The rounded normalizer must match bit for bit, with an explicit ambiguous-rounding fallback case.

`tf-llama-test dflash2-lattice` compares real-shaped device outputs with the old scalar lattice.
It checks six widths, seven row kinds, five byte-identical repeats, exact IDs and values, bounded log mass,
exact certified normalizers and NaN/inf rejection.
Compile-only checks are not device qualification.
