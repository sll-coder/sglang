# Supplementary Results: Fused INT8 Mamba Checkpoint Codec

This supplement accompanies the concise PR description. It preserves the detailed results from the full English draft, including unfavorable results and validation limitations.

Performance measurements use v0.5.22 `baa9befa381511cd4d56f83980951e5a45fffc6d` plus the codec change. Main correctness/regression validation uses `a8e1539ea6c2fe563854e87e40602c8e64813418`. The historical results below are preserved; the following refresh adds local correctness/regression runs on November 3, 2026. Performance measurements were not repeated.

## Refreshed main validation — November 3, 2026

The codec patch was applied without conflicts to main `62f9f3f785f889822ab4e78d65de9997d0b58441`. The patch still contains eight files, 778 insertions and one deletion; the independent sampling/JIT fix is excluded. The fused codec and integration logic are unchanged from the previously validated port.

| Local check | Refreshed result |
|---|---|
| Codec, fallback, commit, existing INT8/KDA and indexing regressions | 98 passed, including all 75 new tests |
| Related Mamba/cache regressions | 163 passed, 773 subtests passed |
| Environment handling and exports | 21 passed, 2655 subtests passed |
| Native Rust TreeCore, Mamba filter | 48 passed, 534 deselected |
| Expanded `unit/kernels` suite | 247 passed, 1 skipped, 2 subtests passed |
| Official targeted CPU CI runner | 1/1 file passed; 2 tests |
| Official targeted GPU CI runner, extra-a | 2/2 files passed; 73 tests |
| Applicable pre-commit hooks on all eight changed files | Passed |
| TP2 GDN service, eager versus fused | 24/24 output-token and cached-token pairs match; actual dispatch audited on both ranks |

CI-runner executions overlap with direct pytest runs and are not additional independent test counts.

**Sampling-test limitation:** The expanded kernel suite passes in aggregate, but running `TestCakeSoftmaxCapability` in isolation gives **3 failed, 2 passed** on both the patched tree and a clean worktree at the same refreshed main commit. These are the same missing-Tensor-method test-double failures previously identified. The aggregate outcome is therefore sensitive to test order and must not be described as fixing the sampling tests. No sampling production code or test fixtures were changed in this codec patch.

The real-model comparison uses the same fixed GDN configuration and 24 query pairs as the historical main smoke test, after independent eager/fused service starts. This is a correctness refresh, not a new speed benchmark. Full remote CI, other GPU platforms, and a full fresh environment installation remain outside the local validation scope. Local dependency versions and test counts accompany this supplement in `validation.json`.

## Historical experiment details

## Contents

- [Accuracy and integration coverage](#accuracy-tests)
- [Main regressions and local CI](#main-regressions-and-local-ci)
- [Benchmark setup and version scope](#setup-and-version-scope)
- [Temporary allocations and codec timings](#temporary-memory-and-isolated-kernel-time)
- [Service memory footprint](#service-memory-footprint)
- [C=8 throughput](#c8-end-to-end-throughput)
- [Long-window KDA measurements](#kda-c8-longer-timing-windows)
- [Whole-service Nsight findings](#whole-service-nsight-findings)

## Accuracy Tests

### Kernel and integration coverage

Validated on main **`a8e1539ea6c2fe563854e87e40602c8e64813418`**, pinned on November 2, 2026. The kernel and launcher arithmetic is unchanged from the performance snapshot, as checked by AST comparison.

**75 new tests passed:**

| Coverage | Result |
|---|---:|
| Codec numerics, layouts, indices, model geometry, CUDA Graph replay, large offsets, and CUDA fallback | 65 passed |
| Checkpoint-commit indexing and save/restore integration | 8 passed |
| CPU fallback and switch behavior | 2 passed |

Tests compare quantized data, scales, and restored states against the eager INT8 implementation with `rtol=0, atol=0, equal_nan=True`, and check that untouched slots remain unchanged. Coverage includes empty/single/multiple slots, FP16/BF16/FP32, mixed scale dtypes, noncontiguous layouts, rounding and exceptional values, offsets beyond 2 GiB, graph replay with updated indices, and unsupported CUDA inputs: value dimension 257 and FP64 states.

### Model-output comparisons

Performance/model experiments use the frozen v0.5.22 snapshot described below.

| Experiment | Matching output-token sequences |
|---|---:|
| Original low-concurrency GDN, prefill graph off | 880/880 pairs |
| Original low-concurrency GDN, existing breakable prefill graph on | 879/880 pairs |
| GDN controls with execution path and batching fixed | 12/12 pairs |
| Original low-concurrency KDA AWQ | 880/880 pairs |
| Capacity-8 GDN setup, prefill graph off | 1413/1760 pairs |
| Capacity-8 GDN setup, existing prefill graph on | 1577/1760 pairs |
| Capacity-8 KDA setup | 1760/1760 pairs |
| KDA C=8, 65-second windows | 39917/39917 common input-index pairs |
| GDN service regression on the pinned main | 24/24 pairs; cached-token counts also match |

The capacity-8 experiments include both C=2 and C=8 workloads under the same active=32 service configuration. For C=8 hot hits with 16 output tokens specifically, GDN output sequences match in **262/320 pairs with prefill graphs off** and **290/320 with graphs on**. Cache-hit categories match across both codec modes.

The single mismatch in the original low-concurrency graph experiment was reproduced by changing prefill batching in either codec mode. In the newer concurrent experiments, repeated identical inputs within the **unfused GDN baseline itself** also produce different sequences. These observations do not establish a fused-codec numerical error, but not every model-output difference has been traced to a specific operation or batch composition. **We do not claim bitwise-identical concurrent GDN model outputs.**

The long-window workload is duration-based, so request counts differ between modes. Output comparisons use common cyclic input indices; all requests, including unmatched indices, are checked for lengths, SSE consistency, and absence of retractions.

Separate, untimed verification services compared actual model states against the eager INT8 reference:

| Configuration | Verified saves | Verified restores | Largest restore batch |
|---|---:|---:|---:|
| GDN, prefill graph off | 492 | 12 | 7 slots |
| GDN, existing graph on | 492 | 12 | 7 slots |
| KDA, prefill graph off | 504 | 12 | 7 slots |

Counts are summed over both TP ranks. Every nonempty fused call was verified exactly. The original KDA experiment additionally verified 104 saves and 32 restores.

KDA's initial token outputs matched, but logprobs were not bitwise identical: the maximum observed absolute difference was approximately 0.00507. Within-codec repetitions also showed logprob variability; its precise source was not fully isolated. These tests compare implementations of the **same INT8 codec**, not INT8 against BF16 or AWQ weights against BF16 weights. No full task-accuracy benchmark was run.

### Main regressions and local CI

| Check on the pinned main | Result |
|---|---|
| Codec/fallback/commit tests plus existing INT8, KDA error-bound, and commit-index regressions | **98 passed**, including the 75 new tests |
| Mamba capacity, views, movement/compaction, retraction, cache assembly, and byte accounting | **159 passed, 766 subtests passed** |
| Environment handling and exports | **21 passed, 2643 subtests passed** |
| Native Rust TreeCore integration, filtered to Mamba | **48 passed, 534 deselected** |
| Official CPU CI runner, new fallback file | **1/1 file, 2 tests passed** |
| Official GPU CI runner, two new GPU files in extra-a | **2/2 files, 73 tests passed** |
| Applicable pre-commit hooks on all 8 changed files | **Passed** |
| Clean-base patch application and whitespace checks | **Passed** |
| TP2 GDN service correctness smoke test | **24/24 matching output/cached-token pairs**, fused dispatch confirmed |

Runner executions overlap with pytest coverage and are not added together as independent test counts.

The expanded `unit/kernels` run on the standalone codec patch reports **245 passed, 3 failed, 1 skipped**. The same three sampling-test failures reproduce on clean main at the same SHA: old test doubles lack Tensor methods required by the new JIT eligibility check. The test-only fix is maintained separately and is **not included** here. With that independent fix applied, the expanded run reports **247 passed, 1 skipped, 2 subtests passed**; the combined codec/softmax GPU regression reports **103 passed, 19 subtests passed**.

Full remote GitHub CI, all-repository pre-commit, and all CPU/GPU suites were not run. New GPU tests are registered in **extra-a** and require that tier or targeted scheduling. Actual GPU validation was on RTX 5090; other GPU platforms remain untested.

## Speed Tests and Profiling

### Setup and version scope

Performance results use frozen **v0.5.22, `baa9befa381511cd4d56f83980951e5a45fffc6d`, plus this codec change**. The main port received correctness and regression validation above; these repeated performance experiments have **not** been rerun on main.

Common setup: 2× RTX 5090, TP2, BF16 active states, INT8 checkpoints, fixed KV capacity of 32768 tokens, HiCache/speculative decoding off, 1024-token prefixes and 1032-token queries, greedy sampling with EOS ignored.

- **GDN:** Qwen3.8-27B-NVFP4, FP8 KV; prefill graphs disabled or using the existing breakable backend.
- **KDA:** `cyankiwi/Kimi-Linear-48B-A3B-Instruct-AWQ-4bit`, revision `50cc9c376c3440e8cc465f1274304f903319a9f1`, BF16 KV, prefill graphs disabled. Both arms share separate AWQ-loader/tokenizer/dependency-path compatibility adjustments, excluded from this PR. This does not establish out-of-the-box model compatibility or a KDA service test on main.

| Experiment | Active/checkpoint slots | Service limit / decode graph max batch | Runs |
|---|---|---|---|
| Original low concurrency | 8/16 | 2/2 | 5 paired restarts and 1760 requests per model/graph configuration |
| C=8 extension | 32/16 | 8/8 | 30 service starts, 10560 requests across three configurations; C=2 controls included |
| KDA long windows | 32/16 | 8/8 | 5 paired restarts, 80320 requests; three C=8 scenarios, 65 seconds each |

The original active=8 setting caps service concurrency at 2 under the four-slots-per-request sizing rule. Both arms therefore use active=32 and decode graphs covering batch 8 for the higher-concurrency experiment; checkpoint capacity remains 16. C=2 controls were rerun under this same configuration. The C=8 hot-hit decode batch was confirmed on both ranks; actual codec calls include batched restores. Client concurrency does not imply every GPU batch is 8.

Effects below are medians of per-restart paired `fused/eager - 1` changes, with 95% bootstrap intervals from 10000 resamples, **n=5 independent restart pairs**, without multiple-comparison correction. Profiling, compilation warmup, and debug verification are excluded from performance timing.

### Temporary memory and isolated kernel time

Single-checkpoint measurements, excluding existing pools:

| State geometry per TP rank | Operation | Eager temporary peak | Fused temporary peak | Eager → fused GPU kernel time | Launches |
|---|---|---:|---:|---:|---:|
| GDN: 48 layers × 24 heads × 128×128 BF16 | Save | 252.84 MiB | 0 MiB | 498.18 → 26.78 μs, **18.60×** | 14→1 |
| Same | Restore | 162.84 MiB | 0 MiB | 255.82 → 12.80 μs, **19.99×** | 7→1 |
| KDA: 20 layers × 16 heads × 128×128 BF16 | Save | 70.23 MiB | 0 MiB | 93.94 → 10.02 μs, **9.38×** | 14→1 |
| Same | Restore | 45.23 MiB | 0 MiB | 57.02 → 4.19 μs, **13.60×** | 7→1 |

GPU times are sums of attributed kernel durations after warmup, excluding inter-kernel launch gaps. **0 MiB means no additional PyTorch global intermediate tensor allocations**, not zero total memory or removal of FP32 arithmetic.

For these particular models, the BF16 SSM payload per checkpoint is approximately **10 MiB/rank for KDA versus 36 MiB/rank for GDN**: `20×16/(48×24) ≈ 27.8%`. This smaller state helps explain KDA's lower absolute codec time and temporary allocation size. It does not imply proportional speedups or end-to-end throughput scaling.

### Service memory footprint

Runtime NVML peaks decrease with pool capacities held fixed within each comparison:

| Configuration | Original low-concurrency setup | Capacity-8 setup |
|---|---:|---:|
| GDN, prefill graph off | −340 MiB/GPU | **−1392 MiB/GPU** |
| GDN, existing prefill graph on | −370 MiB/GPU | **−1440 MiB/GPU** |
| KDA, prefill graph off | −70 MiB/GPU | **−206 MiB/GPU** |

Each value is the difference between per-mode medians over five starts, taking the higher peak across the two GPUs per start. NVML is sampled approximately every 100 ms after readiness, excludes weight loading, and can miss short peaks. Capacity-8 runs include both C=2 and C=8 scenarios.

Larger restore batches increase the eager temporary footprint. These reductions are configuration-dependent, not fixed per-request savings. No saved memory was used to expand the checkpoint pool or improve hit rates.

Metrics need to be distinguished: the original KDA allocated peak remains **15917.94 MiB**, while reserved/NVML peaks fall by 70 MiB. In the capacity-8 setup, allocated peaks change **16265.81→16196.35 MiB**, while reserved peaks change **16646→16440 MiB**. Other phases can determine the overall allocated peak.

### C=8 end-to-end throughput

| C=8 workload | GDN, prefill graph off | GDN, existing graph on | KDA, prefill graph off |
|---|---:|---:|---:|
| Hot hit, 1 output token | +3.04% [−1.34%, +11.37%] | **+5.60% [+3.85%, +7.02%]** | +0.22% [−3.03%, +37.72%] |
| Hot hit, 16 output tokens | +1.55% [−0.74%, +2.60%] | **+2.36% [+2.21%, +3.29%]** | −2.55% [−6.85%, +9.74%] |
| 24-prefix rotation, 16 output tokens | −1.47% [−2.84%, +0.93%] | +1.01% [+0.84%, +1.95%] | +1.16% [−1.10%, +1.77%] |

GDN hot-hit gains with the existing prefill graph are positive in all five pairs. Under the same active=32 setup, the 16-token hot-hit gain is +0.28% at C=2 versus +2.36% at C=8. This supports a larger benefit at higher concurrency for that configuration, not a general speedup claim across all models.

GDN without prefill graphs and KDA hot-hit intervals cross zero. The short KDA 16-token samples contain only 64 requests and last approximately 1.6 seconds, motivating the longer experiment below.

### KDA C=8: longer timing windows

Each scenario uses 10 seconds of matching-load warmup, followed by a cache flush and deterministic prefix warmup, then 65 seconds of continuous requests and a final drain. Eight persistent client workers issue requests. The primary metric includes drain time; the steady-window metric counts completions during seconds 5–60 over 55 seconds.

| Workload | Full-window throughput change | Steady-window change |
|---|---:|---:|
| Hot hit, 1 output token | −0.06% [−2.84%, +1.00%] | +0.00% [−2.63%, +1.35%] |
| Hot hit, 16 output tokens | **+0.73% [−1.58%, +1.69%]** | +1.09% [−1.51%, +2.23%] |
| 24-prefix rotation, 16 output tokens | +0.36% [−0.94%, +4.25%] | +0.74% [−0.74%, +4.11%] |

The five 16-token hot-hit effects are **+0.73%, +0.82%, −0.74%, +1.69%, −1.58%**. The earlier −2.55% does not reproduce as a consistent regression, but the longer-window intervals still cross zero: **no consistent end-to-end KDA speedup is established either**.

Warmup and client scheduling also differ from the original short experiment, so the change cannot be attributed to window length alone. Spikes and negative pairs are retained.

Cache-hit rates are 100% for the hot-hit scenarios and 0% for prefix rotation in both modes. These results measure conversion-cost changes without cache-capacity expansion.

### Whole-service Nsight findings

Separate eager/fused services were profiled after all unprofiled measurements, using Nsight Systems 2025.3 with CUDA/NVTX/OSRT and graph-node tracing. Each warmed scenario runs for approximately six seconds; analysis trims 300 ms from each edge. Additional CPU-call diagnostics are tagged separately.

Actual KDA service results on GPU0:

| Operation | Eager GPU kernel sum | Fused GPU kernel sum | Launches |
|---|---:|---:|---:|
| Single-slot save | 99.94 μs | 10.27 μs | 14→1 |
| Seven-slot restore | 686.55 μs | 37.06 μs | 7→1 |

The other GPU shows similar codec results. A typical eight-request group performs eight single-slot saves and restores of one and seven slots. The corresponding reduction in GPU work is approximately **1.42–1.44 ms/rank**, compared with a roughly 200 ms request-group cycle. This is a workload-budget comparison, not a critical-path prediction: CPU/GPU work can overlap and TP-rank times must not be added.

Unprofiled prefill-forward host spans are approximately **54–56 ms/call**, while decode-graph submission spans are approximately **0.74–0.76 ms/step**. Many eager linear/Triton operations and allocations remain, alongside matrix multiplication, MoE, attention, and TP collectives.

In the largest negative long-window pair, −1.58%, TTFT p50 increases **128.56→131.75 ms**. Prefill host spans increase **54.07→55.27 ms** and **54.43→55.67 ms** on the two ranks, while codec submission remains faster. Another negative pair instead increases post-first-token latency. There is no common pattern of slower codec execution.

Profiling introduces visible perturbation: both modes show occasional hundred-millisecond host launch calls, while the associated GPU graph spans are approximately 3.91 ms. Across **97290 unprofiled hot-hit decode forwards**, the largest host span is only **2.63 ms**, with none above 20 ms; profiled throughput is also approximately 20% lower. Those tracing tails are not evidence for the cause of the unprofiled regression. The original short negative runs were not simultaneously traced, so the precise sources of their variation cannot be reconstructed.

The supported conclusion is that the codec becomes faster and uses less temporary memory, while its end-to-end contribution is small and can be obscured by variation in larger execution stages.


