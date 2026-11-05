# Supplementary results: INT8 Mamba checkpoints in HiCache L1/L2

This evidence accompanies standalone change 002. The rollback change (#48049) and fused codec (#48035) are separate. This branch is for review evidence only.

## Refreshed main validation

The change was ported to the HiCache host-pool assembly in main. The production code and tests in the final port are identical to the broad regression run; the intervening upstream commit changed only documentation.

| Check | Result |
|---|---|
| Related CPU regressions | 431 passed, 1 skipped, 283 subtests passed |
| Related NVIDIA GPU regressions | 82 passed, 19 skipped |
| Native Rust Mamba TreeCore integration | 48 passed, 534 deselected |
| Applicable pre-commit hooks and git diff whitespace check | Passed |
| Focused checkpoint admission, host assembly, L2 restore, and configuration tests on final port | 16 passed, 18 subtests passed |

The broad run used base `afea203b`; the final focused run used base `e3b64bf2`. Tests overlap between suites. The local environment used two RTX 5090 GPUs, PyTorch 2.14.1+cu130, and sglang-kernel 0.4.9 for the targeted checks. Full upstream CI and a fresh dependency installation were not part of this validation.

## Benchmark evidence

The serving, throughput, and GSM8K experiments reported in the PR were collected on the earlier standalone 002 commit `15a00b66a6b3` (base `af98fcd545f9`) and were not repeated after the rebase.

- [Full supplementary report (Chinese)](SUPPLEMENTARY_ZH.md): complete performance workload matrix, experimental controls, accuracy and logprob comparisons, the 72-slot memory experiment, and all chained restore depths.
- [Full experiment archive](002-pr-supplement-20261105-v2.zip): raw timing samples, chained-restore per-question generation and scoring, reports, figures, inputs, and an evidence manifest.
- [Ten-workload performance summary (CSV)](PERFORMANCE.csv).
