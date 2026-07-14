| Group | Errors | d′ | MLE meta-d′ | **MLE M-ratio** | **Bayesian M-ratio [95% HDI]** | Evidence |
|---|---:|---:|---:|---:|---|---|
| **human** | 437 | 1.34 | 1.83 | 1.37 | 1.33 [1.16, 1.52] | data-driven ✓ |
| local-qwen3-1.7b | 117 | 2.87 | 0.89 | 0.31 | 0.37 [0.26, 0.52] | data-driven ✓ |
| local-gemma-e2b | 106 | 3.04 | 1.14 | 0.38 | 0.42 [0.31, 0.54] | data-driven ✓ |
| local-gemma-e4b | 45 | 3.88 | 2.68 | 0.69 | 0.79 [0.64, 0.92] | data-driven ✓ |
| local-qwen3-4b | 35 | 3.97 | 2.31 | 0.58 | 0.67 [0.54, 0.80] | data-driven ✓ |
| local-qwen3-14b | 26 | 4.23 | 2.29 | 0.54 | 0.67 [0.54, 0.79] | regularized ~ |
| qwen-turbo | 23 | 4.30 | 1.70 | 0.40 | 0.76 [0.61, 0.91] | regularized ~ |
| gpt-4o-mini | 20 | 4.48 | 3.33 | 0.74 | 0.84 [0.70, 0.96] | regularized ~ |
| gpt-5.4-mini | 15 | 4.60 | 3.64 | 0.79 | 0.94 [0.79, 1.07] | regularized ~ |
| deepseek-v4-flash-nothink | 10 | 5.21 | 1.59 | 0.30 | 0.62 [0.47, 0.76] | regularized ~ |
| deepseek-v4-pro-nothink | 9 | 4.95 | 2.01 | 0.41 | 0.77 [0.64, 0.94] | prior-dominated ✗ |
| deepseek-v4-flash | 9 | 4.94 | 2.69 | 0.55 | 0.75 [0.61, 0.91] | prior-dominated ✗ |
| deepseek-v4-pro | 8 | 5.00 | 3.62 | 0.73 | 0.89 [0.72, 1.03] | prior-dominated ✗ |
| gpt-4o | 6 | 5.24 | 5.06 | 0.97 | 0.97 [0.84, 1.11] | prior-dominated ✗ |
| qwen3.7-plus-nothink | 5 | 5.30 | 3.80 | 0.72 | 1.00 [0.85, 1.17] | prior-dominated ✗ |
| qwen3.7-max-nothink | 4 | 5.51 | 2.28 | 0.41 | 0.89 [0.73, 1.04] | prior-dominated ✗ |
| qwen3.7-plus | 1 | 5.83 | 5.63 | 0.97 | 1.02 [0.87, 1.19] | prior-dominated ✗ |
| local-gemma-26b-a4b-qat | 1 | 5.81 | 5.18 | 0.89 | 1.02 [0.89, 1.19] | prior-dominated ✗ |
| gpt-5.5 | 1 | 5.83 | 5.77 | 0.99 | 1.07 [0.92, 1.25] | prior-dominated ✗ |
| qwen3.7-max | 0 | 6.04 | 5.99 | 0.99 | 1.07 [0.93, 1.23] | prior-dominated ✗ |