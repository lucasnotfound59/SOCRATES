# 待测模型清单 / Models to Test
> 元认知校准研究 · 更新于 2026-07-05
> 目标:覆盖**能力梯度**(弱→中→强→前沿)+ **跨厂商**(每家出至少一个点),以便定位人与不同 LLM 的校准分叉。
> ⚠️ 下方 API 的 `model_id` 为 2026-07 当前名称,**跑之前请对照各家官方文档核对**(版本更新很快)。

---

## 运行配置(当前)
- 语言:**英文**(`languages: ["en"]`)
- 每题采样:`n_samples: 4` · 并发:`concurrency: 5` · `max_tokens: 1024`
- 每模型调用量 ≈ 400 题 × 4 = **1600 次**
- 输出:本地写 `results_local.csv`;建议 API 各家单独 `output_file`(便于分开/控成本)

---

## A. 本地模型(MLX · LM Studio,一次加载一个)

| 状态 | name | 层级 | model_id(以 LM Studio 实际为准) | 思考型 | 备注 |
|---|---|---|---|---|---|
| ✅ 已完成 | local-gemma-e4b | 中 | google/gemma-4-e4b | 是 | 数据在 `results_gemma4-e4b.csv`;acc≈0.97 近天花板 |
| ✅ 已完成 | local-gemma-e2b | **弱** | google/gemma-4-e2b | 是 | **优先**:弱端,最可能给出置信度方差 |
| ✅ 已完成 | local-gemma-26b-a4b-qat | 强 | google/gemma-4-26b-a4b-qat | 是 | MoE,26B 全进内存/仅激活4B → 快;QAT 低比特高质量 |

> 无 MLX 版的 12B / 31B 已放弃(near-ceiling + 运行时不一致,详见讨论)。

---

## B. 云端 API 模型(OpenAI 兼容端点)

> 每家取**前沿旗舰 + 一个更小/更快**两个点,兼顾"跨厂商"与"厂商内梯度",同时控成本。
> 密钥放环境变量;`base_url` 与 `model_id` 见下。大多为思考型 → 保持 `max_tokens` 较大。

### OpenAI  ·  base_url `https://api.openai.com/v1`  ·  env `OPENAI_API_KEY`
| 状态 | 层级 | model_id | 备注 |
|---|---|---|---|
| ⬜ | 前沿 | `gpt-5.5` | 当前旗舰(快照 gpt-5.5-2026-04-23);支持 logprobs |
| ⬜ | 中/快 | `gpt-5.4-mini` | 低延迟低价;可再加 `gpt-5.4-nano` 作更弱点 |

### Anthropic  ·  base_url `https://api.anthropic.com/v1/`  ·  env `ANTHROPIC_API_KEY`
| 状态 | 层级 | model_id | 备注 |
|---|---|---|---|
| ⬜ | 前沿 | `claude-opus-4-8` | 旗舰 |
| ⬜ | 中/快 | `claude-haiku-4-5-20251001` | 快而便宜;(可选中间点 `claude-sonnet-5`) |

### DeepSeek  ·  base_url `https://api.deepseek.com`  ·  env `DEEPSEEK_API_KEY`
| 状态 | 层级 | model_id | 备注 |
|---|---|---|---|
| ⬜ | 前沿 | `deepseek-v4-pro` | MoE,思考模式;1M 上下文 |
| ⬜ | 中/快 | `deepseek-v4-flash` | 便宜高吞吐(注:旧 `deepseek-chat/-reasoner` 2026-07-24 弃用) |

### 通义千问 Qwen  ·  base_url `https://dashscope.aliyuncs.com/compatible-mode/v1`  ·  env `DASHSCOPE_API_KEY`
| 状态 | 层级 | model_id | 备注 |
|---|---|---|---|
| ⬜ | 前沿 | `qwen3.7-max` | 旗舰(纯文本) |
| ⬜ | 中/快 | `qwen3.5-flash` | 快而便宜 |

### Google Gemini  ·  base_url `https://generativelanguage.googleapis.com/v1beta/openai/`  ·  env `GEMINI_API_KEY`
| 状态 | 层级 | model_id | 备注 |
|---|---|---|---|
| ⬜ | 前沿 | `gemini-3-pro` | 旗舰(3.5 Pro 若已开放可换用) |
| ⬜ | 中/快 | `gemini-3.5-flash` | 最新 Flash;或更弱的 `gemini-3.1-flash-lite` |

---

## 优先级建议(先跑哪些)

**P0 — 核心(足以支撑主结论):**
- 本地全梯度:E2B、E4B(已完成)、26B-A4B
- 三家前沿旗舰:`gpt-5.5`、`claude-opus-4-8`、`deepseek-v4-pro`
- 上述三家各配一个快模型:`gpt-5.4-mini`、`claude-haiku-4-5`、`deepseek-v4-flash`

**P1 — 增强(跨厂商更全):**
- Qwen(`qwen3.7-max` + `qwen3.5-flash`)、Gemini(`gemini-3-pro` + `gemini-3.5-flash`)

**P2 — 可选(补更多梯度点):**
- `gpt-5.4-nano`、`claude-sonnet-5`、`gemini-3.1-flash-lite` 等

---

## 通用注意事项

1. **model_id 必核对**:API 版本更新快,上表为 2026-07 名称,跑前对照官方文档确认确切字符串(如 `gpt-5.5` vs 带日期快照)。
2. **思考型模型**:本研究已定"允许思考、调大 max_tokens"。多数前沿模型是思考型,思考会吃 token 且更慢/更贵 → 保持 `max_tokens≥1024`,并盯 `finish_reason` 是否 `length`(是则调大)、看 `reasoning_tokens` 分布。
3. **成本**:每模型英文 1600 次调用。前沿+思考模型 token 消耗大,先用 `--limit 5` 小跑校验解析,再放量;必要时降 `n_samples`。
4. **logprobs(辅助分析)**:仅部分模型返回(OpenAI 通常可)。需要时在该模型配置加 `supports_logprobs: true` 并全局 `capture_logprobs: true`。
5. **本地一次一个**:LM Studio 只按已加载模型回答;每次 `--models` 指定与加载一致,跑后核对 `served_model` 列。
6. **语言**:当前只跑英文。若人类被试用中文作答,需把 `languages` 改回含 `zh` 补跑,否则主对比语言不匹配。
7. **配置写法**:参照 `models.yaml` 里现成的本地块与注释掉的 API 块,填 `model_id`/`api_key_env` 后取消注释即可。

---

## 来源(型号信息,2026-07 核对)
- OpenAI: https://developers.openai.com/api/docs/models/all
- Anthropic: 见本环境模型信息(Opus 4.8 / Sonnet 5 / Haiku 4.5 / Fable 5)
- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing
- Qwen/DashScope: https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
- Gemini: https://ai.google.dev/gemini-api/docs/models
- Gemma 4: https://ai.google.dev/gemma/docs/core
