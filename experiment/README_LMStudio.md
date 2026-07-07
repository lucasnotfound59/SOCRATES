# 实验运行指南 / Experiment Guide
元认知校准研究 — 本地小模型(LM Studio)+ API 大模型的批量测评。

本目录文件:
- `standard_prompt.md` — 标准 prompt(中英、人类版与模型版)。
- `run_experiment.py` — 运行器(读题库→无上下文逐题多次采样→解析→写 CSV)。
- `models.yaml` — 配置(改这个就能增删模型、切语言/采样)。
- `requirements.txt` — Python 依赖。

---

## 0. 一次性准备

```bash
cd "EAI Project/experiment"
python3 -m pip install -r requirements.txt
# 先自测解析器(不联网、不调用模型):
python3 run_experiment.py --selftest
```

---

## 1. 用 LM Studio 跑本地小模型

**LM Studio 的关键点**:它内置一个 **OpenAI 兼容的本地服务器**,所以本程序不需要为它单独写代码,和调用云端 API 是同一套。

步骤:

1. **安装**:到 lmstudio.ai 下载并安装(Mac/Windows/Linux 均可)。

2. **下载模型**:左侧放大镜(Search)→ 搜索并下载小模型。建议覆盖能力梯度、跨厂商,例如:
   - Gemma 系列:`gemma-2-2b-it`、`gemma-2-9b-it`
   - Qwen 系列:`Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`
   - (可按机器显存挑量化版本 Q4_K_M 等;显存不够就选更小的)

3. **启动本地服务器**:左侧 **Developer / Local Server**(不同版本叫法略异)→ 选中已加载的模型 → **Start Server**。默认地址即 `http://localhost:1234/v1`。

4. **拿到 model_id**:服务器页面会显示当前模型的标识符(形如 `google/gemma-2-2b-it` 或 `qwen2.5-3b-instruct`)。**把它原样填进 `models.yaml` 的 `model_id`**。

5. **一次一个 or 多个**:简单起见,**一次加载并测一个本地模型**;测完在 LM Studio 里换下一个模型、改 `models.yaml` 的 `model_id` 再跑。断点续跑会自动跳过已完成的项,不会重复。

> 验证服务器通了:
> ```bash
> curl http://localhost:1234/v1/models
> ```
> 能列出模型即 OK。

---

## 2. 小规模自测(强烈建议先做)

先用本地模型跑前 5 题,确认答案能被正确解析:

```bash
python3 run_experiment.py --limit 5 --models local-gemma
```

打开 `results/results.csv` 检查:`parsed_answer` 有值、`parse_ok` 多为 True、`correct` 有 True/False。
若 `parse_ok` 大量为 False(小模型不听 JSON 指令),见下方"排错"。

---

## 3. 正式运行

```bash
# 全部模型 × 中英 × 400 题 × n_samples 次
python3 run_experiment.py

# 只跑中文:
python3 run_experiment.py --languages zh

# 只跑指定模型:
python3 run_experiment.py --models local-qwen,gpt-4o-mini
```

- **无上下文记忆**:每题都是一次独立调用(仅 system+当题),题间互不影响 —— 符合设计。
- **断点续跑**:中断后重复同一命令即可,已完成的 (模型,语言,题号,采样序号) 会被跳过。
- **进度**:每题打印一个 `.`;每个模型/语言组合结束打印 `done`。

---

## 4. 接入云端 API(大模型)

各家都提供 **OpenAI 兼容端点**,只需在 `models.yaml` 里取消注释对应块、填 `model_id`,并把密钥放进环境变量。

| 提供商 | base_url | 环境变量 | model_id 例 |
|---|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| 通义千问 Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `DASHSCOPE_API_KEY` | `qwen-plus` |
| Anthropic Claude | `https://api.anthropic.com/v1/` | `ANTHROPIC_API_KEY` | `claude-opus-4-1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| OpenRouter(聚合) | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `meta-llama/llama-3.1-8b-instruct` |

设置密钥(当前终端会话):
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
```
> 各家 `model_id` 与端点可能随版本变化,填之前以其官方文档为准。上表 model_id 仅为示例。

---

## 5. 输出 CSV 字段

`results/results.csv` 每行 = 一个模型对一道题的一次采样:

| 字段 | 含义 |
|---|---|
| model / model_id | 模型标识 |
| language | zh / en |
| item_id / type / subtype / difficulty | 题目信息 |
| gold | 标准答案(True/False) |
| sample_idx | 第几次采样(0..n-1) |
| parsed_answer | 解析出的模型答案 |
| correct | 是否正确(与 gold 比对) |
| confidence | 解析出的 1–5 档置信度 |
| parse_ok | 是否严格按 JSON 解析成功 |
| latency_s | 单次耗时 |
| raw_response | 原始回复(截断 1000 字,便于排错) |
| logprobs | 若开启且模型支持,存原始 token 概率(辅助分析) |
| error | 调用报错信息(为空即正常) |

分析阶段:置信度按 `standard_prompt.md` 第 4 节映射到 50%–100% 算 ECE;按 type/difficulty 分层;组级 M-ratio 用 HMeta-d。

---

## 6. 采样与成本

- 当前默认 `n_samples: 5`、`temperature: 0.7`(多次采样估计随机性)。
- 想改成确定性单次:把该模型设 `temperature: 0` 且 `n_samples: 1`。
- **调用量** = 模型数 × 语言(2) × 400 × n_samples。例:1 个模型、中英、5 次 = 4000 次调用。云端 API 按此估算 token 费用;本地模型只花时间。
- API 限流报错:把该模型或全局 `sleep` 调大(如 0.5),`retries` 已默认重试 3 次。

---

## 7. 排错

- **`parse_ok` 大量 False**:小模型不严格输出 JSON。对策:① 看 `raw_response` 确认它其实答了但格式乱;解析器已有关键词回退,`parsed_answer/confidence` 多数仍能拿到。② 可在 LM Studio 里给该模型开启 "JSON"/结构化输出,并在 `models.yaml` 该模型下加 `json_mode: true`(需模型/端点支持)。
- **连不上本地服务器**:确认 LM Studio 里已 Start Server、`curl http://localhost:1234/v1/models` 能通、`model_id` 与页面一致。
- **`confidence` 常缺失**:多为模型没给数字;检查 `raw_response`,必要时在 prompt 里更强调"必须给 1–5"。改 prompt 请同时改 `standard_prompt.md` 保持一致。
- **想重跑某模型**:删掉 `results.csv` 里该模型的行,或换 `output_file` 另存。
