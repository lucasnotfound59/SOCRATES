#!/bin/bash
# ------------------------------------------------------------------
# 本地模型自动清扫:依次 加载 → 跑实验 → 卸载 → 下一个,无人值守。
# 用 LM Studio 的 lms CLI 换模型,比 GUI 点击可靠得多。
#
# 用前:
#   1) 确保 lms 可用:  lms --version   (找不到就 ~/.lmstudio/bin/lms bootstrap)
#   2) 查模型 key:      lms ls
#   3) 把下面 PAIRS 里的 "LM Studio的key" 换成 lms ls 显示的确切 key,
#      "配置name" 保持与 models.yaml 里的 name 一致。
#   4) 运行:  bash run_local_sweep.sh
# ------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

OUT_THINK="results_qwen3_local.csv"           # 思考版输出
OUT_NOTHINK="results_qwen3_local_nothink.csv" # 无思考版输出
CONC=4                                        # 并发
CTX=1000                                       # 加载时的上下文长度上限(省内存)。注意:思考版输出上限2048 > 1000,
                                               # 思考较长时会被截断(finish_reason=length);无思考版(512)不受影响。

# 格式:  "LM Studio的模型key|思考版name|无思考版name"
#   左边 key 用 `lms ls` 查。每个模型只加载一次,连着跑 思考版 + 无思考版。
PAIRS=(
  "qwen/qwen3-1.7b|local-qwen3-1.7b|local-qwen3-1.7b-nothink"
  "qwen/qwen3-4b|local-qwen3-4b|local-qwen3-4b-nothink"
  "qwen/qwen3-14b|local-qwen3-14b|local-qwen3-14b-nothink"
)

# 确保本地服务器在跑
lms server start >/dev/null 2>&1 || true

for pair in "${PAIRS[@]}"; do
  IFS='|' read -r KEY NAME_T NAME_N <<< "$pair"
  if [[ "$KEY" == REPLACE_WITH_KEY_* ]]; then
    echo "!! 还没填模型 key:$NAME_T —— 先用 lms ls 查到 key 填进 PAIRS,跳过"; continue
  fi
  echo ""
  echo "==================== $KEY ===================="
  echo ">> 卸载旧模型 & 加载 $KEY"
  lms unload --all >/dev/null 2>&1 || true
  # --context-length 限制上下文,避免默认自动用满(如8000)导致内存爆炸
  if ! lms load "$KEY" -y --context-length "$CTX"; then
    echo "!! 加载失败:$KEY —— 检查 key 或把 --context-length 换成你版本支持的写法,跳过"; continue
  fi
  echo ">> [思考版]   $NAME_T → $OUT_THINK"
  caffeinate -is python3 run_experiment.py --models "$NAME_T" --output-file "$OUT_THINK" --concurrency "$CONC"
  echo ">> [无思考版] $NAME_N → $OUT_NOTHINK"
  caffeinate -is python3 run_experiment.py --models "$NAME_N" --output-file "$OUT_NOTHINK" --concurrency "$CONC"
  echo ">> $KEY 完成(思考版 + 无思考版)"
done

lms unload --all >/dev/null 2>&1 || true
echo ""
echo "全部完成。查看完成度:"
echo "  python3 run_experiment.py --models local-qwen3-1.7b,local-qwen3-4b,local-qwen3-14b --output-file $OUT_THINK --status"
echo "  python3 run_experiment.py --models local-qwen3-1.7b-nothink,local-qwen3-4b-nothink,local-qwen3-14b-nothink --output-file $OUT_NOTHINK --status"
