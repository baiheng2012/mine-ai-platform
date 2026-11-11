#!/usr/bin/env bash
# 在"干净的克隆"里按CI的步骤完整运行一遍，提前发现"本机有、仓库里没有"的文件问题。
# 用法(在项目根目录)：bash scripts/ci_local.sh
# 注意：克隆的是"已提交"的状态，未提交的改动不会被包含，这正是要检查的。
set -euo pipefail
TMP=$(mktemp -d)
git clone --quiet . "$TMP"
cd "$TMP"
echo "== clean clone at $TMP =="

uv sync --frozen                                   # 与CI一致：严格按uv.lock安装默认依赖组
mkdir -p models
uv run python src/train.py
uv run python src/train_flotation_final.py
uv run python scripts/sample_request.py > sample.json
uv run python scripts/sample_flotation_request.py > sample_flotation.json
uv run python src/monitor_flotation.py
uv run python src/monitor_geohazard.py
uv run pytest -q
echo "== local CI check passed =="
