#!/usr/bin/env bash
# CI流水线的唯一定义：GitHub Actions 和本地 ci_local.sh 都调用这个脚本，
# 这样两边的步骤永远一致，不会出现"本地通过、CI缺步骤"的情况。
# 前提：已经执行过 uv sync --frozen，并且在项目根目录运行。
set -euo pipefail

mkdir -p models                                            # models/ 被.gitignore排除，仓库里没有

uv run python src/train.py                                 # 模块1：岩爆预警模型
uv run python src/train_flotation_final.py                 # 模块2：浮选模型(带晋级门槛，不达标会失败)

uv run python scripts/sample_request.py > sample.json      # 生成API测试用的请求样例
uv run python scripts/sample_flotation_request.py > sample_flotation.json

uv run python src/monitor_flotation.py                     # 监控脚本冒烟测试
uv run python src/monitor_geohazard.py

uv run pytest -v                                           # 全部测试
