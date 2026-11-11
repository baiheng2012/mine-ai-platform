#!/usr/bin/env bash
# 在"干净的克隆"里按CI的步骤完整运行一遍，提前发现"本机有、仓库里没有"的问题。
# 用法(在项目根目录)：bash scripts/ci_local.sh
# 注意：克隆的是"已提交"的状态，未提交的改动不会被检查。
set -euo pipefail

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: uncommitted changes are NOT included in this check:"
  git status --short
fi


# 临时克隆放在家目录下：与uv缓存在同一文件系统，uv可以使用硬链接，几乎不占额外空间
# （/tmp 在部分WSL配置下是容量有限的tmpfs，且与uv缓存不在同一文件系统）
CI_ROOT="$HOME/.cache/mine-ai-ci"
mkdir -p "$CI_ROOT"
TMP=$(mktemp -d -p "$CI_ROOT")
# 无论成功还是失败，脚本退出时都删除临时克隆，避免一次次占满磁盘
trap 'rm -rf "$TMP"' EXIT



git clone --quiet . "$TMP"
cd "$TMP"
echo "== clean clone at $TMP =="

# 数据文件必须在仓库里（缺了说明没提交，或被.gitignore忽略了）
for f in data/seismic-bumps.arff data/flotation_hourly.csv; do
  test -f "$f" || { echo "MISSING in repository: $f"; exit 1; }
done

uv sync --frozen                       # 与CI一致
bash scripts/ci_pipeline.sh            # 与CI使用同一份步骤
docker build -t geohazard-api:ci .     # 与CI最后一步一致
echo "== local CI check passed =="
