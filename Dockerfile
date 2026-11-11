# 基础镜像：官方uv镜像（自带uv + Python），slim版本体积小
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# UV_COMPILE_BYTECODE：预编译字节码，加快服务启动
# UV_LINK_MODE=copy：Docker里用复制而非硬链接，避免跨文件系统警告
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# 【缓存技巧】先只复制依赖清单并安装依赖。
# 只要 pyproject.toml / uv.lock 没变，这一层就会命中缓存，改代码后重建镜像只需几秒。
COPY pyproject.toml uv.lock ./
# --frozen：严格按uv.lock安装，保证与本地版本完全一致（可复现）
# --no-dev --no-group train：不装测试和训练专用依赖，镜像更小
# --no-install-project：本项目自身不是库，不需要安装
RUN uv sync --frozen --no-dev --no-group train --no-install-project

# 再复制源码和模型文件（这些变化频繁，放在后面）
COPY src/ ./src/
COPY models/geohazard.joblib ./models/geohazard.joblib
COPY models/flotation.joblib ./models/flotation.joblib


# 把虚拟环境放进PATH，之后直接用 uvicorn 命令；通过环境变量指定模型路径
#ENV PATH="/app/.venv/bin:$PATH" MODEL_PATH=/app/models/geohazard.joblib
ENV PATH="/app/.venv/bin:$PATH" MODEL_PATH=/app/models/geohazard.joblib FLOTATION_MODEL_PATH=/app/models/flotation.joblib
# 安全实践：不以root用户运行（JD里强调安全合规）
RUN useradd -m appuser
USER appuser

EXPOSE 8000
# 0.0.0.0 表示监听容器内所有网卡，否则宿主机访问不到
CMD ["uvicorn", "app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
