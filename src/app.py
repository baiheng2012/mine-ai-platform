"""
app.py —— 岩爆风险预警 API 服务（FastAPI）

架构要点：
  1. 启动时加载一次模型包（lifespan），之后每个请求直接复用，不重复读磁盘
  2. 请求体用 Pydantic 校验：类别只能是合法取值、数值不能为负，非法输入直接返回422，
     不会让脏数据进入模型（数据质量/安全的第一道防线）
  3. 返回风险分数 + 风险等级 + 模型版本；每次预测写一条结构化审计日志（治理要求）
  4. 模型路径通过环境变量 MODEL_PATH 配置，方便本地、Docker、K8s用同一份代码

运行：uv run uvicorn app:app --app-dir src --reload --port 8000
文档：浏览器打开 http://localhost:8000/docs （FastAPI自动生成的交互式文档）
"""
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/geohazard.joblib")
logger = logging.getLogger("audit")        # 审计日志专用logger
logging.basicConfig(level=logging.INFO, format="%(message)s")

state = {}  # 进程级缓存：存放加载好的模型包与版本号


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时执行一次：加载模型，并计算模型文件哈希作为'模型版本'，
    这样日志里能追溯每次预测用的是哪个版本的模型。"""
    bundle = joblib.load(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        version = hashlib.sha256(f.read()).hexdigest()[:12]
    state.update(bundle=bundle, version=version)
    yield
    state.clear()  # 关闭时清理


app = FastAPI(title="Geohazard Risk API", version="0.1.0", lifespan=lifespan)


class SeismicInput(BaseModel):
    """一个班次的微震监测特征。字段名与训练数据一致。
    Literal限制类别取值，ge=0限制数值非负——校验失败会自动返回422和清晰的错误信息。"""
    seismic: Literal["a", "b", "c", "d"]
    seismoacoustic: Literal["a", "b", "c", "d"]
    shift: Literal["W", "N"]
    genergy: float = Field(ge=0)
    gpuls: float = Field(ge=0)
    gdenergy: float
    gdpuls: float
    ghazard: Literal["a", "b", "c", "d"]
    nbumps: int = Field(ge=0)
    nbumps2: int = Field(ge=0)
    nbumps3: int = Field(ge=0)
    nbumps4: int = Field(ge=0)
    nbumps5: int = Field(ge=0)
    nbumps6: int = Field(ge=0)
    nbumps7: int = Field(ge=0)
    nbumps89: int = Field(ge=0)
    energy: float = Field(ge=0)
    maxenergy: float = Field(ge=0)


class Prediction(BaseModel):
    risk_score: float   # 校准后的风险概率(0~1)，用于排序，不要当作精确概率
    risk_level: Literal["NORMAL", "ELEVATED", "ALERT"]
    threshold: float    # 触发ALERT的阈值（训练时由预警预算决定）
    model_version: str  # 模型文件哈希前12位


def level_of(score: float, thr: float) -> str:
    """风险分级规则（业务假设，需与运营方确认）：
    ≥阈值 → ALERT（触发人工检查）；≥阈值的一半 → ELEVATED（关注）；否则NORMAL。"""
    if score >= thr:
        return "ALERT"
    if score >= thr * 0.5:
        return "ELEVATED"
    return "NORMAL"


@app.get("/health")
def health():
    """健康检查：Kubernetes的存活/就绪探针之后会调用它。"""
    return {"status": "ok", "model_version": state.get("version")}


@app.post("/predict", response_model=Prediction)
def predict(item: SeismicInput):
    b = state["bundle"]
    # 按训练时的特征顺序构造DataFrame，保证列顺序一致（避免特征错位这类隐蔽bug）
    X = pd.DataFrame([item.model_dump()])[b["features"]]
    raw = b["pipeline"].predict_proba(X)[:, 1].reshape(-1, 1)  # 模型原始分数
    score = float(b["platt"].predict_proba(raw)[:, 1][0])      # Platt校准后的分数
    level = level_of(score, b["threshold"])

    # 审计日志：一行JSON，含时间、模型版本、分数、等级。
    # 。
    logger.info(json.dumps({
        "ts": time.time(), "event": "predict", "model_version": state["version"],
        "risk_score": round(score, 4), "risk_level": level,
    }))
    return Prediction(risk_score=score, risk_level=level,
                      threshold=b["threshold"], model_version=state["version"])
