"""
app.py —— 矿山AI平台 API（FastAPI），目前包含两个模型：
  POST /predict            岩爆风险预警（模块1）
  POST /flotation/predict  浮选精矿二氧化硅含量预测（模块2，化验值辅助的修正模型）
  GET  /health             健康检查（K8s探针使用）

设计要点：
  1. 启动时一次性加载模型，计算文件哈希作为"模型版本"，写入响应和审计日志
  2. 输入校验：非法/缺失/非有限值/时间不连续 → 422，脏数据不进入模型
  3. 模块2的特征构造复用 flotation_features.add_lag_features（与训练同一份代码，避免训练/服务特征不一致）
  4. 降级机制：浮选模型文件不存在（例如晋级门槛未通过）时，服务照常启动，
     浮选接口退回"沿用上一小时化验值"的基线预测，并在响应里明确标注

运行：uv run uvicorn app:app --app-dir src --reload --port 8000
"""
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from flotation_features import PREV, add_lag_features, base_columns

MODEL_PATH = os.getenv("MODEL_PATH", "models/geohazard.joblib")
FLOTATION_PATH = os.getenv("FLOTATION_MODEL_PATH", "models/flotation.joblib")
logger = logging.getLogger("audit")        # 审计日志专用logger
logging.basicConfig(level=logging.INFO, format="%(message)s")

state = {}  # 进程级缓存：存放加载好的模型包与版本号


def file_version(path: str) -> str:
    """用文件内容的SHA-256前12位作为模型版本，日志里可追溯每次预测用的是哪个模型。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时执行一次：加载模型。关闭时清理。"""
    state["bundle"] = joblib.load(MODEL_PATH)
    state["version"] = file_version(MODEL_PATH)
    if os.path.exists(FLOTATION_PATH):
        state["flotation"] = joblib.load(FLOTATION_PATH)
        state["flotation_version"] = file_version(FLOTATION_PATH)
    else:
        # 没有通过门槛的浮选模型：不阻止服务启动，浮选接口走基线降级
        state["flotation"] = None
        state["flotation_version"] = None
        logger.warning(json.dumps({"event": "startup", "warning": "flotation model not found, using persistence fallback"}))
    yield
    state.clear()


app = FastAPI(title="Mine AI Platform API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health():
    """健康检查：Kubernetes的存活/就绪探针调用它。"""
    return {"status": "ok", "model_version": state.get("version"),
            "flotation_model_version": state.get("flotation_version"),
            "flotation_mode": "model" if state.get("flotation") else "persistence_fallback"}


# ====================== 模块1：岩爆风险预警 ======================

class SeismicInput(BaseModel):
    """一个班次的微震监测特征。Literal限制类别取值，ge=0限制数值非负。"""
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
    risk_score: float   # 校准后的风险分数(0~1)，用于排序，不要当作精确概率
    risk_level: Literal["NORMAL", "ELEVATED", "ALERT"]
    threshold: float    # 触发ALERT的阈值（训练时由预警预算决定）
    model_version: str


def level_of(score: float, thr: float) -> str:
    """风险分级（业务假设，需与运营方确认）：>=阈值 ALERT；>=阈值一半 ELEVATED；否则 NORMAL。"""
    if score >= thr:
        return "ALERT"
    if score >= thr * 0.5:
        return "ELEVATED"
    return "NORMAL"


@app.post("/predict", response_model=Prediction)
def predict(item: SeismicInput):
    b = state["bundle"]
    # 按训练时的特征顺序构造DataFrame，保证列顺序一致
    X = pd.DataFrame([item.model_dump()])[b["features"]]
    raw = b["pipeline"].predict_proba(X)[:, 1].reshape(-1, 1)
    score = float(b["platt"].predict_proba(raw)[:, 1][0])
    level = level_of(score, b["threshold"])
    logger.info(json.dumps({"ts": time.time(), "event": "predict", "model_version": state["version"],
                            "risk_score": round(score, 4), "risk_level": level}))
    return Prediction(risk_score=score, risk_level=level,
                      threshold=b["threshold"], model_version=state["version"])


# ====================== 模块2：浮选二氧化硅含量预测 ======================

class HourRecord(BaseModel):
    hour_start: datetime                 # 该小时的起始时间（整点），用于检查时间是否连续
    features: dict[str, float]           # 该小时的传感器汇总值：特征名 -> 数值


class FlotationRequest(BaseModel):
    # 恰好3个连续整点小时，按时间升序，最后一个是"当前小时"；
    # 需要前两个小时是因为模型用到了滞后值(lag1/lag2)和变化量(d1)
    hours: list[HourRecord] = Field(min_length=3, max_length=3)
    prev_lab_silica: float = Field(ge=0)  # 当前小时的上一小时的化验值(%)


class FlotationPrediction(BaseModel):
    predicted_silica_pct: float
    prev_lab_silica_pct: float
    delta_pct: float                      # 模型预测的变化量；降级时为0
    model_used: Literal["delta_rf", "persistence_fallback"]
    model_version: Optional[str]
    gate_rmse_improvement_holdout_pct: Optional[float]  # 该模型在留出集上相对基线的改善幅度


def build_flotation_features(req: FlotationRequest, bundle: dict) -> pd.DataFrame:
    """把请求转换成模型输入（一行）。校验失败一律返回422。
    特征构造调用 flotation_features.add_lag_features —— 与训练用的是同一个函数。"""
    lag_cols = bundle["lag_cols"]                       # 需要做滞后的"小时均值"列
    base_cols = base_columns(bundle["features"])        # 当前小时需要的全部原始列

    # 1) 时间必须是3个连续整点小时，升序
    try:
        idx = pd.DatetimeIndex([pd.Timestamp(h.hour_start) for h in req.hours])
    except (TypeError, ValueError):
        raise HTTPException(422, "hour_start values are not comparable (mixed timezones?)")
    if not all((idx[i + 1] - idx[i]) == pd.Timedelta(hours=1) for i in range(2)):
        raise HTTPException(422, "hours must be 3 consecutive hourly records in ascending order")

    # 2) 特征完整性：滞后列在3个小时里都要有；当前小时要有全部原始列
    missing = [c for c in lag_cols if any(c not in h.features for h in req.hours)]
    missing += [c for c in base_cols if c not in req.hours[-1].features and c not in missing]
    if missing:
        raise HTTPException(422, f"missing features: {missing}")

    # 3) 数值必须是有限数（拒绝NaN/Infinity）
    for h in req.hours:
        if not all(np.isfinite(v) for v in h.features.values()):
            raise HTTPException(422, "features must be finite numbers")

    # 4) 只取需要的列构造DataFrame（早期小时缺少__std列没关系，会因缺少滞后被丢弃）
    needed = sorted(set(lag_cols) | set(base_cols))
    df = pd.DataFrame([{k: h.features.get(k) for k in needed} for h in req.hours], index=idx)
    df[PREV] = req.prev_lab_silica
    out = add_lag_features(df, lag_cols)                # 与训练相同的特征构造
    if len(out) != 1:
        raise HTTPException(500, "feature construction failed")
    return out[bundle["features"]]                      # 按训练时的列顺序


@app.post("/flotation/predict", response_model=FlotationPrediction)
def flotation_predict(req: FlotationRequest):
    fl = state.get("flotation")
    prev = req.prev_lab_silica
    if fl is None:
        # 降级：没有可部署模型，沿用上一小时化验值（persistence），并明确标注
        delta, used, version, gate_imp = 0.0, "persistence_fallback", None, None
    else:
        X = build_flotation_features(req, fl)
        delta = float(fl["model"].predict(X)[0])
        used, version = "delta_rf", state["flotation_version"]
        gate_imp = fl["gate"]["rmse_improvement_holdout_pct"]
    pred = max(0.0, prev + delta)                       # 含量不可能为负
    logger.info(json.dumps({"ts": time.time(), "event": "flotation_predict", "model_used": used,
                            "model_version": version, "pred": round(pred, 4), "delta": round(delta, 4)}))
    return FlotationPrediction(predicted_silica_pct=pred, prev_lab_silica_pct=prev, delta_pct=delta,
                               model_used=used, model_version=version,
                               gate_rmse_improvement_holdout_pct=gate_imp)
