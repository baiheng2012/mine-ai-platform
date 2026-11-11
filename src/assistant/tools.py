"""
tools.py —— 助手可以调用的工具。

设计：
  - 工具通过 ctx.api_post 调用线上预测服务，不直接加载模型（与生产路径一致，测试时可替换为TestClient）
  - 工具返回 {"facts": ..., "meta": ...}：
      facts = 可以交给LLM的、只含数字和枚举值的结构化结果(不含时间戳、哈希等标识符)
      meta  = 只写进审计日志的信息(如模型版本)
  - 助手读不到真实标签(class / 化验目标值)：构造请求时已去掉
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from assistant.guardrails import sanitize
from data import load_seismic
from flotation_features import LEAK, TARGET


class ToolError(Exception):
    """工具可预期的失败(参数不合法、数据缺失等)，信息可以直接展示给用户。"""


class DataSource:
    """演示用的数据源：用数据集里的记录模拟"当前班次/当前小时"的实时数据。"""

    def __init__(self):
        self._seis = None
        self._flot = None

    @property
    def seis(self):
        if self._seis is None:
            self._seis = load_seismic()
        return self._seis

    @property
    def flot(self):
        if self._flot is None:
            self._flot = (pd.read_csv("data/flotation_hourly.csv", parse_dates=["date"])
                          .set_index("date").sort_index())
        return self._flot

    def seismic_payload(self, shift_id: int) -> dict:
        df = self.seis
        if not 0 <= shift_id < len(df):
            raise ToolError(f"shift number must be between 0 and {len(df) - 1}")
        row = df.iloc[shift_id].drop("class").to_dict()      # 去掉标签列：助手看不到真实事件结果
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}

    def _continuous(self, i: int) -> bool:
        idx, one = self.flot.index, pd.Timedelta(hours=1)
        return (i >= 3 and idx[i] - idx[i - 1] == one and idx[i - 1] - idx[i - 2] == one
                and idx[i - 2] - idx[i - 3] == one)

    def latest_continuous_hour(self) -> int:
        return next(i for i in range(len(self.flot) - 1, 2, -1) if self._continuous(i))

    def flotation_payload(self, hour_index: int) -> dict:
        df = self.flot
        if not 3 <= hour_index < len(df):
            raise ToolError(f"hour number must be between 3 and {len(df) - 1}")
        if not self._continuous(hour_index):
            raise ToolError("this hour has no continuous 4-hour history, so no forecast can be made")
        i, idx = hour_index, df.index
        hours = []
        for j in (i - 2, i - 1, i):
            row = df.iloc[j].drop([TARGET, LEAK])                # 当前小时的化验结果线上拿不到
            hours.append({"hour_start": idx[j].isoformat(),
                          "features": {k: float(v) for k, v in row.items()}})
        # 上一小时(i-1)的化验值是已知的；当前小时(i)的化验值不使用
        return {"hours": hours, "prev_lab_silica": float(df.iloc[i - 1][TARGET])}


@dataclass
class Ctx:
    api_post: Callable[[str, dict], dict]
    data: DataSource
    orders_path: str = "orders/drafts.jsonl"
    results_dirs: tuple = ("models", "docs/results")          # 监控结果文件的查找目录


def get_geohazard_risk(ctx: Ctx, shift_id: int) -> dict:
    payload = ctx.data.seismic_payload(shift_id)
    r = ctx.api_post("/predict", payload)
    facts = {"shift_id": shift_id, "risk_score": round(r["risk_score"], 4),
             "risk_level": r["risk_level"], "alert_threshold": round(r["threshold"], 4),
             "indicators": {k: round(float(payload[k]), 2) for k in ("genergy", "gpuls", "nbumps")}}
    return {"facts": facts, "meta": {"model_version": r["model_version"]}}


def get_flotation_forecast(ctx: Ctx, hour_index: int) -> dict:
    r = ctx.api_post("/flotation/predict", ctx.data.flotation_payload(hour_index))
    facts = {"hour_index": hour_index,
             "predicted_silica_pct": round(r["predicted_silica_pct"], 2),
             "last_lab_silica_pct": round(r["prev_lab_silica_pct"], 2),
             "expected_change_pct_points": round(r["delta_pct"], 2),
             "method": "model" if r["model_used"] == "delta_rf" else "baseline_fallback"}
    gain = r.get("gate_rmse_improvement_holdout_pct")
    if gain is not None:                                       # 如实告知：模型只比基线好一点点
        facts["model_gain_vs_baseline_pct"] = round(gain, 1)
    return {"facts": facts, "meta": {"model_used": r["model_used"], "model_version": r["model_version"]}}


def _read_json(ctx: Ctx, name: str) -> dict:
    for d in ctx.results_dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise ToolError(f"{name} not found; run the monitoring scripts first")


def get_monitoring_status(ctx: Ctx) -> dict:
    fl = _read_json(ctx, "flotation_monitoring.json")
    gh = _read_json(ctx, "geohazard_monitoring.json")
    facts = {"flotation": {"retrain_recommended": fl["retrain_recommended"],
                           "investigate_drift": fl["investigate_drift"], "windows": fl["n_windows"]},
             "geohazard": {"recalibrate_recommended": gh["recalibrate_recommended"],
                           "investigate_drift": gh["investigate_drift"],
                           "mean_alert_rate_pct": round(gh["mean_alert_rate"] * 100, 1)}}
    return {"facts": facts, "meta": {}}


def draft_inspection_order(ctx: Ctx, shift_id: int, reason: str, created_by: str) -> dict:
    """起草检查工单：只生成"待人工批准"的草稿，不会发送或执行任何动作。
    策略：风险等级为NORMAL时不允许起草(避免噪声工单)；理由是自由文本，只当数据存储，不交给LLM。"""
    risk = get_geohazard_risk(ctx, shift_id)["facts"]
    if risk["risk_level"] == "NORMAL":
        return {"facts": {"shift_id": shift_id, "created": False, "risk_level": "NORMAL",
                          "risk_score": risk["risk_score"]}, "meta": {}}
    os.makedirs(os.path.dirname(ctx.orders_path) or ".", exist_ok=True)
    n = 1
    if os.path.exists(ctx.orders_path):
        with open(ctx.orders_path) as f:
            n = sum(1 for _ in f) + 1
    rec = {"order_no": n, "status": "DRAFT_PENDING_HUMAN_APPROVAL", "shift_id": shift_id,
           "risk_level": risk["risk_level"], "risk_score": risk["risk_score"],
           "reason": sanitize(reason)[:200], "created_by": created_by, "created_ts": round(time.time(), 3)}
    with open(ctx.orders_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"facts": {"shift_id": shift_id, "created": True, "order_no": n,
                      "risk_level": risk["risk_level"], "risk_score": risk["risk_score"]},
            "meta": {"order_no": n}}
