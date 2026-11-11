"""
train_flotation_final.py —— 模块2最终训练脚本（带晋级门槛）

模型：对"相对上一小时化验值的变化量(delta)"建模的随机森林，
      最终预测 = 上一小时化验值 + 模型预测的delta。
评估基准：persistence（直接沿用上一小时化验值）。
晋级门槛(promotion gate)：
  新模型的RMSE必须在 "交叉验证" 和 "留出集" 上都比persistence降低至少 GATE_MIN_IMPROVEMENT 个百分点，
  才允许保存为可部署模型；否则以退出码1结束（CI会因此失败）。
附加：用块自助法(block bootstrap)估计留出集上RMSE改进幅度的不确定性区间。

运行：uv run python src/train_flotation_final.py
"""
import json
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# 复用探索阶段已经写好的函数，保证口径一致
#from flotation_experiments import add_lag_features
from flotation_features import add_lag_features
from train_flotation import LEAK, PREV, TARGET, load, metrics

# 门槛参数（治理假设，需与运营方确认）：RMSE相对persistence至少改善3%
GATE_MIN_IMPROVEMENT = 3.0


def make_model():
    # min_samples_leaf较大+max_features较小：强正则化，防止在几千行数据上过拟合
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.3,
                                 n_jobs=-1, random_state=42)


def improvement_pct(rmse_model, rmse_base):
    """相对基线的RMSE改善百分比：正数=模型更好。"""
    return (1.0 - rmse_model / rmse_base) * 100.0


def block_bootstrap_improvement(err_model, err_base, block=24, n_boot=1000, seed=42):
    """块自助法：按"连续24小时"为单位有放回抽样，保留时间序列的自相关，
    估计改进幅度(%)的5%/50%/95%分位数。
    普通逐点自助法会假设样本独立，对小时级序列会低估不确定性。"""
    rng = np.random.default_rng(seed)
    n = len(err_model)
    n_blocks = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)          # 随机选若干块的起点
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        rm = np.sqrt(np.mean(err_model[idx] ** 2))
        rb = np.sqrt(np.mean(err_base[idx] ** 2))
        out.append(improvement_pct(rm, rb))
    return [float(v) for v in np.percentile(out, [5, 50, 95])]


def main():
    df = load()
    base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]
    # 只对"小时均值"传感器列构造滞后/变化量特征（__std列不做）
    lag_cols = [c for c in base_cols if not c.endswith("__std")]
    df = add_lag_features(df, lag_cols)
    feat_cols = [c for c in df.columns if c not in (TARGET, LEAK)]  # 含PREV、滞后值、变化量
    X, y, prev = df[feat_cols], df[TARGET], df[PREV]
    print(f"Samples: {len(df)}, features: {len(feat_cols)}")

    # 前80%：模型选择与交叉验证；后20%：只在最后评估一次的留出集
    cut = int(len(df) * 0.8)
    X_tr, y_tr, p_tr = X.iloc[:cut], y.iloc[:cut], prev.iloc[:cut]
    X_te, y_te, p_te = X.iloc[cut:], y.iloc[cut:], prev.iloc[cut:]

    # ---------- 1) 训练段上的时间序列交叉验证 ----------
    cv_model, cv_base = [], []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X_tr):
        m = make_model()
        m.fit(X_tr.iloc[tr], (y_tr - p_tr).iloc[tr])                   # 学的是delta
        pred = p_tr.iloc[te] + m.predict(X_tr.iloc[te])                # 还原成化验值水平
        cv_model.append(metrics(y_tr.iloc[te], pred)["rmse"])
        cv_base.append(metrics(y_tr.iloc[te], p_tr.iloc[te])["rmse"])
    imp_cv = improvement_pct(np.mean(cv_model), np.mean(cv_base))

    # ---------- 2) 留出集评估（只做这一次，避免对留出集反复调参） ----------
    m = make_model()
    m.fit(X_tr, y_tr - p_tr)
    pred_te = p_te + m.predict(X_te)
    hold_model = metrics(y_te, pred_te)
    hold_base = metrics(y_te, p_te)
    imp_hold = improvement_pct(hold_model["rmse"], hold_base["rmse"])
    mae_change = (hold_model["mae"] / hold_base["mae"] - 1.0) * 100.0   # 正数=MAE变差

    # ---------- 3) 不确定性：块自助法 ----------
    lo, mid, hi = block_bootstrap_improvement((y_te - pred_te).values, (y_te - p_te).values)

    # ---------- 4) 晋级门槛判定 ----------
    passed = bool(imp_cv >= GATE_MIN_IMPROVEMENT and imp_hold >= GATE_MIN_IMPROVEMENT)
    gate = {
        "passed": passed, "min_improvement_pct": GATE_MIN_IMPROVEMENT,
        "rmse_improvement_cv_pct": round(imp_cv, 2),
        "rmse_improvement_holdout_pct": round(imp_hold, 2),
        "holdout_rmse_model": round(hold_model["rmse"], 4),
        "holdout_rmse_persistence": round(hold_base["rmse"], 4),
        "holdout_mae_change_pct": round(mae_change, 2),
        "bootstrap_improvement_pct_p5_p50_p95": [round(v, 2) for v in (lo, mid, hi)],
        "n_holdout": int(len(y_te)),
    }
    print(json.dumps(gate, indent=2))
    with open("models/flotation_gate.json", "w") as f:
        json.dump(gate, f, indent=2)

    # ---------- 5) 记录到MLflow ----------
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flotation-soft-sensor")
    with mlflow.start_run(run_name="final_delta_rf"):
        mlflow.log_params({"model": "delta_RandomForest", "n_features": len(feat_cols),
                           "gate_min_improvement_pct": GATE_MIN_IMPROVEMENT})
        mlflow.log_metrics({"rmse_improvement_cv_pct": imp_cv,
                            "rmse_improvement_holdout_pct": imp_hold,
                            "holdout_rmse": hold_model["rmse"],
                            "holdout_mae_change_pct": mae_change,
                            "bootstrap_p5": lo, "bootstrap_p95": hi,
                            "gate_passed": float(passed)})
        mlflow.log_artifact("models/flotation_gate.json")

    # ---------- 6) 门槛不通过：不保存可部署模型，以失败退出 ----------
    if not passed:
        print("GATE FAILED: model does not beat the persistence baseline by the required margin. "
              "No deployable model saved.")
        sys.exit(1)

    # ---------- 7) 通过：用全部数据重新训练再保存（常规做法：评估用留出集，部署用全量） ----------
    final = make_model()
    final.fit(X, y - prev)
    joblib.dump({"model": final, "features": feat_cols, "target": TARGET,
                 "lag_cols": lag_cols, "gate": gate}, "models/flotation.joblib")
    # 参考分布：后面做数据漂移监控时，拿线上数据和它比较
    X.to_csv("models/flotation_reference.csv")
    print("GATE PASSED: deployable model saved to models/flotation.joblib")


if __name__ == "__main__":
    main()
