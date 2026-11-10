"""
monitor_flotation.py —— 浮选模型的漂移监控（离线回放）

回放设定：
  - 参考期 = 前60%的小时数据；用它训练模型(相当于"模型在这个时间点部署")
  - 监控期 = 后40%，按168小时(一周)切成窗口，逐窗口检查：
      1) 输入漂移：各传感器特征相对参考期的PSI，显著漂移特征占比
      2) 模型表现：化验值到达后，模型RMSE相对persistence基线的改善幅度
  - 触发规则：连续2个窗口状态为DRIFT，或连续2个窗口模型不如基线 → 建议重训练
输出：models/flotation_monitoring.json、models/flotation_drift.png，并记录到MLflow
运行：uv run python src/monitor_flotation.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from drift import PSI_SIGNIFICANT, consecutive_true, drift_table, drifted_share, status_from_share
from flotation_features import LEAK, PREV, TARGET, add_lag_features
from train_flotation import load, metrics
from train_flotation_final import GATE_MIN_IMPROVEMENT, make_model

REF_FRAC = 0.6      # 前60%作为参考/训练期
WINDOW = 168        # 窗口长度：168小时 = 1周
CONSECUTIVE = 2     # 连续多少个窗口异常才触发


def main():
    df = load()
    base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]   # 原始小时级传感器特征
    lag_cols = [c for c in base_cols if not c.endswith("__std")]
    df = add_lag_features(df, lag_cols)
    feat_cols = [c for c in df.columns if c not in (TARGET, LEAK)]

    cut = int(len(df) * REF_FRAC)
    ref, mon = df.iloc[:cut], df.iloc[cut:]
    print(f"Reference hours: {len(ref)}, monitoring hours: {len(mon)}, window: {WINDOW}h")

    # 用参考期训练模型（与最终训练脚本相同的模型定义）
    model = make_model()
    model.fit(ref[feat_cols], ref[TARGET] - ref[PREV])

    rows = []
    for w, start in enumerate(range(0, len(mon) - WINDOW + 1, WINDOW)):   # 不足一个窗口的尾部丢弃
        cur = mon.iloc[start:start + WINDOW]
        tab = drift_table(ref, cur, base_cols)
        share = drifted_share(tab)
        pred = cur[PREV] + model.predict(cur[feat_cols])
        rmse_m = metrics(cur[TARGET], pred)["rmse"]
        rmse_b = metrics(cur[TARGET], cur[PREV])["rmse"]
        rows.append({
            "window": w, "start": str(cur.index[0]), "end": str(cur.index[-1]),
            "drifted_share": share, "status": status_from_share(share),
            "top_drift": ", ".join(f"{r.feature}({r.psi:.2f})" for r in tab.head(3).itertuples()),
            "rmse_model": rmse_m, "rmse_persistence": rmse_b,
            "improvement_pct": (1 - rmse_m / rmse_b) * 100,
        })
    res = pd.DataFrame(rows)
    print(res.drop(columns=["start", "end"]).round(3).to_string(index=False))

    # 触发规则
    drift_trigger = consecutive_true((res["status"] == "DRIFT").tolist(), CONSECUTIVE)
    perf_trigger = consecutive_true((res["improvement_pct"] < 0).tolist(), CONSECUTIVE)
    """
    decision = {
        "retrain_recommended": bool(drift_trigger or perf_trigger),
        "drift_trigger": bool(drift_trigger), "performance_trigger": bool(perf_trigger),
        "rule": f"{CONSECUTIVE} consecutive windows with status DRIFT, or model worse than persistence",
        "n_windows": int(len(res)),
        "windows_by_status": res["status"].value_counts().to_dict(),
    }
    """
    decision = {
        # 重训练只由"模型表现"触发；输入漂移只触发"排查"。
        # 原因：回放中输入漂移持续存在(9个窗口全部DRIFT)，但模型仍优于基线；
        # 若漂移也触发重训练，会每周无谓重训练(告警疲劳)。
        "retrain_recommended": bool(perf_trigger),
        "investigate_drift": bool(drift_trigger),
        "drift_trigger": bool(drift_trigger), "performance_trigger": bool(perf_trigger),
        "rule": f"retrain: model worse than persistence for {CONSECUTIVE} consecutive windows; "
                f"investigate: {CONSECUTIVE} consecutive windows with status DRIFT",
        "n_windows": int(len(res)),
        "windows_by_status": res["status"].value_counts().to_dict(),
    }

    print(json.dumps(decision, indent=2))
    with open("models/flotation_monitoring.json", "w") as f:
        json.dump(decision, f, indent=2)
    res.to_csv("models/flotation_monitoring_windows.csv", index=False)

    # 图：上=显著漂移特征占比；下=模型相对基线的改善幅度
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].bar(res["window"], res["drifted_share"] * 100)
    ax[0].axhline(40, color="r", ls="--", label="DRIFT threshold (40%)")
    ax[0].axhline(20, color="orange", ls="--", label="WARN threshold (20%)")
    ax[0].set_ylabel(f"% features with PSI>{PSI_SIGNIFICANT}")
    ax[0].legend()
    ax[1].bar(res["window"], res["improvement_pct"])
    ax[1].axhline(0, color="k")
    ax[1].axhline(GATE_MIN_IMPROVEMENT, color="g", ls="--", label="promotion gate")
    ax[1].set_ylabel("RMSE improvement vs persistence (%)")
    ax[1].set_xlabel("weekly window after deployment")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("models/flotation_drift.png", dpi=150)
    plt.close()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("monitoring-flotation")
    with mlflow.start_run():
        mlflow.log_metrics({"max_drifted_share": float(res["drifted_share"].max()),
                            "mean_improvement_pct": float(res["improvement_pct"].mean()),
                            "retrain_recommended": float(decision["retrain_recommended"])})
        mlflow.log_artifact("models/flotation_monitoring.json")
        mlflow.log_artifact("models/flotation_drift.png")


if __name__ == "__main__":
    main()
