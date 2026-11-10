"""
monitor_geohazard.py —— 岩爆预警模型的漂移与预警比例监控（离线回放）

回放设定：参考期=训练段(前70%)；监控期=后30%，每100个班次一个窗口。
每个窗口检查：
  1) 输入漂移：各特征PSI
  2) 分数漂移：模型风险分数相对参考期的PSI
  3) 预警比例：窗口内分数>=阈值的比例，应接近预警预算(10%)；
     连续2个窗口落在[预算/2, 预算*2]之外 → 建议重新校准阈值或重训练
     （我们之前在测试集上观察到实际预警比例只有2.8%，这里把它系统化）
注意：窗口只有100个样本时PSI的随机波动较大，且本数据集"是否严格按时间排列"我并不确定，
      结论要谨慎表述。
运行：uv run python src/monitor_geohazard.py   （需要先运行 train.py 产出 models/geohazard.joblib）
"""
import json

import joblib
import mlflow
import pandas as pd

from data import load_seismic, time_split
from drift import PSI_SIGNIFICANT, consecutive_true, drift_table, drifted_share, psi, status_from_share
from train import ALERT_BUDGET

WINDOW = 100
CONSECUTIVE = 2


def main():
    df = load_seismic()
    features = [c for c in df.columns if c != "class"]
    train_all, test = time_split(df, 0.3)
    b = joblib.load("models/geohazard.joblib")
    thr = b["threshold"]

    def scores(d):
        """与线上服务一致：模型原始分数 -> Platt校准后的风险分数。"""
        raw = b["pipeline"].predict_proba(d[features])[:, 1].reshape(-1, 1)
        return b["platt"].predict_proba(raw)[:, 1]

    ref_scores = scores(train_all)
    print(f"Reference rows: {len(train_all)}, monitoring rows: {len(test)}, window: {WINDOW}, "
          f"threshold: {thr:.4f}, alert budget: {ALERT_BUDGET:.0%}")

    rows = []
    for w, start in enumerate(range(0, len(test) - WINDOW + 1, WINDOW)):
        cur = test.iloc[start:start + WINDOW]
        s = scores(cur)
        tab = drift_table(train_all, cur, features)
        share = drifted_share(tab)
        alert_rate = float((s >= thr).mean())
        rows.append({
            "window": w, "drifted_share": share, "status": status_from_share(share),
            "score_psi": psi(ref_scores, s),
            "alert_rate": alert_rate,
            "alert_rate_out_of_range": not (ALERT_BUDGET / 2 <= alert_rate <= ALERT_BUDGET * 2),
            "event_rate": float(cur["class"].mean()),     # 实际岩爆事件比例(事后才知道)
            "top_drift": ", ".join(f"{r.feature}({r.psi:.2f})" for r in tab.head(3).itertuples()),
        })
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    drift_trigger = consecutive_true((res["status"] == "DRIFT").tolist(), CONSECUTIVE)
    alert_trigger = consecutive_true(res["alert_rate_out_of_range"].tolist(), CONSECUTIVE)
    
    """
    decision = {
        "retrain_recommended": bool(drift_trigger or alert_trigger),
        "drift_trigger": bool(drift_trigger), "alert_rate_trigger": bool(alert_trigger),
        "rule": f"{CONSECUTIVE} consecutive windows with status DRIFT, or alert rate outside "
                f"[{ALERT_BUDGET / 2:.0%}, {ALERT_BUDGET * 2:.0%}]",
        "n_windows": int(len(res)),
        "mean_alert_rate": float(res["alert_rate"].mean()),
    }
    """
    decision = {
        # 预警比例持续偏离预算 → 阈值失准，建议重新校准阈值；输入漂移只触发排查
        "recalibrate_recommended": bool(alert_trigger),
        "investigate_drift": bool(drift_trigger),
        "drift_trigger": bool(drift_trigger), "alert_rate_trigger": bool(alert_trigger),
        "rule": f"recalibrate: alert rate outside [{ALERT_BUDGET / 2:.0%}, {ALERT_BUDGET * 2:.0%}] "
                f"for {CONSECUTIVE} consecutive windows; investigate: {CONSECUTIVE} consecutive DRIFT windows",
        "n_windows": int(len(res)),
        "mean_alert_rate": float(res["alert_rate"].mean()),
    }        



    print(json.dumps(decision, indent=2))
    with open("models/geohazard_monitoring.json", "w") as f:
        json.dump(decision, f, indent=2)
    res.to_csv("models/geohazard_monitoring_windows.csv", index=False)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("monitoring-geohazard")
    """
    with mlflow.start_run():
        mlflow.log_metrics({"mean_alert_rate": decision["mean_alert_rate"],
                            "max_drifted_share": float(res["drifted_share"].max()),
                            "retrain_recommended": float(decision["retrain_recommended"])})
        mlflow.log_artifact("models/geohazard_monitoring.json")
    """
    with mlflow.start_run():
        mlflow.log_metrics({"mean_alert_rate": decision["mean_alert_rate"],
                            "max_drifted_share": float(res["drifted_share"].max()),
                            "recalibrate_recommended": float(decision["recalibrate_recommended"])})
        mlflow.log_artifact("models/geohazard_monitoring.json")



if __name__ == "__main__":
    main()
