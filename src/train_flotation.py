"""
train_flotation.py —— 模块2：浮选二氧化硅含量软测量

目标变量：% Silica Concentrate（精矿中的二氧化硅杂质含量，越低越好）

评估设计（核心）：
  1. 时间序列交叉验证(TimeSeriesSplit)：训练集永远在测试集之前
  2. 对比基线：
       - mean：永远预测训练集均值（"什么都没学到"）
       - persistence：直接沿用上一小时的化验值（软测量领域的强基线）
  3. 两种泄漏演示：随机切分；把"% Iron Concentrate"当特征
     (铁精矿含量与硅含量是同一次化验的两个结果，实际预测时拿不到，属于"结果类变量")
  4. "有无上一小时化验值"两种场景：
       process_only：只用实时传感器（化验结果出来之前就能预测，才是软测量的真正价值）
       with_prev_lab：额外用上一小时化验值（假设化验结果能及时拿到）
最后：选出最佳"仅传感器"模型，在留出的最后20%时间上评估，记录到MLflow，保存模型包。

运行：uv run python src/train_flotation.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/flotation_hourly.csv"
TARGET = "% Silica Concentrate"
LEAK = "% Iron Concentrate"      # 结果类变量：与目标同一次化验产生，预测时不可得
PREV = "prev_lab"                # 上一小时的目标化验值


def load():
    df = pd.read_csv(DATA, parse_dates=["date"]).set_index("date").sort_index()
    # 上一小时的化验值：只有当前行与上一行恰好相差1小时才有效，有时间缺口则置为NaN
    gap_ok = df.index.to_series().diff() == pd.Timedelta("1h")
    df[PREV] = df[TARGET].shift(1).where(gap_ok)
    df = df.dropna(subset=[PREV])  # 丢掉没有上一小时值的行，保证所有方案在同一批样本上比较
    return df


def metrics(y, p):
    return {"rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p))}


def cv_eval(make_model, X, y, splitter):
    """在splitter的每一折上训练+评估，返回各折指标的平均值。"""
    rows = []
    for tr, te in splitter.split(X):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[te], m.predict(X.iloc[te])))
    return pd.DataFrame(rows).mean().to_dict()


MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "rand_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                 n_jobs=-1, random_state=42),
    "hist_gb": lambda: HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05,
                                                     max_iter=300, random_state=42),
}


def main():
    df = load()
    y = df[TARGET]
    base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]  # 仅传感器特征
    X_proc = df[base_cols]
    X_lab = df[base_cols + [PREV]]
    print(f"样本数 {len(df)}，特征数(仅传感器) {len(base_cols)}")

    # 留出最后20%时间作为最终测试集；前80%用于模型选择(交叉验证)
    cut = int(len(df) * 0.8)
    tr_idx, te_idx = df.index[:cut], df.index[cut:]
    ts = TimeSeriesSplit(n_splits=5)

    rows = []
    # --- 基线 ---
    rows.append({"variant": "mean_baseline", "model": "dummy",
                 **cv_eval(lambda: DummyRegressor(), X_proc.loc[tr_idx], y.loc[tr_idx], ts)})
    # persistence不需要训练：直接用prev_lab当预测值，在同样的折上评估
    pers = []
    Xl, yl = X_lab.loc[tr_idx], y.loc[tr_idx]
    for _, te in ts.split(Xl):
        pers.append(metrics(yl.iloc[te], Xl.iloc[te][PREV]))
    rows.append({"variant": "persistence", "model": "prev_lab", **pd.DataFrame(pers).mean().to_dict()})

    # --- 正式方案：仅传感器 / 加上一小时化验值 ---
    for variant, X in [("process_only", X_proc), ("with_prev_lab", X_lab)]:
        for name, mk in MODELS.items():
            rows.append({"variant": variant, "model": name,
                         **cv_eval(mk, X.loc[tr_idx], y.loc[tr_idx], ts)})

    # --- 泄漏演示（只用hist_gb，仅用于说明问题，不能作为真实结果） ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 随机切分：相邻小时被拆到两边
    rows.append({"variant": "LEAK_random_split", "model": "hist_gb",
                 **cv_eval(MODELS["hist_gb"], X_proc.loc[tr_idx], y.loc[tr_idx], kf)})
    X_leak = df[base_cols + [LEAK]]                        # 把结果类变量当特征
    rows.append({"variant": "LEAK_iron_concentrate", "model": "hist_gb",
                 **cv_eval(MODELS["hist_gb"], X_leak.loc[tr_idx], y.loc[tr_idx], ts)})

    res = pd.DataFrame(rows).round(4)
    print(res.to_string(index=False))
    res.to_csv("models/flotation_cv_results.csv", index=False)

    # --- 选择最佳"仅传感器"模型（按交叉验证RMSE） ---
    cand = res[res["variant"] == "process_only"].sort_values("rmse")
    best_name = cand.iloc[0]["model"]
    print(f"\n最佳仅传感器模型：{best_name}")

    # --- 留出集最终评估 ---
    model = MODELS[best_name]()
    model.fit(X_proc.loc[tr_idx], y.loc[tr_idx])
    p = model.predict(X_proc.loc[te_idx])
    final = metrics(y.loc[te_idx], p)
    final_mean = metrics(y.loc[te_idx], np.full(len(te_idx), y.loc[tr_idx].mean()))
    final_pers = metrics(y.loc[te_idx], X_lab.loc[te_idx, PREV])
    print("留出集 模型   :", json.dumps({k: round(v, 4) for k, v in final.items()}))
    print("留出集 均值基线:", json.dumps({k: round(v, 4) for k, v in final_mean.items()}))
    print("留出集 沿用上小时:", json.dumps({k: round(v, 4) for k, v in final_pers.items()}))

    # --- 特征重要性：排列重要性(打乱某特征看误差上升多少)，是模型无关的可解释方法 ---
    pi = permutation_importance(model, X_proc.loc[te_idx], y.loc[te_idx], n_repeats=5,
                                random_state=42, scoring="neg_root_mean_squared_error")
    order = np.argsort(pi.importances_mean)[-15:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(base_cols)[order], pi.importances_mean[order])
    plt.title("Permutation importance (holdout, process-only model)")
    plt.tight_layout()
    plt.savefig("models/flotation_importance.png", dpi=150)
    plt.close()

    # --- 记录到MLflow并保存模型包 ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("flotation-soft-sensor")
    with mlflow.start_run():
        mlflow.log_params({"model": best_name, "target": TARGET, "n_features": len(base_cols)})
        mlflow.log_metrics({f"holdout_{k}": v for k, v in final.items()})
        mlflow.log_metrics({f"holdout_mean_baseline_{k}": v for k, v in final_mean.items()})
        mlflow.log_metrics({f"holdout_persistence_{k}": v for k, v in final_pers.items()})
        mlflow.log_artifact("models/flotation_cv_results.csv")
        mlflow.log_artifact("models/flotation_importance.png")
        # 训练集特征样本，留作后面"数据漂移监控"的参考分布
        X_proc.loc[tr_idx].to_csv("models/flotation_reference.csv")
        joblib.dump({"model": model, "features": base_cols, "target": TARGET},
                    "models/flotation.joblib")
        mlflow.log_artifact("models/flotation.joblib")


if __name__ == "__main__":
    main()
