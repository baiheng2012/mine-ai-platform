"""
flotation_experiments.py —— 模块2的最后一轮限时实验

要回答的问题：
  Q1. ridge为什么爆炸？(诊断：训练集里近似常数的特征，在测试期被放大成巨大的z分数)
  Q2. 加入"传感器滞后/变化量特征"并对"相对上一小时的变化量"建模，
      能否打赢"直接沿用上一小时化验值"(persistence)？

做法：
  - 目标改成 delta = 当前化验值 - 上一小时化验值；最终预测 = 上一小时化验值 + 模型预测的delta
  - 若模型预测的delta恒为0，就退化为persistence，所以persistence就是天然的比较对象
  - 评估方式与train_flotation.py一致：时间序列交叉验证 + 最后20%留出集

运行：uv run python src/flotation_experiments.py
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

# 复用上一个脚本里的数据加载与指标函数（保证两个脚本口径一致）
from train_flotation import LEAK, PREV, TARGET, load, metrics


def add_lag_features(df, cols, lags=(1, 2)):
    """给传感器"小时均值"列添加：滞后值 lagK、以及相对上一小时的变化量 d1。
    浮选是有滞后的过程(矿浆在槽里停留需要时间)，上游几小时前的工况可能影响现在的品位。
    只有当前行与第k行之前恰好相差k小时才有效，遇到时间缺口置为NaN，最后统一丢弃。"""
    out = df.copy()
    idx = df.index.to_series()
    for k in lags:
        ok = (idx - idx.shift(k)) == pd.Timedelta(hours=k)   # 检查时间是否连续
        for c in cols:
            out[f"{c}__lag{k}"] = df[c].shift(k).where(ok)
    ok1 = (idx - idx.shift(1)) == pd.Timedelta(hours=1)
    for c in cols:
        out[f"{c}__d1"] = (df[c] - df[c].shift(1)).where(ok1)  # 当前值-上一小时值
    return out.dropna()


def diagnose_scaling(X_train, X_test, top=6):
    """诊断：用训练集的均值/标准差去标准化测试集，看哪些特征的|z|异常大。
    若某特征在训练期几乎不变(如传感器冻结)，测试期一变化，z分数就会爆炸，线性模型随之失控。"""
    sc = StandardScaler().fit(X_train)
    z = np.abs(sc.transform(X_test))
    worst = pd.Series(z.max(axis=0), index=X_train.columns).sort_values(ascending=False)
    print("\n[诊断] 测试期里|z|最大的特征（数值上百说明存在训练/测试分布严重不一致）：")
    print(worst.head(top).round(1).to_string())
    zero_var = [c for c in X_train.columns if X_train[c].std() < 1e-9]
    print(f"[诊断] 训练期方差为0的特征：{zero_var}")


def cv_delta(make_model, X, y, prev, splitter):
    """对delta=y-prev建模，用 prev+预测delta 作为最终预测，在每折上评估。"""
    rows = []
    for tr, te in splitter.split(X):
        m = make_model()
        m.fit(X.iloc[tr], (y - prev).iloc[tr])
        pred = prev.iloc[te] + m.predict(X.iloc[te])
        rows.append(metrics(y.iloc[te], pred))
    return pd.DataFrame(rows).mean().to_dict()


# 强正则化的模型：样本只有几千行、特征几十到上百个，必须防过拟合
DELTA_MODELS = {
    "delta_mean": lambda: DummyRegressor(),   # 预测"平均变化量"，约等于persistence
    "delta_ridge_robust": lambda: make_pipeline(   # 稳健缩放(基于分位数)+裁剪到[-5,5]，抵御z分数爆炸
        RobustScaler(), FunctionTransformer(lambda a: np.clip(a, -5, 5)), Ridge(alpha=300.0)),
    "delta_rand_forest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=20, max_features=0.3, n_jobs=-1, random_state=42),
    "delta_hist_gb": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.03, max_iter=150, l2_regularization=5.0, random_state=42),
}


def main():
    df = load()
    base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]
    # 只对"小时均值"列(不含__std、不含化验类)做滞后特征；这里包含进料品位两列，它们是上游输入
    mean_cols = [c for c in base_cols if not c.endswith("__std")]
    df = add_lag_features(df, mean_cols)
    feat_cols = [c for c in df.columns if c not in (TARGET, LEAK)]   # 含PREV(当前水平)和滞后/变化量特征
    X, y, prev = df[feat_cols], df[TARGET], df[PREV]
    print(f"样本数 {len(df)}，特征数 {len(feat_cols)}")

    cut = int(len(df) * 0.8)
    tr_idx, te_idx = df.index[:cut], df.index[cut:]
    ts = TimeSeriesSplit(n_splits=5)

    # Q1：诊断ridge爆炸（只用原始传感器特征，复现之前的设定）
    diagnose_scaling(df.loc[tr_idx, base_cols], df.loc[te_idx, base_cols])

    # Q2：对delta建模，与persistence比较
    pers = pd.DataFrame([metrics(y.loc[tr_idx].iloc[te], prev.loc[tr_idx].iloc[te])
                         for _, te in ts.split(X.loc[tr_idx])]).mean().to_dict()
    rows = [{"model": "persistence", **pers}]
    for name, mk in DELTA_MODELS.items():
        rows.append({"model": name, **cv_delta(mk, X.loc[tr_idx], y.loc[tr_idx], prev.loc[tr_idx], ts)})
    res = pd.DataFrame(rows)
    # 相对persistence的RMSE变化：负数=比persistence好
    res["rmse_vs_persistence_%"] = (res["rmse"] / pers["rmse"] - 1) * 100
    print("\n[交叉验证，仅训练段(前80%)]")
    print(res.round(4).to_string(index=False))

    # 留出集：只评估"CV里最好的delta模型"，同时给出persistence对照
    best = res[res["model"].str.startswith("delta_")].sort_values("rmse").iloc[0]["model"]
    m = DELTA_MODELS[best]()
    m.fit(X.loc[tr_idx], (y - prev).loc[tr_idx])
    pred = prev.loc[te_idx] + m.predict(X.loc[te_idx])
    print(f"\n[留出集] 最佳delta模型 = {best}")
    print("模型      :", {k: round(v, 4) for k, v in metrics(y.loc[te_idx], pred).items()})
    print("persistence:", {k: round(v, 4) for k, v in metrics(y.loc[te_idx], prev.loc[te_idx]).items()})


if __name__ == "__main__":
    main()
