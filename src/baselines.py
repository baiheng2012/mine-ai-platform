"""
baselines.py —— 基线对比实验

目的：
1. 用"滚动时间窗口交叉验证"代替单次切分，得到更稳定的评估（单次切分在正样本很少时波动极大）
2. 对比 Dummy / 逻辑回归 / 随机森林 / 梯度提升，看复杂模型是否真的更好
3. 加入"历史滞后特征"（前几个班次的微震指标），验证时序信息有没有帮助
4. 对比"随机切分"与"时间切分"，量化数据泄漏造成的指标虚高
运行方式（在项目根目录）：uv run python src/baselines.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_seismic

CAT = ["seismic", "seismoacoustic", "shift", "ghazard"]  # 类别型特征，其余为数值型


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    给关键微震指标加"滞后特征"：当前班次之前第1/2/3个班次的值，以及最近5个班次的滚动均值。
    直觉：岩爆往往不是突然发生，之前的能量释放趋势可能是预警信号。
    注意：shift(k) 只用过去的数据，不会泄漏未来信息；开头几行没有历史，用0填充。
    假设：行与行是连续的班次记录（如果数据由多个工作面拼接，边界处会有少量噪声）。
    """
    out = df.copy()
    for col in ["genergy", "gpuls", "nbumps", "energy"]:
        for k in (1, 2, 3):
            out[f"{col}_lag{k}"] = out[col].shift(k)
        # 滚动均值先 shift(1)，确保窗口里不包含当前行
        out[f"{col}_roll5"] = out[col].shift(1).rolling(5).mean()
    return out.fillna(0)


def make_preprocessor(feature_cols):
    """类别特征做one-hot，数值特征做标准化（逻辑回归需要标准化，树模型不受影响）。"""
    num = [c for c in feature_cols if c not in CAT]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), num),
    ])


def get_models(feature_cols):
    """返回 {模型名: Pipeline}。每个模型都用同样的预处理，保证对比公平。"""
    def pipe(clf):
        return Pipeline([("pre", make_preprocessor(feature_cols)), ("clf", clf)])
    return {
        # Dummy：永远输出训练集的正样本比例，代表"什么都没学到"的下限
        "dummy": pipe(DummyClassifier(strategy="prior")),
        "logreg": pipe(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "rand_forest": pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1)),
        "hist_gb": pipe(HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200,
            class_weight="balanced", random_state=42)),
    }


def evaluate(model, X, y, splitter):
    """在 splitter 给出的每一折上训练并评估，返回 (ROC-AUC列表, PR-AUC列表, 测试集正样本比例列表)。"""
    roc, pr, base = [], [], []
    for tr, te in splitter.split(X, y):
        if y.iloc[te].sum() == 0:      # 测试折里没有正样本时AUC无定义，跳过
            continue
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        roc.append(roc_auc_score(y.iloc[te], p))
        pr.append(average_precision_score(y.iloc[te], p))
        base.append(y.iloc[te].mean())
    return roc, pr, base


def main():
    df = load_seismic()
    y = df["class"]
    feats_base = [c for c in df.columns if c != "class"]
    df_lag = add_lag_features(df)
    feats_lag = [c for c in df_lag.columns if c != "class"]

    # 滚动时间窗口：训练集总是在测试集之前，逐步扩大（共5折）
    ts = TimeSeriesSplit(n_splits=5)
    # 随机分层切分：仅用来演示"泄漏"带来的虚高，不能作为真实评估
    rnd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for feat_name, data, feats in [("base", df, feats_base), ("with_lags", df_lag, feats_lag)]:
        for name, model in get_models(feats).items():
            roc, pr, base = evaluate(model, data[feats], y, ts)
            rows.append({"features": feat_name, "model": name, "split": "time",
                         "roc_auc": np.mean(roc), "pr_auc": np.mean(pr),
                         "pr_auc_random_guess": np.mean(base)})

    # 泄漏演示：只用随机森林+基础特征，对比随机切分
    name = "rand_forest"
    roc, pr, base = evaluate(get_models(feats_base)[name], df[feats_base], y, rnd)
    rows.append({"features": "base", "model": name, "split": "random(leaky)",
                 "roc_auc": np.mean(roc), "pr_auc": np.mean(pr),
                 "pr_auc_random_guess": np.mean(base)})

    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    res.to_csv("models/baseline_results.csv", index=False)  # 存下来，README里直接引用


if __name__ == "__main__":
    main()
