"""
train.py —— 模块1：岩爆(geohazard)风险预警模型训练

流程：
  读取数据 → 按时间切分(训练/测试) → 训练集内再按时间切(fit/calib)
  → 训练逻辑回归 → 在calib集上做Platt校准并按"代价"选阈值
  → 在测试集上评估(含 与"从不预警"基线的代价对比、top10% lift)
  → 记录到MLflow，保存模型包(模型+校准器+阈值+特征列表)

运行方式（项目根目录）：uv run python src/train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # 无界面环境下画图（WSL里没有显示器）
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import load_seismic, time_split

CAT = ["seismic", "seismoacoustic", "shift", "ghazard"]  # 类别型特征
COST_FN = 10.0  # 漏报一次岩爆的代价（业务假设，需与运营方确认，README中注明）
COST_FP = 1.0   # 误报一次（多做一次人工检查）的代价
# 预警预算：运营方每个周期最多愿意额外检查的班次比例（业务假设，README中注明）。
# 决策规则 = "风险分数排在前10%的班次触发预警"，比固定概率阈值更抗分布偏移。
ALERT_BUDGET = 0.10


def build_pipeline(feature_cols):
    """预处理 + 逻辑回归。把预处理和模型封进同一个Pipeline，
    服务端只要调用 pipeline.predict_proba，就保证和训练时用的是同一套预处理。"""
    num = [c for c in feature_cols if c not in CAT]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),  # 类别 → 0/1列
        ("num", StandardScaler(), num),                        # 数值标准化（逻辑回归需要）
    ])
    # class_weight="balanced"：正样本少，提高它的权重，避免模型全部预测"安全"
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    return Pipeline([("pre", pre), ("clf", clf)])


def best_threshold(y, p, c_fn, c_fp):
    """遍历阈值，选使"总代价 = c_fn*漏报数 + c_fp*误报数"最小的那个。
    阈值是业务决策：漏报比误报严重，所以通常远低于默认的0.5。"""
    grid = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in grid:
        pred = (p >= t).astype(int)
        fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        costs.append(c_fn * fn + c_fp * fp)
    return float(grid[int(np.argmin(costs))])


def lift_at(y, p, frac=0.1):
    """风险最高的frac比例的班次里，捕获了多少比例的真实岩爆。
    随机排序时期望值就是frac本身（如10%）；越高说明排序越有价值。"""
    k = max(1, int(len(y) * frac))
    top = np.argsort(-p)[:k]          # 按风险从高到低取前k个
    return float(y[top].sum() / max(1, y.sum()))


def plot_coefficients(pipe, path, top_n=15):
    """逻辑回归的可解释性：画出影响最大的特征系数（正=风险上升，负=风险下降）。"""
    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    idx = np.argsort(np.abs(coef))[-top_n:]  # 取绝对值最大的top_n个
    plt.figure(figsize=(8, 6))
    plt.barh(names[idx], coef[idx])
    plt.title("Logistic regression coefficients (standardized features)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    df = load_seismic()
    features = [c for c in df.columns if c != "class"]

    # 第一次切分：时间顺序，前70%为训练集，后30%为测试集（防止"偷看未来"）
    train_all, test = time_split(df, 0.3)
    # 第二次切分：训练集前80%训练模型，后20%用于校准与选阈值（这部分模型没见过）
    cut = int(len(train_all) * 0.8)
    fit, calib = train_all.iloc[:cut], train_all.iloc[cut:]

    pipe = build_pipeline(features)
    pipe.fit(fit[features], fit["class"])

    # Platt校准：用一个单变量逻辑回归把"模型原始分数"映射成更可信的概率
    raw_calib = pipe.predict_proba(calib[features])[:, 1].reshape(-1, 1)
    platt = LogisticRegression().fit(raw_calib, calib["class"])
    #thr = best_threshold(calib["class"].values,
    #                     platt.predict_proba(raw_calib)[:, 1], COST_FN, COST_FP)

    p_calib = platt.predict_proba(raw_calib)[:, 1]
    # 参考用：按"漏报/误报代价"选出的阈值（上次的做法，在小样本下不稳定，仅记录不使用）
    cost_thr = best_threshold(calib["class"].values, p_calib, COST_FN, COST_FP)
    # 实际使用：取校准集分数的 (1 - 预算) 分位数。
    # 例：预算10% → 取90%分位数，分数高于它的班次触发预警。
    thr = float(np.quantile(p_calib, 1 - ALERT_BUDGET))


    # 测试集评估
    raw_test = pipe.predict_proba(test[features])[:, 1].reshape(-1, 1)
    p_test = platt.predict_proba(raw_test)[:, 1]
    y_test = test["class"].values
    pred = (p_test >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    model_cost = COST_FN * fn + COST_FP * fp
    never_alert_cost = COST_FN * y_test.sum()  # 基线：从不预警，所有岩爆都漏报
    metrics = {
        "pr_auc": average_precision_score(y_test, p_test),
        "pr_auc_random": float(y_test.mean()),   # 随机猜的PR-AUC ≈ 正样本比例
        "roc_auc": roc_auc_score(y_test, p_test),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "lift_top10pct": lift_at(y_test, p_test, 0.10),  # 随机水平=0.10
        "lift_top20pct": lift_at(y_test, p_test, 0.20),  # 随机水平=0.20
        "threshold": thr,
        "cost_based_threshold": cost_thr,       # 仅供参考，对比两种阈值策略
        "alert_rate_test": float(pred.mean()),  # 测试集上实际触发预警的比例，应接近预算10%
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "model_cost": float(model_cost),
        "never_alert_cost": float(never_alert_cost),
    }
    print(json.dumps(metrics, indent=2))
    print("模型代价低于'从不预警'基线" if model_cost < never_alert_cost
          else "注意：模型代价未低于'从不预警'基线，应定位为风险排序而非自动决策")

    # 记录到MLflow（实验跟踪）
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("geohazard-seismic-bumps")
    with mlflow.start_run():
        mlflow.log_params({"model": "LogisticRegression", "C": 0.5,
                           "calibration": "platt", "cost_fn": COST_FN, "cost_fp": COST_FP,"alert_budget": ALERT_BUDGET})
        mlflow.log_metrics(metrics)
        plot_coefficients(pipe, "models/coefficients.png")
        mlflow.log_artifact("models/coefficients.png")

        # 模型包：把预处理+模型、校准器、阈值、特征列表一起保存，
        # 部署时直接加载这一个文件，避免训练与服务不一致
        joblib.dump({"pipeline": pipe, "platt": platt, "threshold": thr,
                     "features": features}, "models/geohazard.joblib")
        mlflow.log_artifact("models/geohazard.joblib")


if __name__ == "__main__":
    main()
