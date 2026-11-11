"""
flotation_features.py —— 浮选模型的特征构造（训练与线上服务共用）

注意：这个模块只能依赖pandas，不能引入matplotlib/mlflow等训练专用库，
因为线上服务镜像里没有它们。
"""
import re

import pandas as pd

TARGET = "% Silica Concentrate"   # 目标：精矿二氧化硅含量
LEAK = "% Iron Concentrate"       # 结果类变量：预测时拿不到，不能当特征
PREV = "prev_lab"                 # 上一小时的化验值

# 滞后值/变化量特征的列名后缀，如 "Starch Flow__lag1"、"Starch Flow__d1"
_DERIVED = re.compile(r"__(lag\d+|d1)$")


def add_lag_features(df, cols, lags=(1, 2)):
    """给指定列添加：滞后值 lagK、以及相对上一小时的变化量 d1。
    只有当前行与第k行之前恰好相差k小时才有效，遇到时间缺口则置为NaN，最后丢弃含NaN的行。
    （与 flotation_experiments.py 里的同名函数逻辑一致；以后以这里为准）"""
    out = df.copy()
    idx = df.index.to_series()
    for k in lags:
        ok = (idx - idx.shift(k)) == pd.Timedelta(hours=k)
        for c in cols:
            out[f"{c}__lag{k}"] = df[c].shift(k).where(ok)
    ok1 = (idx - idx.shift(1)) == pd.Timedelta(hours=1)
    for c in cols:
        out[f"{c}__d1"] = (df[c] - df[c].shift(1)).where(ok1)
    return out.dropna()


def base_columns(features):
    """从模型的完整特征列表里，还原出"原始小时级特征"（去掉PREV和滞后/变化量衍生列）。"""
    return [f for f in features if f != PREV and not _DERIVED.search(f)]
