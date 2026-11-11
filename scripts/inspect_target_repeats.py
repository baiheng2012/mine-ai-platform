"""诊断：目标变量(精矿二氧化硅)是否经常与上一小时完全相同？
以及模型相对persistence的优势，在"目标不变的小时"和"目标变化的小时"上是否不同。
注意：按"目标是否变化"分组用到了真实标签，只能用于事后理解，不能用于线上决策。"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from flotation_features import LEAK, PREV, TARGET, add_lag_features
from train_flotation import load, metrics
from train_flotation_final import make_model

# ---- 1) 各月"与上一小时相同"的比例（用容差1e-9，避免浮点均值的微小误差） ----
raw = pd.read_csv("data/flotation_hourly.csv", parse_dates=["date"]).set_index("date").sort_index()
same = pd.DataFrame({c: raw[c].diff().abs() < 1e-9 for c in [TARGET, LEAK]}).resample("MS").mean() * 100
print("Share of hours (%) where the lab value equals the previous hour, by month:")
print(same.round(1).to_string())

# ---- 2) 留出集上，按"目标是否变化"分组比较模型与persistence ----
df = load()
base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]
df = add_lag_features(df, [c for c in base_cols if not c.endswith("__std")])
feat_cols = [c for c in df.columns if c not in (TARGET, LEAK)]
cut = int(len(df) * 0.8)
tr, te = df.iloc[:cut], df.iloc[cut:]

m = make_model()
m.fit(tr[feat_cols], tr[TARGET] - tr[PREV])           # 与最终训练脚本相同：学习delta
pred = te[PREV] + m.predict(te[feat_cols])

changed = (te[TARGET] - te[PREV]).abs() > 1e-9         # 当前小时的化验值相对上一小时是否变化
print(f"\nHoldout hours: {len(te)}, share with changed target: {changed.mean():.1%}")
for name, mask in [("unchanged hours", ~changed), ("changed hours", changed)]:
    if mask.sum() == 0:
        continue
    mm = metrics(te.loc[mask, TARGET], pred[mask])
    pb = metrics(te.loc[mask, TARGET], te.loc[mask, PREV])
    print(f"{name:16s} n={int(mask.sum()):4d} | model RMSE {mm['rmse']:.3f} MAE {mm['mae']:.3f} "
          f"| persistence RMSE {pb['rmse']:.3f} MAE {pb['mae']:.3f}")
