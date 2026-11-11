"""从真实的小时级数据中取"4个连续整点小时"（其中3小时作为请求，第2个小时的化验值作为上一小时化验值），
输出成 /flotation/predict 的请求体JSON。"""
import json
import sys

import pandas as pd

sys.path.insert(0, "src")
from flotation_features import LEAK, TARGET

df = pd.read_csv("data/flotation_hourly.csv", parse_dates=["date"]).set_index("date").sort_index()
idx, one = df.index, pd.Timedelta(hours=1)

# 从最新数据往回找，第一个满足"连续4小时"的位置
pos = next(i for i in range(len(df) - 1, 2, -1)
           if idx[i] - idx[i - 1] == one and idx[i - 1] - idx[i - 2] == one and idx[i - 2] - idx[i - 3] == one)

hours = []
for j in (pos - 2, pos - 1, pos):
    row = df.iloc[j].drop([TARGET, LEAK])      # 去掉化验结果列（线上拿不到当前小时的化验值）
    hours.append({"hour_start": idx[j].isoformat(), "features": {k: float(v) for k, v in row.items()}})

# 当前小时(pos)的上一小时是pos-1，它的化验值是已知的
print(json.dumps({"hours": hours, "prev_lab_silica": float(df.iloc[pos - 1][TARGET])}))
