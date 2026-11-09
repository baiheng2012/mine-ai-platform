"""从真实数据取一行，打印成API请求体JSON，用于手动测试 /predict。"""
import json
import sys
sys.path.insert(0, "src")          # 让脚本能 import src/data.py
from data import load_seismic

df = load_seismic()
# 取测试期（后30%）里的第一行；去掉标签列 class（API不接收标签）
row = df.iloc[int(len(df) * 0.7)].drop("class").to_dict()
# numpy类型转成Python原生类型，json才能序列化
row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
print(json.dumps(row))
