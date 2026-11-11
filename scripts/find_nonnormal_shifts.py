"""列出风险等级不是NORMAL的班次编号，用来演示ELEVATED/ALERT和工单起草。"""
import sys

sys.path.insert(0, "src")
from fastapi.testclient import TestClient

from app import app
from assistant.tools import DataSource

ds = DataSource()
with TestClient(app) as c:
    hits = []
    for i in range(1808, 2584):                       # 数据集后30%(测试段)
        r = c.post("/predict", json=ds.seismic_payload(i)).json()
        if r["risk_level"] != "NORMAL":
            hits.append((i, r["risk_level"], round(r["risk_score"], 3)))
print(f"{len(hits)} non-NORMAL shifts; first ones (shift, level, score):")
print(hits[:12])
