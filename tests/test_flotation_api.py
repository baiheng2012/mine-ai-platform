"""
浮选接口测试，重点：
  - 正常预测 / 各类非法输入返回422
  - 降级机制：浮选模型缺失时退回基线并标注
  - 训练/服务特征一致性(skew)测试：同一个时间点，服务端构造的特征必须与训练路径完全一致
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import app, build_flotation_features, state, FlotationRequest
from flotation_features import add_lag_features
from train_flotation import load      # 训练路径的数据加载（需要先运行过prepare_flotation）


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:        # with触发lifespan，加载模型
        yield c


@pytest.fixture()
def sample():
    with open("sample_flotation.json") as f:
        return json.load(f)


def test_flotation_ok(client, sample):
    r = client.post("/flotation/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "delta_rf"          # 门槛通过后应使用真实模型
    assert body["predicted_silica_pct"] >= 0
    assert body["model_version"]
    # 预测应该在上一小时化验值附近（修正量有限），防止明显的单位/特征错位
    assert abs(body["delta_pct"]) < 3


def test_non_consecutive_hours_rejected(client, sample):
    t = pd.Timestamp(sample["hours"][1]["hour_start"]) - pd.Timedelta(hours=3)
    sample["hours"][1]["hour_start"] = t.isoformat()
    assert client.post("/flotation/predict", json=sample).status_code == 422


def test_missing_feature_rejected(client, sample):
    del sample["hours"][2]["features"]["Starch Flow"]
    assert client.post("/flotation/predict", json=sample).status_code == 422


def test_wrong_number_of_hours_rejected(client, sample):
    sample["hours"] = sample["hours"][:2]
    assert client.post("/flotation/predict", json=sample).status_code == 422


def test_negative_prev_lab_rejected(client, sample):
    sample["prev_lab_silica"] = -1
    assert client.post("/flotation/predict", json=sample).status_code == 422


def test_fallback_to_persistence(client, sample, monkeypatch):
    # 模拟"没有通过门槛的模型"：把浮选模型置空，接口应降级为基线并标注
    monkeypatch.setitem(state, "flotation", None)
    r = client.post("/flotation/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "persistence_fallback"
    assert body["predicted_silica_pct"] == pytest.approx(sample["prev_lab_silica"])
    assert body["delta_pct"] == 0.0


def test_no_training_serving_skew(client, sample):
    """服务端用请求构造的特征，应与训练流程对同一时间点算出的特征一致。"""
    bundle = state["flotation"]
    # 服务路径
    X_serving = build_flotation_features(FlotationRequest(**sample), bundle)
    # 训练路径：load()加载小时级数据并加prev_lab，再统一构造滞后特征
    ts = pd.Timestamp(sample["hours"][-1]["hour_start"])
    train_df = add_lag_features(load(), bundle["lag_cols"])
    X_training = train_df.loc[[ts], bundle["features"]]
    assert list(X_serving.columns) == list(X_training.columns)
    assert np.allclose(X_serving.values.astype(float), X_training.values.astype(float))
