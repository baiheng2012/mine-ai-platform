"""
API测试。

关键点：用 pytest fixture 管理 TestClient 的生命周期。
  - fixture(scope="module")：整个文件只启动一次应用（只加载一次模型），所有测试共用
  - yield 之前 = 启动(触发lifespan，加载模型)；yield 之后 = 关闭(触发lifespan收尾)
  - 这样测试运行期间，应用一直处于"已启动"状态，state里的模型始终可用

前提：需要先有 models/geohazard.joblib（即先跑过train.py）和 sample.json。
"""
import json

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture(scope="module")
def client():
    # with 语句保证lifespan被触发；测试期间应用保持启动状态
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample():
    """每个测试拿到一份独立的请求体副本，修改它不会影响其他测试。"""
    with open("sample.json") as f:
        return json.load(f)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # 之前缺了这一条：模型版本不为空，才说明模型真的加载成功了
    assert body["model_version"] is not None


def test_predict_ok(client, sample):
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0                       # 分数是合法概率
    assert body["risk_level"] in {"NORMAL", "ELEVATED", "ALERT"}  # 等级是合法取值
    assert body["model_version"]                                  # 返回了模型版本


def test_invalid_category_rejected(client, sample):
    sample["shift"] = "X"                                         # 非法类别
    assert client.post("/predict", json=sample).status_code == 422


def test_negative_value_rejected(client, sample):
    sample["genergy"] = -1                                        # 能量不可能为负
    assert client.post("/predict", json=sample).status_code == 422


def test_missing_field_rejected(client, sample):
    del sample["nbumps"]                                          # 缺少必填字段
    assert client.post("/predict", json=sample).status_code == 422
