"""drift.py 的单元测试：用已知分布验证PSI的行为，固定随机种子保证可复现。"""
import numpy as np
import pandas as pd

from drift import consecutive_true, psi, status_from_share


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=2000), rng.normal(size=2000)) < 0.1     # 同分布：应当稳定


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=2000), rng.normal(loc=1.5, size=2000)) > 0.25   # 均值平移1.5个标准差


def test_psi_detects_new_category():
    ref = pd.Series(["a"] * 90 + ["b"] * 10)
    cur = pd.Series(["a"] * 50 + ["b"] * 10 + ["c"] * 40)               # 出现了参考期没有的类别c
    assert psi(ref, cur) > 0.25


def test_psi_empty_returns_nan():
    assert np.isnan(psi(pd.Series([1.0, 2.0, 3.0]), pd.Series([], dtype=float)))


def test_consecutive_true():
    assert consecutive_true([False, True, True], 2)
    assert not consecutive_true([True, False, True], 2)               # 不连续不触发


def test_status_thresholds():
    assert status_from_share(0.1) == "OK"
    assert status_from_share(0.25) == "WARN"
    assert status_from_share(0.5) == "DRIFT"
