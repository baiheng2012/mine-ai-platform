"""
drift.py —— 数据漂移检测的通用函数（只依赖numpy/pandas，不依赖训练库）

核心指标 PSI = sum((当前占比 - 参考占比) * ln(当前占比 / 参考占比))
  - 连续特征：按参考数据的分位数分10箱
  - 取值很少的特征(<=10个不同值，如类别、计数)：直接按取值统计占比，出现新取值也能被发现
经验阈值：<0.1 稳定；0.1~0.25 轻微变化；>0.25 显著变化（经验规则，非统计检验）
"""
import numpy as np
import pandas as pd

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def _to_proportions(counts, eps):
    """计数转占比；用下限eps避免占比为0时出现log(0)。"""
    p = counts / max(counts.sum(), 1.0)
    return np.clip(p, eps, None)


def psi(ref, cur, bins=10, max_categories=10, eps=1e-4):
    """计算一个特征的PSI；任何一边没有有效数据则返回NaN。"""
    ref = pd.Series(ref).dropna()
    cur = pd.Series(cur).dropna()
    if ref.empty or cur.empty:
        return float("nan")

    if ref.nunique() <= max_categories:
        # 类别型/低基数特征：按取值统计，参考集里没出现过的新取值占比下限为eps，PSI会很大
        cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        rc = ref.value_counts().reindex(cats, fill_value=0).values.astype(float)
        cc = cur.value_counts().reindex(cats, fill_value=0).values.astype(float)
    else:
        # 连续特征：用参考数据的分位数作为内部切分点；两端开放，当前数据超出参考范围也会被计入首尾箱
        inner = np.unique(np.quantile(ref.values, np.linspace(0, 1, bins + 1))[1:-1])
        rc = np.bincount(np.searchsorted(inner, ref.values, side="right"),
                         minlength=len(inner) + 1).astype(float)
        cc = np.bincount(np.searchsorted(inner, cur.values, side="right"),
                         minlength=len(inner) + 1).astype(float)

    p, q = _to_proportions(rc, eps), _to_proportions(cc, eps)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(ref_df, cur_df, cols):
    """逐特征计算PSI，按从大到小排序，便于定位"是哪些特征变了"。"""
    rows = [{"feature": c, "psi": psi(ref_df[c], cur_df[c])} for c in cols]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drifted_share(table, thr=PSI_SIGNIFICANT):
    """PSI超过显著阈值的特征所占比例（单个特征波动很常见，看"比例"更稳健）。"""
    return float((table["psi"] > thr).mean())


def status_from_share(share, warn=0.2, drift=0.4):
    """窗口状态：显著漂移的特征占比>=40% 为DRIFT，>=20% 为WARN，否则OK（治理参数，需与运营方确认）。"""
    if share >= drift:
        return "DRIFT"
    if share >= warn:
        return "WARN"
    return "OK"


def consecutive_true(flags, n=2):
    """是否存在连续n个以上为True的窗口。用"连续"而不是"单次"触发，避免偶发波动导致频繁重训练。"""
    run = 0
    for f in flags:
        run = run + 1 if f else 0
        if run >= n:
            return True
    return False
