"""
prepare_flotation.py —— 浮选数据预处理：把20秒级原始数据聚合为小时级

为什么要聚合：
  原始数据每20秒一行(约73万行)，但化验值("% Silica Concentrate"等)每小时才测一次，
  同一小时内的约180行化验值完全相同。如果直接把这些行随机划分到训练/测试集，
  同一小时的数据会同时出现在两边——这是严重的数据泄漏，指标会虚高。
  聚合成小时级后，一小时就是一个样本，才能做公平评估。

输出：data/flotation_hourly.csv（体积小，可以提交到Git，供CI使用）
运行：uv run python src/prepare_flotation.py [原始csv路径]
"""
import sys

import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw/MiningProcess_Flotation_Plant_Database.csv"
OUT = "data/flotation_hourly.csv"

# 化验类列：每小时才有一个值，聚合时取均值即可（同一小时内本来就相同）
LAB_COLS = ["% Iron Feed", "% Silica Feed", "% Iron Concentrate", "% Silica Concentrate"]


def main():
    # decimal="," ：这个文件用逗号当小数点
    df = pd.read_csv(RAW, decimal=",")
    df.columns = [c.strip() for c in df.columns]      # 去掉列名首尾空格，避免隐蔽的KeyError
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    missing = [c for c in LAB_COLS if c not in df.columns]
    assert not missing, f"找不到这些列，请核对列名：{missing}\n实际列名：{list(df.columns)}"
    print(f"原始数据：{df.shape[0]} 行, {df.shape[1]} 列, 时间 {df.index.min()} ~ {df.index.max()}")

    # 过程传感器列 = 除化验列以外的所有列（流量、pH、密度、气流、液位等）
    process_cols = [c for c in df.columns if c not in LAB_COLS]

    # 按小时分组：floor("h") 把时间截断到整点
    g = df.groupby(df.index.floor("h"))
    mean = g.mean()                                           # 每个小时所有列的均值
    # 过程变量再加"小时内标准差"：反映这一小时工况是否稳定（波动大本身可能影响品位）
    std = g[process_cols].std().add_suffix("__std")
    hourly = pd.concat([mean, std], axis=1).dropna()
    hourly.index.name = "date"

    # 检查时间缺口：真实数据常有停机/缺失，后面做滞后特征和时间切分要注意
    gaps = hourly.index.to_series().diff()
    big = gaps[gaps > pd.Timedelta("1h")]
    print(f"小时级样本：{len(hourly)} 行；时间缺口(>1h)共 {len(big)} 处")
    if len(big):
        print(big.sort_values(ascending=False).head(5).to_string())

    hourly.to_csv(OUT)
    print(f"已保存 {OUT}")


if __name__ == "__main__":
    main()
