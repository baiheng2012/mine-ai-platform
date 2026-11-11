"""诊断：浮选数据的"漂移"究竟是什么？对比参考期和监控期的关键特征分布，并看进料品位的月度变化。"""
import pandas as pd

df = pd.read_csv("data/flotation_hourly.csv", parse_dates=["date"]).set_index("date").sort_index()
cut = int(len(df) * 0.6)                      # 与监控脚本相同的60%切分（近似；监控脚本里还丢弃了少量行）
ref, mon = df.iloc[:cut], df.iloc[cut:]
print("Reference period:", ref.index.min(), "->", ref.index.max())
print("Monitoring period:", mon.index.min(), "->", mon.index.max())

cols = ["% Iron Feed", "% Silica Feed", "Ore Pulp Flow", "Ore Pulp pH",
        "Flotation Column 04 Air Flow", "Flotation Column 05 Air Flow",
        "Flotation Column 07 Air Flow__std"]
summary = pd.DataFrame({
    "ref_p10": ref[cols].quantile(0.1), "ref_median": ref[cols].median(), "ref_p90": ref[cols].quantile(0.9),
    "mon_p10": mon[cols].quantile(0.1), "mon_median": mon[cols].median(), "mon_p90": mon[cols].quantile(0.9),
})
print("\nDistribution summary (10th / 50th / 90th percentile):")
print(summary.round(2).to_string())

# 进料品位的月度中位数：看是渐变还是突变（配矿/矿源切换通常表现为台阶式变化）
print("\nMonthly median of feed grades:")
print(df[["% Iron Feed", "% Silica Feed"]].resample("MS").median().round(2).to_string())

# 传感器"冻结"检查：整小时内几乎不变(标准差≈0)的比例，参考期 vs 监控期
for c in ["Flotation Column 04 Air Flow", "Flotation Column 05 Air Flow"]:
    s = c + "__std"
    print(f"\n{c}: share of hours with ~zero within-hour std -> "
          f"reference {(ref[s] < 1e-9).mean():.1%}, monitoring {(mon[s] < 1e-9).mean():.1%}")
# ---- 追加检查：是否存在大段"数值与上一小时完全相同"(卡死/前向填充)的时期 ----
print("\nShare of hours (%) where the value is identical to the previous hour, by month:")
chk = ["% Iron Feed", "% Silica Feed", "Flotation Column 04 Air Flow", "Flotation Column 05 Air Flow"]
same = pd.DataFrame({c: (df[c].diff() == 0) for c in chk}).resample("MS").mean()
print((same * 100).round(1).to_string())

# 参考期里最常见的取值及占比：若某个值占比很高，说明分位数分箱会出现大量重复边界
print("\nMost common feed-grade values in the reference period (share of hours, %):")
for c in ["% Iron Feed", "% Silica Feed"]:
    print(c)
    print((ref[c].round(2).value_counts(normalize=True).head(3) * 100).round(1).to_string())
