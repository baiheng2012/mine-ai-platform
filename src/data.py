import pandas as pd
from scipy.io import arff

def load_seismic(path="data/seismic-bumps.arff") -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    # ARFF中的名义变量读出来是bytes，统一解码
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")
    df["class"] = df["class"].astype(int)  # 1 = 高能量微震(岩爆风险)
    return df

def time_split(df: pd.DataFrame, test_frac=0.3):
    """数据按时间顺序排列，所以按时间切分，避免用未来数据训练。"""
    n = len(df)
    cut = int(n * (1 - test_frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
