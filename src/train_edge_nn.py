"""
train_edge_nn.py —— 浮选delta模型的神经网络版本：训练 → 评估(同样的晋级门槛口径) → 导出ONNX → 数值对拍

评估口径与 train_flotation_final.py 一致：时间切分，最后20%为留出集，对比persistence；
晋级门槛不通过时，仍会导出ONNX用于延迟基准测试，但明确标注"不可部署"。
运行：uv run --group edge python src/train_edge_nn.py
"""
import json

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from edge_model import DeltaNet, export_onnx
from flotation_features import LEAK, PREV, TARGET, add_lag_features
from train_flotation import load, metrics
from train_flotation_final import GATE_MIN_IMPROVEMENT

SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # ---------- 数据：与最终训练脚本完全相同的特征与切分 ----------
    df = load()
    base_cols = [c for c in df.columns if c not in (TARGET, LEAK, PREV)]
    df = add_lag_features(df, [c for c in base_cols if not c.endswith("__std")])
    feat_cols = [c for c in df.columns if c not in (TARGET, LEAK)]
    X = df[feat_cols].values.astype("float32")
    prev = df[PREV].values.astype("float32")
    y = df[TARGET].values.astype("float32")
    delta = y - prev                                  # 学习目标：变化量

    cut = int(len(df) * 0.8)                          # 前80%训练+验证，后20%留出
    val_cut = int(cut * 0.85)                         # 训练段内部再留最后15%用于早停(按时间，不打乱)
    X_tr, d_tr = X[:val_cut], delta[:val_cut]
    X_va, d_va = X[val_cut:cut], delta[val_cut:cut]
    X_te, y_te, p_te = X[cut:], y[cut:], prev[cut:]

    # 稳健缩放参数只用训练段计算(防止泄漏)；四分位距过小的特征(如冻结的传感器)改用1，避免除以接近0
    center = np.median(X_tr, axis=0)
    iqr = np.subtract(*np.percentile(X_tr, [75, 25], axis=0))
    scale = np.where(iqr < 1e-6, 1.0, iqr).astype("float32")

    model = DeltaNet(center, scale, hidden=128, depth=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.MSELoss()
    t = lambda a: torch.as_tensor(a, device=device)   # numpy -> 张量(放到训练设备)
    Xtr_t, dtr_t, Xva_t = t(X_tr), t(d_tr), t(X_va)

    # ---------- 训练：mini-batch + 早停(验证集RMSE连续20轮不下降就停止，恢复最佳权重) ----------
    best, best_state, patience, bad = float("inf"), None, 20, 0
    for epoch in range(300):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss_fn(model(Xtr_t[idx]), dtr_t[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(loss_fn(model(Xva_t), t(d_va))))
        if val_rmse < best - 1e-5:
            best, bad = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    print(f"Stopped at epoch {epoch}, best validation delta-RMSE: {best:.4f}")

    # ---------- 留出集评估：与persistence比较 ----------
    model.eval()
    with torch.no_grad():
        pred = p_te + model(t(X_te)).cpu().numpy()
    m, base = metrics(y_te, pred), metrics(y_te, p_te)
    imp = (1 - m["rmse"] / base["rmse"]) * 100
    passed = bool(imp >= GATE_MIN_IMPROVEMENT)
    print(f"Holdout RMSE model {m['rmse']:.4f} vs persistence {base['rmse']:.4f} "
          f"-> improvement {imp:.2f}% (gate {GATE_MIN_IMPROVEMENT}%): {'PASS' if passed else 'FAIL'}")
    print(f"Holdout MAE model {m['mae']:.4f} vs persistence {base['mae']:.4f}")
    if not passed:
        print("NOTE: this network is NOT deployable; the ONNX file is exported for latency benchmarking only.")

    # ---------- 导出ONNX，并与PyTorch逐点对拍(确认导出没有改变模型行为) ----------
    export_onnx(model, X.shape[1], "models/flotation_edge.onnx")
    sess = ort.InferenceSession("models/flotation_edge.onnx", providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"features": X_te})[0]
    with torch.no_grad():
        torch_out = model(t(X_te)).cpu().numpy()
    max_diff = float(np.abs(onnx_out - torch_out).max())
    print(f"PyTorch vs ONNX Runtime (CPU) max abs difference: {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX output differs from PyTorch"

    meta = {"features": feat_cols, "n_features": len(feat_cols), "gate_passed": passed,
            "holdout_rmse_improvement_pct": round(imp, 2), "onnx_max_abs_diff": max_diff,
            "deployable": passed}
    with open("models/flotation_edge_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models/flotation_edge.onnx and models/flotation_edge_meta.json")


if __name__ == "__main__":
    main()
