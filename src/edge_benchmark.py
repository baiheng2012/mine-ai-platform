"""
edge_benchmark.py —— 多后端推理延迟基准测试

后端：torch_cpu / torch_cuda / ort_cpu / ort_cuda / ort_trt(TensorRT，可选)
模型：
  - trained_h128：真实训练的小模型(models/flotation_edge.onnx)
  - synthetic_h1024x6：合成的大模型(随机权重，只用于测延迟，不是真实模型！)，
    用来找到"GPU/TensorRT开始划算"的规模交叉点
批大小：1(边缘实时推理)、64、1024、8192(批量处理)
公平性：所有后端都测"numpy输入 -> numpy输出"的端到端时间，包含主机↔GPU数据拷贝。
同时报告每个后端相对ort_cpu的最大绝对误差(FP16的TensorRT会有小误差，需要关注)。

运行：uv run --group edge python src/edge_benchmark.py
"""
import os
import time
import copy
import numpy as np
import pandas as pd
import torch                      # 注意：必须在onnxruntime之前导入，torch会预加载ONNX Runtime需要的CUDA库
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from edge_model import DeltaNet, export_onnx

N_IN = 104                        # 特征数(与训练脚本一致)
BATCHES = [1, 64, 1024, 8192]
WARMUP, ITERS = 20, 200
HAS_CUDA = torch.cuda.is_available()
PROVIDERS = ort.get_available_providers()


def timeit(fn, batch):
    """返回(p50毫秒, p95毫秒)。大批量时减少迭代次数，避免跑太久。"""
    iters = ITERS if batch <= 1024 else 50
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 95))


def make_trt_session(path):
    """TensorRT执行提供器(FP16)。TensorRT需要知道动态batch的范围，这里给出min/opt/max形状。
    引擎构建耗时较长，缓存到models/trt_cache，第二次运行会快很多。"""
    os.makedirs("models/trt_cache", exist_ok=True)
    opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "models/trt_cache",
            "trt_profile_min_shapes": f"features:1x{N_IN}",
            "trt_profile_opt_shapes": f"features:64x{N_IN}",
            "trt_profile_max_shapes": f"features:8192x{N_IN}"}
    s = ort.InferenceSession(path, providers=[("TensorrtExecutionProvider", opts),
                                              "CUDAExecutionProvider", "CPUExecutionProvider"])
    return s if s.get_providers()[0] == "TensorrtExecutionProvider" else None   # 回退了就不算TensorRT


def build_backends(name, onnx_path, torch_model):
    """返回 {后端名: 函数(numpy输入x)->numpy输出}；某个后端不可用就跳过并打印原因。"""
    be = {}
    #be["torch_cpu"] = lambda x, m=torch_model.cpu().eval(): m(torch.from_numpy(x)).detach().numpy()
    #if HAS_CUDA:
    #    mc = torch_model.to("cuda").eval()
    m_cpu = copy.deepcopy(torch_model).cpu().eval()

    def torch_cpu(x):
        with torch.no_grad():
            return m_cpu(torch.from_numpy(x)).numpy()
    be["torch_cpu"] = torch_cpu
    if HAS_CUDA:
        mc = copy.deepcopy(torch_model).to("cuda").eval()        
        
        def torch_cuda(x):
            with torch.no_grad():
                return mc(torch.from_numpy(x).to("cuda")).cpu().numpy()   # 含H2D与D2H拷贝，保证公平
        be["torch_cuda"] = torch_cuda
    
    
    cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    be["ort_cpu"] = lambda x, s=cpu: s.run(None, {"features": x})[0]
    if "CUDAExecutionProvider" in PROVIDERS:
        cu = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if cu.get_providers()[0] == "CUDAExecutionProvider":
            be["ort_cuda"] = lambda x, s=cu: s.run(None, {"features": x})[0]
        else:
            print(f"[{name}] ort_cuda skipped: CUDA provider failed to load (see warnings above)")
    if "TensorrtExecutionProvider" in PROVIDERS:
        try:
            trt = make_trt_session(onnx_path)
            if trt is not None:
                be["ort_trt_fp16"] = lambda x, s=trt: s.run(None, {"features": x})[0]
            else:
                print(f"[{name}] ort_trt skipped: session fell back (TensorRT libraries not found?)")
        except Exception as e:
            print(f"[{name}] ort_trt skipped: {type(e).__name__}: {str(e)[:200]}")
    else:
        print(f"[{name}] ort_trt skipped: TensorrtExecutionProvider not available")
    return be


def main():
    print(f"CUDA available for torch: {HAS_CUDA}; ONNX Runtime providers: {PROVIDERS}")
    if HAS_CUDA:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    rng = np.random.default_rng(0)

    # 小模型：用训练好的真实ONNX；PyTorch侧用同结构随机权重仅用于torch后端的延迟对比(结构相同，延迟可比)
    small = DeltaNet(np.zeros(N_IN), np.ones(N_IN), hidden=128, depth=2)
    big = DeltaNet(np.zeros(N_IN), np.ones(N_IN), hidden=1024, depth=6)
    export_onnx(big, N_IN, "models/synthetic_big.onnx")
    models = {"trained_h128": ("models/flotation_edge.onnx", small),
              "synthetic_h1024x6": ("models/synthetic_big.onnx", big)}

    rows = []
    for name, (path, tmodel) in models.items():
        be = build_backends(name, path, tmodel)
        for b in BATCHES:
            x = rng.normal(size=(b, N_IN)).astype("float32")
            ref = be["ort_cpu"](x)                                   # 以ort_cpu的输出作为精度参照
            for bname, fn in be.items():
            #    p50, p95 = timeit(lambda: fn(x), b)
            #    err = float(np.abs(np.asarray(fn(x)) - ref).max())
                p50, p95 = timeit(lambda: fn(x), b)
                # trained_h128 的 torch 后端用的是随机权重(与ONNX中的训练权重不同)，与ort_cpu比较误差没有意义，记为NaN；
                # 基于同一个ONNX文件的后端(ort_*)，以及合成模型(同一个模型对象导出)的误差是有意义的
                comparable = bname.startswith("ort") or name.startswith("synthetic")
                err = float(np.abs(np.asarray(fn(x)) - ref).max()) if comparable else float("nan")
            
                rows.append({"model": name, "backend": bname, "batch": b,
                             "p50_ms": p50, "p95_ms": p95,
                             "samples_per_s": b / (p50 / 1000), "max_abs_err_vs_ort_cpu": err})
                print(f"{name:18s} {bname:13s} batch={b:5d} p50={p50:8.3f} ms p95={p95:8.3f} ms")

    res = pd.DataFrame(rows)
    res.to_csv("models/edge_benchmark.csv", index=False)
    print("\n" + res.round(4).to_string(index=False))

    # 图：每个模型一张子图，横轴batch(对数)，纵轴p50延迟(对数)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, g) in zip(axes, res.groupby("model")):
        for bname, gg in g.groupby("backend"):
            ax.plot(gg["batch"], gg["p50_ms"], marker="o", label=bname)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(name); ax.set_xlabel("batch size"); ax.set_ylabel("p50 latency (ms)")
        ax.legend()
    plt.tight_layout()
    plt.savefig("models/edge_benchmark.png", dpi=150)
    print("Saved models/edge_benchmark.csv and models/edge_benchmark.png")


if __name__ == "__main__":
    main()
