"""
edge_model.py —— 边缘端神经网络模型定义（训练、导出、基准测试共用）

关键设计：把"预处理"(稳健缩放+裁剪)做成模型的一部分(buffer)，随模型一起导出到ONNX。
好处：边缘端只需喂入原始特征，不可能出现"训练和部署用了不同缩放参数"的skew，
      这与前面服务里"共用特征函数"是同一个思路。
"""
import copy
import torch
from torch import nn


class DeltaNet(nn.Module):
    """预测"相对上一小时化验值的变化量(delta)"的多层感知机。
    center/scale：训练集上的中位数与四分位距(稳健缩放参数)，作为buffer保存(不参与训练，但随模型保存/导出)。
    clip：缩放后裁剪到[-clip, clip]，防止传感器异常值(如之前观察到的z分数爆炸)把网络带偏。"""

    def __init__(self, center, scale, hidden=128, depth=2, dropout=0.2, clip=5.0):
        super().__init__()
        n_in = len(center)
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.clip = clip
        layers, d = [], n_in
        for _ in range(depth):                       # depth个"线性层+ReLU+Dropout"
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, 1))               # 输出一个标量：delta
        self.net = nn.Sequential(*layers)

    def forward(self, x):                            # x: (batch, n_in) 原始特征
        z = torch.clamp((x - self.center) / self.scale, -self.clip, self.clip)
        return self.net(z).squeeze(-1)               # (batch,)


def export_onnx(model, n_in, path):
    """导出ONNX：batch维度设为动态，边缘端可以按需喂1条或多条。"""
    model = copy.deepcopy(model).eval().cpu()
    dummy = torch.zeros(1, n_in, dtype=torch.float32)
    kwargs = dict(input_names=["features"], output_names=["delta"],
                  dynamic_axes={"features": {0: "batch"}, "delta": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kwargs)   # 经典导出路径，依赖少、稳定
    except TypeError:                                                   # 较旧版本的torch没有dynamo参数
        torch.onnx.export(model, dummy, path, **kwargs)
