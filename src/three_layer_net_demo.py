"""
三层神经网络：基本架构与信号传递（可运行示例）

---------------------------------------------------------------------------
「三层」指什么
---------------------------------------------------------------------------
在入门教程里，**三层神经网络**通常指三层 **神经元**：

  输入层 → 隐藏层 → 输出层

对应 **两组可训练权重**（两个 Linear / 全连接层）：

  输入向量 x → 线性变换 + 偏置 → 隐藏激活 → 线性变换 + 偏置 → logits

若把 Flatten 也算作一层「处理」，则数据流仍是：图像张量 → 向量 → 隐藏 → 输出。
**注意**：有的书用「层数」专指隐藏层个数，说法不一；本文件采用「输入 / 隐藏 / 输出」三层节点划分。

---------------------------------------------------------------------------
信号传递（前向，批大小为 N）
---------------------------------------------------------------------------
设展平后输入 X 形状为 (N, 784)（如 MNIST）。

1. z₁ = X W₁ᵀ + b₁     → 形状 (N, H)，**预激活 / logits of hidden**
2. h = σ(z₁)           → 形状 (N, H)，**隐藏层表示**（σ 常为 ReLU）
3. z₂ = h W₂ᵀ + b₂     → 形状 (N, C)，**输出 logits**（多类分类常 **不** 在最后接 Softmax，
   与 `nn.CrossEntropyLoss` 配合，损失内部在 log 域计算）

反向传播时梯度从 loss 经 z₂、h、z₁ 传回 W₂、b₂、W₁、b₁（及输入侧若需要）。

---------------------------------------------------------------------------
与 `src/model.py` 的关系
---------------------------------------------------------------------------
项目中的 `MLP` 即为：**Flatten → Linear → ReLU → Linear**，与本文件「三层结构」一致；
本 demo 把中间张量 **逐步打印**，便于对照形状与数据流。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.three_layer_net_demo
---------------------------------------------------------------------------
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _ensure_utf8_stdio() -> None:
    import sys

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


class ThreeLayerMLP(nn.Module):
    """
    输入层(784) → 隐藏层(H) → 输出层(C)，与 `MLP` 同构，但 forward 分步便于教学。
    """

    def __init__(self, in_dim: int = 784, hidden_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.flatten(x))))


def trace_forward(model: ThreeLayerMLP, x: torch.Tensor) -> None:
    """逐步打印各阶段张量形状与简单统计量。"""
    _print_header("信号传递：从输入到 logits（逐步）")

    x0 = x
    print(f"0) 原始输入 x0        形状 {tuple(x0.shape)}  (N, 1, 28, 28)")

    x1 = model.flatten(x0)
    print(f"1) Flatten 后         形状 {tuple(x1.shape)}  (N, 784)")

    z1 = model.fc1(x1)
    print(f"2) z1 = fc1(x1)       形状 {tuple(z1.shape)}  (N, H) 预激活")
    print(f"      z1 均值/标准差: {z1.mean().item():.4f} / {z1.std().item():.4f}")

    h = model.act(z1)
    print(f"3) h = ReLU(z1)       形状 {tuple(h.shape)}  (N, H) 隐藏层激活")
    print(f"      h 非零比例(示意): {(h > 0).float().mean().item():.4f}")

    logits = model.fc2(h)
    print(f"4) logits = fc2(h)    形状 {tuple(logits.shape)}  (N, C) 输出 logits")
    print(f"      logits 均值: {logits.mean().item():.4f}")

    print()
    print("说明：训练时 CrossEntropyLoss(logits, y) 内部对 logits 做 log_softmax；")
    print("推理时常用 argmax(logits, dim=1) 作为预测类别。")


def demo_batch_matrix_view(device: torch.device) -> None:
    """单样本时矩阵乘法视角：x 行向量 · Wᵀ。"""
    _print_header("矩阵视角（单样本 N=1）")

    in_dim, h_dim, c = 784, 32, 10
    x = torch.randn(1, in_dim, device=device)
    W1 = torch.randn(h_dim, in_dim, device=device)
    b1 = torch.randn(h_dim, device=device)
    z1 = x @ W1.T + b1
    print(f"x 形状 (1, {in_dim})，W1 形状 ({h_dim}, {in_dim})")
    print(f"z1 = x @ W1.T + b1  → 形状 {tuple(z1.shape)}，即 (1, {h_dim})")
    print("一批 N 个样本时，x 为 (N, 784)，同一套 W1、b1 并行作用于每一行。")


def demo_match_project_mlp(device: torch.device) -> None:
    """与 `src.model.MLP` 输出对齐（同结构、同随机种子时）。"""
    _print_header("与 src.model.MLP 数值一致（同权重拷贝）")

    from src.model import MLP

    hidden_dim = 32
    torch.manual_seed(0)
    t3 = ThreeLayerMLP(hidden_dim=hidden_dim, num_classes=10).to(device)
    torch.manual_seed(0)
    mlp = MLP(hidden_dim=hidden_dim, num_classes=10).to(device)

    t3.fc1.weight.data.copy_(mlp.net[1].weight.data)
    t3.fc1.bias.data.copy_(mlp.net[1].bias.data)
    t3.fc2.weight.data.copy_(mlp.net[3].weight.data)
    t3.fc2.bias.data.copy_(mlp.net[3].bias.data)

    x = torch.randn(4, 1, 28, 28, device=device)
    y1 = t3(x)
    y2 = mlp(x)
    err = (y1 - y2).abs().max().item()
    print(f"max |ThreeLayerMLP(x) - MLP(x)|: {err:.3e}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.manual_seed(42)
    model = ThreeLayerMLP(hidden_dim=64, num_classes=10).to(device)
    x = torch.randn(8, 1, 28, 28, device=device)
    trace_forward(model, x)

    demo_batch_matrix_view(device)
    demo_match_project_mlp(device)

    _print_header("ASCII 结构示意")
    print(
        "  [Batch,1,28,28]\n"
        "       │ Flatten\n"
        "       ▼\n"
        "  [Batch, 784] ──► Linear(784→H) ──► [Batch, H] ──► ReLU ──► [Batch, H]\n"
        "                                                              │\n"
        "                                                              ▼\n"
        "                                                    Linear(H→C)\n"
        "                                                              │\n"
        "                                                              ▼\n"
        "                                                    [Batch, C] logits\n"
    )


if __name__ == "__main__":
    main()
