"""
ReLU（Rectified Linear Unit，整流线性单元）：说明 + 可运行示例

---------------------------------------------------------------------------
概念摘要（与代码中的 demo 对应）
---------------------------------------------------------------------------

1. 定义
   ReLU(x) = max(0, x) = x 当 x ≥ 0，否则为 0。
   分段线性：负半轴为常数 0，正半轴为恒等映射。

2. 导数（次梯度）
   x < 0：0；x > 0：1；x = 0：经典分析中不可导，实现中常取 0 或 1（PyTorch 在 0 处
   的行为以实现为准，见下方 demo）。反向传播时负半轴梯度为 0。

3. 优点（实践中广泛使用）
   - 计算简单、无昂贵 exp。
   - 正半轴梯度恒为 1，减轻 Sigmoid/Tanh 在饱和区的梯度消失问题。
   - 输出稀疏（大量神经元可被置 0），有一定正则化意味。

4. 缺点与变体
   - **神经元死亡（dying ReLU）**：若某神经元长期接收负输入，则恒输出 0 且梯度为 0，
     参数不再更新。缓解：**LeakyReLU / PReLU**（负半轴小斜率）、**ELU**、更好的初始化、
     较小学习率、**BatchNorm** 等。
   - 输出非零中心（均为 ≥0），后续层可学偏置补偿；深层网络中常与 BN、残差等配合。

5. 与项目代码的关系
   `src/model.py` 中 MLP 在隐藏层使用 `nn.ReLU(inplace=True)`。

6. PyTorch
   `torch.relu`、`F.relu`、`nn.ReLU`；就地版本 `inplace=True` 可省显存但会覆盖输入，
   若后续计算仍需要原张量则勿用 inplace。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.relu_demo
---------------------------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def relu_manual(x: torch.Tensor) -> torch.Tensor:
    """ReLU(x)=max(0,x)，与 torch.relu 等价（教学对照用）。"""
    return torch.maximum(x, torch.zeros_like(x))


def relu_grad_heuristic(x: torch.Tensor) -> torch.Tensor:
    """示意：(ReLU'(x)) 在 x>0 为 1，x<0 为 0；x=0 处这里取 0（与常见实现一致）。"""
    return (x > 0).to(x.dtype)


def demo_match_torch(device: torch.device) -> None:
    _print_header("1. max(0,x) 与 torch.relu / F.relu / nn.ReLU 一致")

    x = torch.tensor([-2.0, -0.0, 0.0, 0.0, 1.5, 3.0], device=device)
    y_m = relu_manual(x)
    y_t = torch.relu(x)
    y_f = F.relu(x)
    y_n = nn.ReLU()(x)
    print(f"max |manual - torch.relu|: {(y_m - y_t).abs().max().item():.3e}")
    print(f"max |F.relu - torch.relu|: {(y_f - y_t).abs().max().item():.3e}")
    print(f"max |nn.ReLU - torch.relu|: {(y_n - y_t).abs().max().item():.3e}")
    print(f"x:       {x.cpu().numpy().tolist()}")
    print(f"ReLU(x): {y_t.cpu().numpy().round(6).tolist()}")


def demo_derivative_at_points(device: torch.device) -> None:
    _print_header("2. 梯度：负半轴为 0，正半轴为 1；x=0 处见打印")

    # 单点分别反传，避免同一 x 多次 backward 混淆
    points = [-1.0, 0.0, 1.0]
    print("x     ReLU'(autograd)   启发式 (x>0)->1 else 0")
    for pv in points:
        x = torch.tensor([pv], device=device, requires_grad=True)
        y = F.relu(x)
        y.sum().backward()
        g = x.grad.item()
        h = relu_grad_heuristic(torch.tensor(pv, device=device)).item()
        print(f"{pv:4.1f}  {g:17.6f}  {h:17.0f}")


def demo_leaky_vs_relu(device: torch.device) -> None:
    _print_header("3. LeakyReLU：负半轴小斜率，减轻「全负则梯度恒为 0」")

    x = torch.tensor([-3.0, -1.0, 0.0, 2.0], device=device)
    r = F.relu(x)
    lr = F.leaky_relu(x, negative_slope=0.01)
    print(f"x:           {x.cpu().tolist()}")
    print(f"ReLU(x):     {r.cpu().numpy().round(4).tolist()}")
    print(f"LeakyReLU:   {lr.cpu().numpy().round(4).tolist()}")


def demo_match_mlp_style(device: torch.device) -> None:
    _print_header("4. 与 model.MLP 一致：Flatten -> Linear -> ReLU -> Linear")

    from src.model import MLP

    torch.manual_seed(0)
    m = MLP(hidden_dim=32, num_classes=10).to(device)
    x = torch.randn(2, 1, 28, 28, device=device)
    y = m(x)
    print(f"输入形状 {tuple(x.shape)} -> logits 形状 {tuple(y.shape)}")
    print("(内部第二层为 nn.ReLU(inplace=True)，概念与本文件一致。)")


def demo_plot(project_root: Path) -> None:
    _print_header("5. 保存 ReLU(x) 与示意「导数」曲线图")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    x = torch.linspace(-3.0, 3.0, 500)
    y = F.relu(x)
    # 示意 dReLU/dx（0/1 阶梯；绘图用细线连接，非数学上的导数在 0 点）
    g = (x > 0).float()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x.numpy(), y.numpy(), label="ReLU(x)", linewidth=2)
    ax.plot(x.numpy(), g.numpy(), label="ReLU'(x) heuristic (0/1)", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("ReLU and piecewise derivative")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 3.5)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "relu_and_derivative.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_match_torch(device)
    demo_derivative_at_points(device)
    demo_leaky_vs_relu(device)
    demo_match_mlp_style(device)
    demo_plot(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
