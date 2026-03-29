"""
双曲正切 Tanh：说明 + 可运行示例

---------------------------------------------------------------------------
概念摘要（与代码中的 demo 对应）
---------------------------------------------------------------------------

1. 定义
   tanh(x) = sinh(x) / cosh(x) = (e^x - e^{-x}) / (e^x + e^{-x})。
   为奇函数：tanh(-x) = -tanh(x)，且 tanh(0) = 0。

2. 与 Sigmoid 的关系（便于记忆与实现对照）
   tanh(x) = 2 · σ(2x) - 1，其中 σ 为 Sigmoid。
   即将 Sigmoid 输出从 (0,1) 仿射变换到 (-1, 1)，并在原点对称。

3. 值域与「零中心」
   对任意 x，有 -1 < tanh(x) < 1。与 Sigmoid 在 x=0 处为 0.5 不同，Tanh 在 0 处
   输出为 0，**输出关于原点近似零中心**，历史上在隐藏层中有时被认为有利于学习
   （减轻后续层输入偏置）；现代 CNN/MLP 隐藏层更常见仍是用 ReLU/GELU 等。

4. 导数
   d/dx tanh(x) = 1 - tanh^2(x) = sech^2(x)。
   在 x = 0 处导数为 **1**（同一点上 Sigmoid 的导数最大仅为 0.25），但 |x| 很大时
   仍趋于饱和，导数趋近 0，深层网络中仍可能出现梯度消失。

5. 典型用途（简述）
   RNN/LSTM/GRU 中的门控与候选状态常出现 tanh/sigmoid 组合；全连接隐藏层在
   早期网络中常见 Tanh，现多被 ReLU 系替代。输出层若需要有界到 (-1,1) 也可用 Tanh。

6. PyTorch
   `torch.tanh`、`nn.Tanh` 与 NumPy 的 `numpy.tanh` 行为一致。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.tanh_demo
---------------------------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path

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


def tanh_from_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """恒等关系：tanh(x) = 2*sigmoid(2x) - 1（用于与 torch.tanh 对照）。"""
    return 2.0 * torch.sigmoid(2.0 * x) - 1.0


def tanh_derivative_formula(t: torch.Tensor) -> torch.Tensor:
    """解析导数：(tanh)'(x) = 1 - tanh^2(x)，这里 t 即 tanh(x)。"""
    return 1.0 - t * t


def demo_tanh_equals_2sigmoid2x_minus_1(device: torch.device) -> None:
    _print_header("1. tanh(x) 与 2·σ(2x)-1 数值一致")

    x = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0], device=device)
    y_t = torch.tanh(x)
    y_s = tanh_from_sigmoid(x)
    err = (y_t - y_s).abs().max().item()
    print(f"max |tanh(x) - (2*sigmoid(2x)-1)|: {err:.3e}")
    print(f"x:        {x.cpu().tolist()}")
    print(f"tanh(x):  {y_t.cpu().numpy().round(6).tolist()}")


def demo_derivative_autograd_vs_formula(device: torch.device) -> None:
    _print_header("2. 导数：autograd 与公式 1 - tanh^2 一致")

    x = torch.linspace(-3.0, 3.0, 7, device=device, requires_grad=True)
    y = torch.tanh(x)
    y.sum().backward()
    grad_auto = x.grad.detach()
    t = torch.tanh(x.detach())
    grad_formula = tanh_derivative_formula(t)
    err = (grad_auto - grad_formula).abs().max().item()
    print(f"max |grad_auto - (1-tanh^2)|: {err:.3e}")
    print(f"x:         {x.detach().cpu().numpy().round(4)}")
    print(f"tanh'(x):  {grad_formula.cpu().numpy().round(4)}")


def demo_saturation_and_compare_max_slope(device: torch.device) -> None:
    _print_header("3. 饱和：|x| 大时 tanh' 接近 0；x=0 处最大斜率为 1（对比 Sigmoid 为 0.25）")

    xs = torch.tensor([-5.0, -2.0, 0.0, 2.0, 5.0], device=device)
    t = torch.tanh(xs)
    tp = tanh_derivative_formula(t)
    s = torch.sigmoid(xs)
    sp = s * (1.0 - s)
    print("x      tanh(x)   tanh'(x)   sigmoid'(x)")
    for i in range(xs.numel()):
        print(
            f"{xs[i].item():5.1f}  {t[i].item():8.5f}  {tp[i].item():9.4f}  {sp[i].item():11.4f}"
        )


def demo_nn_module(device: torch.device) -> None:
    _print_header("4. nn.Tanh 与 torch.tanh 一致")

    m = nn.Tanh().to(device)
    x = torch.randn(3, 4, device=device)
    err = (m(x) - torch.tanh(x)).abs().max().item()
    print(f"max |nn.Tanh(x) - torch.tanh(x)|: {err:.3e}")


def demo_plot(project_root: Path) -> None:
    _print_header("5. 保存 tanh(x) 与 tanh'(x) 曲线图")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    x = torch.linspace(-4.0, 4.0, 500)
    t = torch.tanh(x)
    td = tanh_derivative_formula(t)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x.numpy(), t.numpy(), label="tanh(x)", linewidth=2)
    ax.plot(x.numpy(), td.numpy(), label="tanh'(x)", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Tanh and its derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tanh_and_derivative.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_tanh_equals_2sigmoid2x_minus_1(device)
    demo_derivative_autograd_vs_formula(device)
    demo_saturation_and_compare_max_slope(device)
    demo_nn_module(device)
    demo_plot(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
