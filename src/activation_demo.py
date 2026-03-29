"""
激活函数：可运行示例（形状、输出范围、梯度；无非线性时多层线性可合并为一层）。

在项目根目录执行:
  python -m src.activation_demo

可选：在 notebooks/ 下生成激活函数曲线图 activation_curves.png（需 matplotlib）。
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


def demo_linear_without_activation_is_single_layer(device: torch.device) -> None:
    """
    无激活时：Linear(Linear(x)) 等价于一次线性变换（含合并后的权重）。
    加上 ReLU 后一般不能再写成单层线性。
    """
    _print_header("1. 无非线性：两层 Linear 等价于一层；加 ReLU 后不再等价")

    d = 8
    n = 16
    torch.manual_seed(0)
    x = torch.randn(n, d, device=device)

    lin1 = nn.Linear(d, d, bias=False).to(device)
    lin2 = nn.Linear(d, d, bias=False).to(device)
    relu = nn.ReLU(inplace=False)

    y_stack = lin2(lin1(x))
    W_merged = lin2.weight @ lin1.weight
    y_merged = nn.functional.linear(x, W_merged)
    err = (y_stack - y_merged).abs().max().item()
    print(f"无激活: lin2(lin1(x)) 与 单次 linear(x, W2@W1) 最大误差: {err:.2e}  (应接近机器精度)")

    y_relu = lin2(relu(lin1(x)))
    err2 = (y_relu - y_merged).abs().mean().item()
    print(
        f"有 ReLU: lin2(relu(lin1(x))) 与 单层线性 平均绝对误差: {err2:.4f}  (通常显著 > 0，表达能力不同)"
    )


def demo_activation_outputs(device: torch.device) -> None:
    """常见激活在相同输入上的数值范围与梯度。"""
    _print_header("2. 常见激活：前向数值与反向梯度（示意）")

    x = torch.linspace(-3.0, 3.0, 7, device=device)
    x_row = x.unsqueeze(0)  # (1, 7) 模拟一条特征

    activations: dict[str, nn.Module] = {
        "ReLU": nn.ReLU(),
        "LeakyReLU(0.01)": nn.LeakyReLU(0.01),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "GELU": nn.GELU(),
        "SiLU(Swish)": nn.SiLU(),
    }

    print(f"输入 x（7 个点）: {x.cpu().numpy().round(4)}")
    print()

    for name, act in activations.items():
        y = act(x_row)
        print(f"{name:16s} 输出范围 [{y.min().item():.4f}, {y.max().item():.4f}]")

    # 梯度：对 sum(act(x)) 反传，看 ∂/∂x
    print()
    print("对 sum(act(x)) 关于 x 的梯度（展示在 x=0 附近的行为，如 ReLU 的「死区」）:")
    for label, act in [("ReLU", nn.ReLU()), ("Sigmoid", nn.Sigmoid()), ("Tanh", nn.Tanh())]:
        xg = torch.linspace(-2.0, 2.0, 5, device=device, requires_grad=True)
        z = act(xg)
        z.sum().backward()
        g = xg.grad.detach().cpu().numpy().round(4)
        print(f"  {label:8s} ∂sum/∂x @ [-2,-1,0,1,2]: {g}")


def demo_plot_curves(project_root: Path) -> None:
    """将常见激活曲线保存为图片，便于与教材对照。"""
    _print_header("3. 保存激活函数曲线图（可选）")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    t = torch.linspace(-4.0, 4.0, 400)
    acts = {
        "ReLU": nn.ReLU(),
        "LeakyReLU(0.1)": nn.LeakyReLU(0.1),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, m in acts.items():
        y = m(t.unsqueeze(0)).squeeze(0).detach().numpy()
        ax.plot(t.numpy(), y, label=name, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y = activation(x)")
    ax.set_title("Common activation functions")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "activation_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    project_root = Path(__file__).resolve().parents[1]

    demo_linear_without_activation_is_single_layer(device)
    demo_activation_outputs(device)
    demo_plot_curves(project_root)

    _print_header("小结")
    print(
        "ReLU: 计算快、缓解梯度消失（正半轴导数为 1），负半轴为 0（可能「神经元死亡」）。\n"
        "Sigmoid/Tanh: 有界、平滑，深层网络中易出现饱和导致梯度很小。\n"
        "GELU/SiLU: 现代 Transformer/CNN 中常见，光滑且非单调区间更灵活。\n"
        "你项目中的 MLP 使用 nn.ReLU，见 src/model.py。"
    )


if __name__ == "__main__":
    main()
