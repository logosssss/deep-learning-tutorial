"""
其他常用激活函数：补充说明 + 可运行示例

---------------------------------------------------------------------------
为何需要「其他」激活
---------------------------------------------------------------------------
ReLU / Sigmoid / Tanh 已单独有 demo（见 `relu_demo.py`、`sigmoid_demo.py`、`tanh_demo.py`；
总览与曲线见 `activation_demo.py`）。本节补充在 **Transformer、现代 CNN、移动端、
自归一化网络** 等场景中常见的光滑、可学习或分段线性近似等变体。

---------------------------------------------------------------------------
概念摘要（与代码中的 demo 对应）
---------------------------------------------------------------------------

1. **Softplus** — `softplus(x) = log(1 + exp(x))`
   处处光滑、处处可导，为 ReLU 的平滑近似（x 大时近似 x）。VAE 等模型中常见。

2. **ELU** — Exponential Linear Unit
   x ≥ 0 为 x；x < 0 为 α·(exp(x)-1)，默认 α=1。负半轴光滑趋近 -α，减轻 ReLU 在 0 处的
   尖角；输出均值可更接近 0（仍非严格零中心）。

3. **SELU** — Scaled ELU
   固定 α、λ 使在特定初始化与正则下网络层输出近似保持零均值单位方差（**自归一化**）。
   需配合 **LeCun normal** 初始化、`AlphaDropout` 等；用法不当可能不稳定，入门可先了解即可。

4. **GELU** — Gaussian Error Linear Unit
   近似理解为「按高斯权重对 identity 做门控」，在 **BERT、GPT、ViT** 等中广泛使用。
   PyTorch 中可选近似实现（如 `tanh` 近似）以加速。

5. **SiLU / Swish** — `x · σ(x)`
   光滑、非单调（略低于 0 的小负区间）；**EfficientNet** 等用过 SiLU。

6. **Mish** — `x · tanh(softplus(x))`
   光滑、非单调，部分检测/分类任务中作为 ReLU 替代品实验。

7. **PReLU** — Parametric ReLU
   负半轴斜率 **可学习**（标量或逐通道）。比固定 `LeakyReLU` 更灵活，参数略增。

8. **Hardsigmoid / Hardtanh / Hardswish**
   用分段线性或 `relu6` 近似 Sigmoid/Swish，**计算更省**，适合移动端与嵌入式。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.other_activations_demo
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


def _build_modules(device: torch.device) -> dict[str, nn.Module]:
    """PyTorch 内置模块；GELU 近似方式随版本可能不同，此处用默认。"""
    m: dict[str, nn.Module] = {
        "Softplus": nn.Softplus(),
        "ELU": nn.ELU(alpha=1.0),
        "SELU": nn.SELU(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "Mish": nn.Mish(),
        "PReLU": nn.PReLU(num_parameters=1, init=0.25),
        "Hardswish": nn.Hardswish(),
        "Hardsigmoid": nn.Hardsigmoid(),
    }
    return {k: v.to(device) for k, v in m.items()}


def demo_forward_ranges(device: torch.device) -> None:
    _print_header("1. 同一输入上各激活的输出范围（[-3,3] 上 7 个点）")

    x = torch.linspace(-3.0, 3.0, 7, device=device).unsqueeze(0)
    print(f"x: {x.squeeze(0).cpu().numpy().round(2).tolist()}")
    print()

    modules = _build_modules(device)
    for name, act in modules.items():
        y = act(x)
        print(f"{name:12s}  [min, max] = [{y.min().item():7.4f}, {y.max().item():7.4f}]")


def demo_silu_vs_sigmoid_identity(device: torch.device) -> None:
    _print_header("2. SiLU：SiLU(x) = x · σ(x)；在 x=0 处值为 0，导数为 0.5")

    x = torch.tensor([0.0], device=device, requires_grad=True)
    y = F.silu(x)
    y.backward()
    print(f"SiLU(0) = {y.item():.6f}")
    print(f"SiLU'(0) = {x.grad.item():.6f}  (理论 0.5)")


def demo_prelu_learnable(device: torch.device) -> None:
    _print_header("3. PReLU：负斜率可学习（此处 1 个标量参数）")

    p = nn.PReLU(num_parameters=1, init=0.25).to(device)
    n = sum(p.numel() for p in p.parameters())
    print(f"PReLU 可训练参数个数: {n}")
    x = torch.tensor([-2.0, 1.0], device=device).view(1, 2)
    y = p(x)
    print(f"x = [-2, 1] -> PReLU(x) = {y.squeeze(0).detach().cpu().numpy().round(4).tolist()}")


def demo_gelu_approx_note(device: torch.device) -> None:
    _print_header("4. GELU：默认与 approximate='tanh' 近似（若当前 PyTorch 支持）")

    x = torch.linspace(-2.0, 2.0, 5, device=device)
    g_def = nn.GELU()
    y0 = g_def(x)
    # 新版本支持 approximate 参数
    try:
        g_tanh = nn.GELU(approximate="tanh")
        y1 = g_tanh(x)
        err = (y0 - y1).abs().max().item()
        print(f"GELU() 与 GELU(approximate='tanh') 最大差: {err:.3e}")
    except TypeError:
        print("当前 torch.nn.GELU 无 approximate 参数，仅打印默认 GELU 输出：")
    print(f"x:    {x.cpu().numpy().round(2)}")
    print(f"GELU: {y0.detach().cpu().numpy().round(4)}")


def demo_plot(project_root: Path) -> None:
    _print_header("5. 保存「其他激活」曲线图（与 activation_curves 互补）")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    t = torch.linspace(-4.0, 4.0, 500)
    pairs = [
        ("Softplus", nn.Softplus()),
        ("ELU", nn.ELU()),
        ("SELU", nn.SELU()),
        ("GELU", nn.GELU()),
        ("SiLU", nn.SiLU()),
        ("Mish", nn.Mish()),
        ("Hardswish", nn.Hardswish()),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, m in pairs:
        y = m(t.unsqueeze(0)).squeeze(0).detach().numpy()
        ax.plot(t.numpy(), y, label=name, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("More activation functions (overview)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "other_activations_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_forward_ranges(device)
    demo_silu_vs_sigmoid_identity(device)
    demo_prelu_learnable(device)
    demo_gelu_approx_note(device)
    demo_plot(Path(__file__).resolve().parents[1])

    _print_header("小结与索引")
    print(
        "ReLU/Sigmoid/Tanh/Softmax 见各自 *_demo.py；总览图见 notebooks/activation_curves.png。\n"
        "本仓库 MLP 仍用 nn.ReLU（src/model.py）；换激活只需在 model 里替换对应 nn 模块并调参。"
    )


if __name__ == "__main__":
    main()
