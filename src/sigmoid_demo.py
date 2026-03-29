"""
Sigmoid（S 型）函数：说明 + 可运行示例

---------------------------------------------------------------------------
概念摘要（与代码中的 demo 对应）
---------------------------------------------------------------------------

1. 定义
   σ(x) = 1 / (1 + exp(-x))，定义域为全体实数。

2. 值域与形状
   对任意 x，有 0 < σ(x) < 1；x → -∞ 时趋近 0，x → +∞ 时趋近 1，过点 (0, 0.5)。
   曲线呈「S」形，故常称 logistic / sigmoid。

3. 导数（链式法则中常用）
   σ'(x) = σ(x) · (1 - σ(x))。
   在 x = 0 处导数最大，为 0.25；|x| 很大时 σ 接近 0 或 1，导数接近 0，
   即「饱和」，深层网络中连续乘很多小导数易导致梯度消失（历史上隐藏层
   更常用 ReLU 等；Sigmoid 仍常见于二分类输出层、门控结构等）。

4. 与损失函数搭配
   二分类时若最后一层输出「概率」，常与二元交叉熵组合。PyTorch 中更推荐
   **不**在模型末尾手写 Sigmoid，而用 `BCEWithLogitsLoss`：内部对 **logits**
   做数值稳定的 sigmoid + BCE，避免先 exp 再 log 带来的不稳定。

5. 与 Softmax 的区别（简述）
   Sigmoid 常对每个标量独立压缩到 (0,1)；Softmax 对一组 logits 归一化
   为概率分布且和为 1，多用于多类单标签分类。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.sigmoid_demo
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


def sigmoid_manual_stable(x: torch.Tensor) -> torch.Tensor:
    """
    分域写法减轻大正/大负时的 exp 溢出（教学用；工程上直接用 torch.sigmoid）。
    x >= 0: 1 / (1 + exp(-x))
    x < 0:  exp(x) / (1 + exp(x))
    """
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    ex = torch.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def sigmoid_derivative_formula(s: torch.Tensor) -> torch.Tensor:
    """解析导数：σ'(x) = σ(x)(1 - σ(x))，这里 s 即 σ(x)。"""
    return s * (1.0 - s)


def demo_match_torch(device: torch.device) -> None:
    _print_header("1. 手写稳定 Sigmoid 与 torch.sigmoid 数值一致")

    x = torch.tensor([-1000.0, -10.0, 0.0, 10.0, 1000.0], device=device)
    y_m = sigmoid_manual_stable(x)
    y_t = torch.sigmoid(x)
    err = (y_m - y_t).abs().max().item()
    print(f"max |manual - torch.sigmoid|: {err:.3e}")
    print(f"x: {x.cpu().tolist()}")
    print(f"σ(x): {y_t.cpu().numpy().round(6).tolist()}")


def demo_derivative_autograd_vs_formula(device: torch.device) -> None:
    _print_header("2. 导数：autograd 与公式 σ(1-σ) 一致")

    x = torch.linspace(-3.0, 3.0, 7, device=device, requires_grad=True)
    s = torch.sigmoid(x)
    loss = s.sum()
    loss.backward()
    grad_auto = x.grad.detach()
    s_no_grad = torch.sigmoid(x.detach())
    grad_formula = sigmoid_derivative_formula(s_no_grad)
    err = (grad_auto - grad_formula).abs().max().item()
    print(f"max |grad_auto - σ(1-σ)|: {err:.3e}")
    print(f"x:        {x.detach().cpu().numpy().round(4)}")
    print(f"σ'(x):    {grad_formula.cpu().numpy().round(4)}")


def demo_saturation(device: torch.device) -> None:
    _print_header("3. 饱和：|x| 大时 σ'(x) 接近 0（梯度消失风险）")

    xs = torch.tensor([-10.0, -2.0, 0.0, 2.0, 10.0], device=device)
    s = torch.sigmoid(xs)
    gp = sigmoid_derivative_formula(s)
    print("x      σ(x)      σ'(x)")
    for i in range(xs.numel()):
        print(f"{xs[i].item():6.1f}  {s[i].item():.6f}  {gp[i].item():.6e}")


def demo_bce_with_logits_hint(device: torch.device) -> None:
    _print_header("4. 二分类：BCEWithLogitsLoss（推荐）在 logits 上算稳定")

    torch.manual_seed(0)
    logits = torch.randn(4, device=device)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)

    loss_fn = nn.BCEWithLogitsLoss()
    loss_a = loss_fn(logits, target)

    prob = torch.sigmoid(logits)
    loss_b = F.binary_cross_entropy(prob, target)

    print(f"BCEWithLogitsLoss(logits, y):     {loss_a.item():.6f}")
    print(f"BCE(sigmoid(logits), y):          {loss_b.item():.6f}")
    print("(两者在数学上应对同一目标；优先用前者，大 |logit| 时更稳定。)")


def demo_plot(project_root: Path) -> None:
    _print_header("5. 保存 σ(x) 与 σ'(x) 曲线图")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    x = torch.linspace(-8.0, 8.0, 500)
    s = torch.sigmoid(x)
    sp = sigmoid_derivative_formula(s)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x.numpy(), s.numpy(), label="sigmoid(x)", linewidth=2)
    ax.plot(x.numpy(), sp.numpy(), label="sigmoid'(x)", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Sigmoid and its derivative")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sigmoid_and_derivative.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_match_torch(device)
    demo_derivative_autograd_vs_formula(device)
    demo_saturation(device)
    demo_bce_with_logits_hint(device)
    demo_plot(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
