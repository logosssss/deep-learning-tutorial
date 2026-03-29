"""
Softmax：说明 + 可运行示例（多类 logits → 概率分布）

---------------------------------------------------------------------------
概念摘要（与代码中的 demo 对应）
---------------------------------------------------------------------------

1. 定义（对向量 x = (x_1, …, x_C)）
   softmax(x)_i = exp(x_i) / Σ_j exp(x_j)。
   输出各分量为正，且 **和为 1**，常把最后一层「类别得分」解释为未归一化的 **logits**，
   经 Softmax 后得到类别概率（在分类的生成式叙述下与多项分布相联系；实际训练时常直接用
   **对数域** 的损失，见下）。

2. 数值稳定
   对任意常数 c，softmax(x) = softmax(x - c)。常取 c = max_j x_j，只对 (x - max) 做 exp，
   避免 exp 上溢、并减轻下溢（**log-sum-exp** 技巧的思想）。

3. 二类与 Sigmoid 的关系
   若只有两个 logits [a, b]，则第二类概率
   p_2 = exp(b)/(exp(a)+exp(b)) = sigmoid(b - a)。
   特别地 a=0 时 p_2 = sigmoid(b)。即二类 Softmax 与「单 logit + Sigmoid」等价。

4. PyTorch 中的损失（与 `train.py` / MNIST 一致）
   `nn.CrossEntropyLoss` / `F.cross_entropy` 的输入应是 **logits**（未 softmax），
   内部用 **log_softmax + NLL**，数值稳定。**不要**先把 logits 做 softmax 再喂给 CE
   （除非你知道自己在做别的目标；标准多类分类不要这样做）。

5. 维度
   分类网络常输出形状 (N, C)；对最后一维 C 做 softmax：`dim=-1` 或 `dim=1`。

6. 温度（可选）
   softmax(x / T)：**T > 1** 分布更平，**T < 1** 更尖，用于知识蒸馏等场景。

---------------------------------------------------------------------------
运行（项目根目录）:
  python -m src.softmax_demo
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


def softmax_manual_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定 Softmax：先减 max，再 exp，再归一化。"""
    x_max = x.amax(dim=dim, keepdim=True)
    ex = torch.exp(x - x_max)
    return ex / ex.sum(dim=dim, keepdim=True)


def demo_match_torch(device: torch.device) -> None:
    _print_header("1. 手写稳定 Softmax 与 F.softmax 一致")

    torch.manual_seed(0)
    x = torch.randn(3, 5, device=device)
    y_m = softmax_manual_stable(x, dim=-1)
    y_t = F.softmax(x, dim=-1)
    err = (y_m - y_t).abs().max().item()
    print(f"max |manual - F.softmax|: {err:.3e}")
    row_sums = y_t.sum(dim=-1)
    print(f"每行和（应全为 1）: {row_sums.cpu().numpy().round(6)}")
    print(f"最小概率（应 > 0）: {y_t.min().item():.3e}")


def demo_two_class_vs_sigmoid(device: torch.device) -> None:
    _print_header("2. 二类：softmax([0, b]) 的第二维 = sigmoid(b)")

    bs = torch.tensor([-2.0, 0.0, 2.0, 5.0], device=device)
    for b in bs:
        logits = torch.tensor([0.0, b.item()], device=device)
        p = F.softmax(logits, dim=-1)
        p2_softmax = p[1].item()
        p2_sigmoid = torch.sigmoid(b).item()
        print(f"b={b.item():5.1f}  p(class2) softmax={p2_softmax:.6f}  sigmoid(b)={p2_sigmoid:.6f}")


def demo_cross_entropy_logits_only(device: torch.device) -> None:
    _print_header("3. CrossEntropyLoss：应传入 logits；若误传 softmax 概率会错")

    torch.manual_seed(1)
    n, c = 4, 5
    logits = torch.randn(n, c, device=device)
    target = torch.tensor([0, 1, 2, 3], device=device)

    loss_ok = F.cross_entropy(logits, target)
    probs = F.softmax(logits, dim=-1)
    loss_bad = F.cross_entropy(probs, target)
    print(f"F.cross_entropy(logits,  target)     = {loss_ok.item():.6f}  （正确用法）")
    print(f"F.cross_entropy(softmax(logits), …)  = {loss_bad.item():.6f}  （错误：输入应是 logits）")


def demo_log_softmax_equivalence(device: torch.device) -> None:
    _print_header("4. log_softmax 与 log(softmax) 在稳定实现下一致")

    x = torch.randn(2, 4, device=device)
    ls = F.log_softmax(x, dim=-1)
    manual = torch.log(F.softmax(x, dim=-1) + 1e-30)
    err = (ls - manual).abs().max().item()
    print(f"max |log_softmax - log(softmax)|: {err:.3e}")
    print("(训练用 log_softmax + NLL 等价于 CE(logits)，且更稳。)")


def demo_temperature(device: torch.device) -> None:
    _print_header("5. 温度 T：softmax(x/T) 改变分布尖锐程度")

    x = torch.tensor([[2.0, 1.0, 0.1]], device=device)
    for t in [0.5, 1.0, 2.0]:
        p = F.softmax(x / t, dim=-1)
        print(f"T={t}  p = {p.cpu().numpy().round(4).tolist()}")


def demo_plot(project_root: Path) -> None:
    _print_header("6. 保存示意图：一组 logits 经 Softmax 后的概率（柱状图）")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过作图。")
        return

    torch.manual_seed(42)
    logits = torch.randn(5)
    probs = F.softmax(logits, dim=-1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(5), probs.numpy(), tick_label=[f"z{i}" for i in range(5)])
    ax.set_ylabel("probability")
    ax.set_title("Softmax probabilities for 5 logits")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    out_dir = project_root / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "softmax_bar.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_match_torch(device)
    demo_two_class_vs_sigmoid(device)
    demo_cross_entropy_logits_only(device)
    demo_log_softmax_equivalence(device)
    demo_temperature(device)
    demo_plot(Path(__file__).resolve().parents[1])

    _print_header("与 train.py 的联系")
    print(
        "MNIST 多类分类使用 nn.CrossEntropyLoss()，模型输出 logits，"
        "与本节「CE 吃 logits」一致；不要在 MLP 末尾再套 Softmax 再接 CE。"
    )


if __name__ == "__main__":
    main()
