"""
神经网络基础概念的可运行示例（与教程概念一一对应）。

在项目根目录执行:
  python -m src.concepts_demo
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model import MLP


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_forward_loss_backward_step(device: torch.device) -> None:
    """前向传播 → 损失 → 反向传播 → 优化器更新一步；打印张量形状与参数是否变化。"""
    _print_header("1. 前向、损失、反向传播、优化器一步（与 train_utils 中循环一致）")

    torch.manual_seed(0)
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    model = MLP(hidden_dim=32, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"输入 x 形状: {tuple(x.shape)}  (batch, 通道, 高, 宽)")
    print(f"标签 y 形状: {tuple(y.shape)}  (batch,)")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(x)
    print(f"输出 logits 形状: {tuple(logits.shape)}  (batch, 类别数)，未经 softmax 的类别得分")

    loss = criterion(logits, y)
    print(f"标量 loss: {loss.item():.4f}  (CrossEntropyLoss 内部含 log-softmax)")

    lin = model.net[1]
    w_before = lin.weight.detach().clone()
    loss.backward()
    assert lin.weight.grad is not None
    print(f"第一层 Linear 权重梯度范数: {lin.weight.grad.norm().item():.6f}")

    optimizer.step()
    delta = (lin.weight.detach() - w_before).abs().mean().item()
    print(f"optimizer.step() 后，第一层权重平均变化量: {delta:.6f}  (非零说明参数已更新)")


class TinyMLPWithDropout(nn.Module):
    """仅用于演示 train()/eval() 对 Dropout 的影响；正式任务仍用 model.MLP。"""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo_train_vs_eval(device: torch.device) -> None:
    """Dropout 在 model.train() 时随机置零部分神经元，在 model.eval() 时关闭。"""
    _print_header("2. train() 与 eval()（含 Dropout 时行为不同）")

    torch.manual_seed(42)
    x = torch.randn(2, 1, 28, 28, device=device)
    model = TinyMLPWithDropout().to(device)

    model.train()
    o1 = model(x)
    o2 = model(x)
    diff_train = (o1 - o2).abs().max().item()
    print(f"train() 下同一输入前向两次，logits 最大差异: {diff_train:.6f}  (Dropout 随机，通常 > 0)")

    model.eval()
    with torch.no_grad():
        o3 = model(x)
        o4 = model(x)
    diff_eval = (o3 - o4).abs().max().item()
    print(f"eval() 下同一输入前向两次，logits 最大差异: {diff_eval:.6f}  (Dropout 关闭，应为 0)")


def _ensure_utf8_stdio() -> None:
    import sys

    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    _ensure_utf8_stdio()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    demo_forward_loss_backward_step(device)
    demo_train_vs_eval(device)

    _print_header("说明")
    print(
        "完整训练循环 = 对 DataLoader 中每个 batch 重复："
        "zero_grad → forward → loss → backward → step；"
        "验证阶段用 model.eval() 与 torch.no_grad()，见 src/train_utils.py。"
    )


if __name__ == "__main__":
    main()
