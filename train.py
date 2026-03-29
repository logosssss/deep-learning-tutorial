"""
入口：从 configs/default.yaml 读配置，训练 MNIST 上的 MLP。
用法（在项目根目录）:
  python train.py
  python train.py --config configs/default.yaml
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config
from src.dataset import get_mnist_loaders
from src.model import MLP
from src.train_utils import evaluate, train_one_epoch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML 配置文件路径",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    device_str = cfg["train"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    set_seed(cfg["train"]["seed"])

    data_dir = Path(cfg["data"]["data_dir"])
    train_loader, test_loader = get_mnist_loaders(
        data_dir=data_dir,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = MLP(
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
    )

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch}/{cfg['train']['epochs']} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    name = cfg.get("experiment_name", "run")
    torch.save(
        {"model_state": model.state_dict(), "config": cfg},
        ckpt_dir / f"{name}_last.pt",
    )
    print(f"已保存: {ckpt_dir / f'{name}_last.pt'}")


if __name__ == "__main__":
    main()
