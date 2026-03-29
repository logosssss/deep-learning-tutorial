# 深度学习教程项目骨架

基于 **PyTorch** 的最小可运行示例：在 **MNIST** 上训练简单 **MLP**，便于对照教程扩展（换数据、换模型、加验证集等）。

## 目录说明

| 路径 | 作用 |
|------|------|
| `configs/` | 超参数与路径（YAML） |
| `data/` | 数据目录；`raw` / `processed` 可按教程划分 |
| `src/` | 配置加载、数据集、模型、训练/验证循环 |
| `checkpoints/` | 保存的 `.pt` 权重 |
| `notebooks/` | Jupyter 实验笔记 |
| `train.py` | 命令行训练入口 |

## 环境

```bash
cd deep-learning-tutorial
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

无 NVIDIA GPU 时，请在 `configs/default.yaml` 里把 `train.device` 改为 `cpu`。

## 运行

```bash
python train.py
# 或指定配置
python train.py --config configs/default.yaml
```

首次运行会自动下载 MNIST 到 `data/`。

## 扩展建议

- 换任务：改 `src/dataset.py` 与 `src/model.py`。
- 加 TensorBoard / wandb：在 `train.py` 里记录标量与图像。
- 加早停与学习率调度：在 `train.py` 的 epoch 循环中接入 `torch.optim.lr_scheduler`。
