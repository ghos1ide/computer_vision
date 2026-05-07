# 计算机视觉：基于Jittor的图像分类任务

本项目实现了使用国产深度学习框架 **Jittor (计图)** 进行图像分类任务的完整流程。使用 CIFAR-10 数据集，并且实现了一个轻量级的 ResNet 架构。

## 项目结构
- `model.py`: 包含了加入 ResNet 模块的卷积神经网络 (`SimpleResNet`)。创新地引入了 Residual Block 让网络支持更深的层次并加速收敛。
- `train.py`: 完整的数据加载、训练、测试评估和可视化代码。
- `compare_ablation.py`: 逐个对比随机裁剪、批归一化、Dropout、动量和权重衰减的消融实验程序。
- `README.md`: 项目说明及超参数探讨。

## 环境要求
- Python 3.7+
- Jittor `pip install jittor`
- Matplotlib `pip install matplotlib`

## 运行方法
运行以下命令开始下载数据集并训练模型：
```bash
python train.py
```
训练完成后，将在当前目录下生成 `training_metrics.png`，展示了训练过程中的Loss曲线和测试集准确率曲线。

如果需要直接比较各个设计模块的贡献，可以运行：
```bash
python compare_ablation.py --epochs 20 --batch-size 32
```
脚本会分别训练基线配置、去除随机裁剪、去除批归一化、去除 Dropout、去除动量和去除权重衰减的版本，并输出汇总表与两张对比图。

## 超参数影响探讨
在深度学习和CNN的训练中，几个关键超参数对训练效果有着决定性影响：

1. **学习率 (Learning Rate)**: 
   - 设定的为 `0.01`。如果偏大，损失可能会震荡或发散；稍微减小可以让模型在收敛后期寻找到更优的局部解。可以引入学习率衰减策略（如StepLR）提高最终精度。
2. **批大小 (Batch Size)**: 
   - 设定的为 `64`。较小的 Batch Size 提供了正则化效果，使得每个 Batch 的梯度带有一定噪声，能帮助跳出局部最优；过大会导致泛化性下降和显存不足。
3. **动量 (Momentum)**与**权重衰减 (Weight Decay)**:
   - 动量 `0.9` 加速了模型在一致梯度上的收敛。
   - 权重衰减 `1e-4` 起到了 L2 正则化的作用，限制网络权重过大，防止在 CIFAR-10 这类中等规模数据集上产生过拟合现象。
