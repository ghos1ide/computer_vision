# Track 项目：基于 Jittor 的 VOC2012 语义分割

本项目实现了一个完整的经典视觉任务流水线（语义分割）：

- 数据集：PASCAL VOC2012 分割数据集
- 框架：Jittor
- 基线模型：U-Net
- 创新模块：可选 SE 注意力模块（可通过命令行开关切换）
- 输出：训练/验证/测试指标、曲线、检查点、历史记录和超参数对比报告

## 1. 项目文件

- `prepare_voc.py`：下载 VOC2012 并创建自定义 `train/val/test` 划分
- `dataset.py`：图像与掩码同步预处理，以及 Jittor 数据集封装
- `model.py`：U-Net + 可选 SE 注意力
- `metrics.py`：像素准确率、平均准确率、mIoU、fwIoU
- `train.py`：完整的训练/验证/测试流水线和指标绘图
- `evaluate.py`：在 train/val/test 划分上评估已保存的检查点
- `tune.py`：超参数实验和结果排序
- `check_gpu.py`：Jittor CUDA 检查工具
- `run_in_wsl.sh`：在 WSL 虚拟环境中运行训练

## 2. 环境

推荐使用 Python 3.9 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```

检查 Jittor GPU 运行环境：

```bash
python check_gpu.py
```

## 3. 数据准备

准备 VOC2012 和划分文件（`train.txt`、`val.txt`、`test.txt`）：

```bash
python prepare_voc.py --data-root ./data --val-ratio 0.5 --seed 42
```

划分策略：

- train：官方 VOC train 划分
- val/test：从官方 VOC val 划分中按 `val_ratio` 再切分
- 如果要强制重新生成划分，可加 `--force-split`

## 4. 训练

默认训练命令：

```bash
python train.py --data-root ./data --save-dir ./runs/unet_se
```

常用参数：

```bash
python train.py \
  --epochs 80 \
  --batch-size 8 \
  --lr 0.003 \
  --lr-scheduler onecycle \
  --max-lr 0.006 \
  --min-lr 3e-4 \
  --warmup-epochs 2 \
  --crop-size 320 \
  --base-size 512 \
  --base-channels 32 \
  --use-se
```

当前训练代码的默认优化策略：

- 损失函数为交叉熵（CE），不再使用 CE + Dice 组合。
- 优化器为 SGD，默认学习率为 `0.003`。
- 学习率调度默认使用 `onecycle`，并支持 `constant` 和 `cosine`。
- `onecycle` 默认配合 `--max-lr 0.006`、`--min-lr 3e-4` 和 `--warmup-epochs 2`。

关闭创新模块（SE）：

```bash
python train.py --data-root ./data --save-dir ./runs/unet_no_se --no-se
```

`save-dir` 下的主要输出：

- `best_model.pkl`：按验证集 mIoU 选出的最佳检查点
- `training_curves.png`：loss、像素准确率、mIoU 曲线
- `history.json`：训练历史、最佳验证分数、测试指标
- `config.json`：完整运行配置

## 5. 评估

在测试集划分上评估已训练好的检查点：

```bash
python evaluate.py \
  --data-root ./data \
  --checkpoint ./runs/unet_se/best_model.pkl \
  --split test
```

## 6. 超参数实验

运行一个小型实验网格：

```bash
python tune.py \
  --data-root ./data \
  --save-root ./runs/tuning \
  --epochs 12 \
  --lrs 0.01,0.003 \
  --batch-sizes 4,8 \
  --use-se-options 1,0
```

输出文件：

- `tuning_results.csv`
- `tuning_results.json`
- `tuning_results.png`

## 7. 与实验要求的对应关系

- 经典视觉任务：语义分割（VOC2012 上的 U-Net）
- 完整可运行流水线：数据准备、训练、评估、调参
- 数据集标准化：VOC2012 + 确定性 train/val/test 划分生成
- 经典算法复现：在 Jittor 中从零实现 U-Net
- 创新设计：可选 SE 注意力模块进行特征重标定
- 性能评估：记录并保存 loss、像素准确率、mean accuracy、mIoU、fwIoU
- 超参数分析：学习率 / batch size / 注意力开关实验

## 8. 备注

- 在某些 Windows 环境中，Jittor 编译可能会因工具链或路径问题而失败。
- 如果遇到这种情况，请使用 Linux/WSL 中的 `run_in_wsl.sh` 运行训练。

## 9. 卡住问题排查

如果命令在 Windows 上看起来“卡住”，通常是 Jittor 在启动时进行编译。

- 首次启动可能需要几分钟。
- 如果出现 C++ 编译错误（例如 `error C2440`），通常是本地工具链问题。
- 项目脚本现在默认会阻止原生 Windows 下的 Jittor 导入，并直接引导你使用 WSL 命令。

推荐的处理方式：

```bash
cd track
bash run_in_wsl.sh check
bash run_in_wsl.sh prepare --data-root ./data --val-ratio 0.5 --seed 42
bash run_in_wsl.sh train --data-root ./data --save-dir ./runs/unet_se
```

如果 WSL 提示缺少虚拟环境或模块，请先在 WSL 中完成一次初始化：

```bash
cd /mnt/d/sjtuwht/3-2/计算机视觉/track
python3 -m venv ~/.venv_linux
source ~/.venv_linux/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

`run_in_wsl.sh` 行为：

- 会自动从 `$HOME/.venv_linux`、`$HOME/.venv`、`$HOME/venv` 中检测 Linux 虚拟环境
- 可以通过 `WSL_VENV_PATH` 覆盖虚拟环境路径
- 可以通过 `WSL_PYTHON_BIN` 覆盖 Python 命令

示例：

```bash
export WSL_VENV_PATH=$HOME/.venv_linux
export WSL_PYTHON_BIN=python3
bash run_in_wsl.sh check
```

其他实用建议：

- 在 Windows 上建议保持 `--num-workers 0`，避免 dataloader worker 卡死。
- 确保 Linux 虚拟环境中已安装 `jittor`、`numpy`、`pillow` 和 `matplotlib`。
- 如果你仍想尝试原生 Windows 执行，可在运行脚本前设置环境变量 `JT_FORCE_WINDOWS_IMPORT=1`。
