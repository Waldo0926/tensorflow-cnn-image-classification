# TensorFlow CNN 图像分类

[English README](README.md)

这是一个由 **2024 年机器学习课程设计**整理、重构而来的 Portfolio 项目，使用 **TensorFlow/Keras 卷积神经网络（CNN）**完成 **CIFAR-10** 与 **MNIST** 图像分类，并比较网络深度、filter 数量、卷积核、池化方式和训练时长对模型效果的影响。

GitHub 版本保留原课程设计的实验问题，但不把当年的原始脚本直接当作正式工程代码，而是重新整理为结构清晰、可复现、适合公开展示的训练与实验流程。

## 这个项目可以展示什么？

- TensorFlow/Keras 图像分类完整流程
- 通过 Keras 自动下载 CIFAR-10 与 MNIST，无需配置本地数据路径
- 固定 random seed，并采用按类别分层的 train/validation 划分
- 将原课程设计中的不同 CNN 结构整理为命名 preset
- 使用完整测试集评估 accuracy、loss、macro precision/recall/F1 和每类指标
- 自动生成训练曲线、混淆矩阵，以及明确区分的**预测正确 / 预测错误样例**和置信度
- 批量运行多个 CNN 后自动生成 CSV + Markdown 模型比较表并按准确率排名
- 记录模型参数量、训练耗时、运行环境版本和 model summary
- GitHub Actions 自动执行轻量语法与单元测试检查
- 原始中文课程报告保留作为项目来源证明，同时对学号进行真正 PDF 脱敏

## 目录结构

```text
.
├── .github/workflows/quality.yml
├── README.md
├── README.zh-CN.md
├── requirements.txt
├── src/
│   ├── train.py
│   └── ml_project/
│       ├── data.py
│       ├── evaluation.py
│       ├── models.py
│       ├── reporting.py
│       ├── training.py
│       └── visualization.py
├── experiments/
│   ├── run_experiments.py
│   └── summarize_results.py
├── tests/
│   └── test_reporting.py
├── docs/
│   ├── EXPERIMENTS.md
│   ├── ORIGINAL_WORK.md
│   └── original/
│       └── 2024-course-report-cn.pdf
└── results/
    └── .gitkeep
```

## 快速开始

### 1. 创建虚拟环境

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

建议使用当前 TensorFlow 版本支持的 Python；对这个项目而言，Python 3.11 / 3.12 是相对稳妥的选择。

### 2. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 查看模型配置

```bash
python src/train.py --list-presets
```

### 4. 开始训练

CIFAR-10 基础模型：

```bash
python src/train.py --dataset cifar10 --preset baseline --epochs 20
```

MNIST 四层卷积 / 两层池化模型：

```bash
python src/train.py --dataset mnist --preset four_conv_two_pool --epochs 15
```

快速确认项目可以运行：

```bash
python src/train.py --dataset mnist --preset baseline --epochs 2 --train-limit 5000
```

第一次运行时 Keras 会自动下载数据集。`--train-limit` 只限制训练子集，validation 和 test 保持完整，因此快速实验仍然使用稳定的验证/测试基准。

## Command-line 参数

不需要修改代码即可控制实验：

```bash
python src/train.py \
  --dataset cifar10 \
  --preset four_layer \
  --epochs 30 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --validation-fraction 0.2 \
  --patience 5 \
  --seed 42
```

支持的主要参数包括：

- `--dataset {cifar10,mnist}`
- `--preset <name>`
- `--epochs <n>`
- `--batch-size <n>`
- `--learning-rate <value>`
- `--validation-fraction <0..1>`
- `--train-limit <n>`：快速本地测试
- `--patience <n>`：可选 early stopping
- `--seed <n>`
- `--output-dir <path>`
- `--no-save-model`：只保留指标/图表，不生成较大的 `.keras` 模型

## 单次实验会输出什么？

默认输出到：

```text
results/<dataset>/<preset>/
```

包括：

```text
metrics.json
history.csv
per_class_metrics.csv
model_summary.txt
training_curves.png
confusion_matrix.png
prediction_examples.png
model.keras                 # 使用 --no-save-model 时不会生成
```

其中 `prediction_examples.png` 不再只是拿测试集前几张图片，而是**主动筛选正确预测和错误预测**两组样例，并显示预测类别的 confidence。

`metrics.json` 还会记录：

- test accuracy / loss
- macro precision / recall / F1
- best validation accuracy / loss
- 模型参数量
- 实际训练 epoch 数
- 训练耗时
- train / validation / test 样本数
- seed 和超参数
- Python / TensorFlow / NumPy 版本与运行平台

生成结果默认由 `.gitignore` 忽略，避免训练模型和大量实验文件把 GitHub 仓库变得臃肿。

## 自动比较不同 CNN 架构

运行全部预设：

```bash
python experiments/run_experiments.py --dataset all --epochs 20
```

电脑性能有限时可以先跑：

```bash
python experiments/run_experiments.py --dataset all --epochs 3 --train-limit 10000
```

批量实验默认**不会为每个模型都保存 `.keras` 文件**，避免磁盘被多个模型占满；每个实验仍会保留 metrics、曲线、混淆矩阵等分析结果。

全部运行结束后会自动生成：

```text
results/experiment_summary.csv
results/experiment_summary.md
```

Markdown 表格会在每个数据集内部按 **test accuracy 排名**，并展示：

- Test Accuracy
- Test Loss
- Best Validation Accuracy
- Parameters
- Epochs
- Training Time

如果已经有各实验的 `metrics.json`，不想重新训练，可以直接：

```bash
python experiments/summarize_results.py
```

重新生成比较表。

## 原课程设计实验结果

以下数据来自 2024 年课程设计报告中的运行截图，属于**历史课程实验结果**，不是新版代码重新训练出的 benchmark。

| 数据集 | 实验 | 原报告准确率 |
|---|---|---:|
| CIFAR-10 | 三层卷积基础模型 | 58.6% |
| CIFAR-10 | filters 16 -> 32 -> 16 | 51.5% |
| CIFAR-10 | filters 32 -> 64 -> 64 | 55.7% |
| CIFAR-10 | 两层卷积模型 | 55.5% |
| CIFAR-10 | 四层卷积模型 | 60.3% |
| CIFAR-10 | 四层卷积 + 更长训练 | **64.3%** |
| MNIST | 原基础模型 | 98.1% |
| MNIST | 四层卷积 / 两层池化 | 98.7% |
| MNIST | 最后一层池化改为 1x1 | 98.3% |
| MNIST | 增加训练次数 | **99.0%** |

完整历史实验说明见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 可复现性说明

- 默认 random seed：`42`
- 在 TensorFlow/backend 支持时启用 deterministic operations
- train / validation 采用确定性的**按类别分层划分**
- CIFAR-10 / MNIST 官方 test set 独立保留，不参与训练/验证
- 不同 TensorFlow 版本、CPU/GPU/Metal 后端和浮点内核仍可能导致轻微结果差异
- 原课程报告的历史结果和新版程序产生的新实验结果严格分开说明

## 轻量质量检查

不安装 TensorFlow，也可以先检查仓库本身：

```bash
python -m compileall -q src experiments tests
python -m unittest discover -s tests -v
```

GitHub Actions 在 push / pull request 时也会自动运行这些检查。CI 没有假装执行完整 CNN 训练，因为完整 TensorFlow 训练明显比仓库级质量检查更重。

## 相比原课程代码做了什么改进？

原始 `.py` 文件存在格式、缩进、变量初始化、本地绝对路径、测试集归一化和手工预测统计等问题。新版在保留课程设计目标的基础上重新实现：

- 删除 `venv/`、`.idea/`、本地数据集等不应上传 GitHub 的内容；
- 去除 `D:\...` Windows 硬编码路径；
- 统一训练集和测试集归一化；
- 修复原代码 SyntaxError、缩进和变量问题；
- 使用完整测试集评估，而不是脆弱的手工前 1000 条循环；
- 增加 random seed 和分层 train/validation split；
- 将 CNN 结构整理成命名 preset；
- 通过 CLI 控制训练和超参数；
- 输出每类 Precision/Recall/F1、混淆矩阵和正确/错误预测样例；
- 批量实验后自动生成架构比较表；
- 记录模型大小、耗时和运行环境，提升实验可追溯性。

## 原课程报告

公开版课程报告保存在 [`docs/original/2024-course-report-cn.pdf`](docs/original/2024-course-report-cn.pdf)。为了 Public Repository 的隐私安全，**学号已通过真正的 PDF redaction 从底层内容中移除**；姓名及课程技术内容保留，用于证明项目来源。

更多说明见 [`docs/ORIGINAL_WORK.md`](docs/ORIGINAL_WORK.md)。

## 技术栈

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- CIFAR-10
- MNIST

## 作者

**Shuoxun Wen（温硕勋）**

本仓库由本人早期机器学习课程设计整理、修复并重构，用于技术 Portfolio 展示和可复现实验。
