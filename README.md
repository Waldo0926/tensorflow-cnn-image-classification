# TensorFlow CNN Image Classification

[中文说明](README.zh-CN.md)

A portfolio-oriented reconstruction of a 2024 machine-learning course design exploring **convolutional neural networks (CNNs)** for image classification on **CIFAR-10** and **MNIST** with TensorFlow/Keras.

The repository preserves the original experiment questions - network depth, filter counts, pooling choices, kernel sizes, and training duration - while replacing the old coursework scripts with a cleaner, reproducible experiment pipeline.

## What this project demonstrates

- End-to-end TensorFlow/Keras image-classification workflow
- CIFAR-10 and MNIST loaded automatically through Keras
- Deterministic seed control and class-stratified train/validation splitting
- Named CNN architecture presets derived from the original coursework
- Full test-set evaluation with accuracy, loss, macro precision/recall/F1, and per-class metrics
- Learning curves, confusion matrix, and explicit **correct vs. incorrect prediction examples** with confidence scores
- Automatic multi-model experiment comparison in CSV and Markdown
- Model parameter counts, training time, environment metadata, and model summaries
- Lightweight GitHub Actions checks for syntax and dependency-light tests
- Public archive of the original Chinese course report with the student number truly redacted for privacy

## Project structure

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

## Quick start

### 1. Create a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Use a Python version supported by your TensorFlow release; Python 3.11 or 3.12 is a conservative choice for this project.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. List available model presets

```bash
python src/train.py --list-presets
```

### 4. Train one model

CIFAR-10 baseline:

```bash
python src/train.py --dataset cifar10 --preset baseline --epochs 20
```

MNIST four-convolution/two-pooling experiment:

```bash
python src/train.py --dataset mnist --preset four_conv_two_pool --epochs 15
```

Quick CPU smoke run:

```bash
python src/train.py --dataset mnist --preset baseline --epochs 2 --train-limit 5000
```

Keras downloads the selected dataset automatically on first use. `--train-limit` limits only the training subset; validation and test sets remain intact for more meaningful evaluation.

## Command-line controls

The main CLI supports experiment parameters without editing source code:

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

Useful options include:

- `--dataset {cifar10,mnist}`
- `--preset <name>`
- `--epochs <n>`
- `--batch-size <n>`
- `--learning-rate <value>`
- `--validation-fraction <0..1>`
- `--train-limit <n>` for quick local tests
- `--patience <n>` for optional early stopping
- `--seed <n>`
- `--output-dir <path>`
- `--no-save-model` to export metrics/figures without a large `.keras` file

## Per-run outputs

A standard run writes to `results/<dataset>/<preset>/`:

```text
metrics.json
history.csv
per_class_metrics.csv
model_summary.txt
training_curves.png
confusion_matrix.png
prediction_examples.png
model.keras                 # omitted when --no-save-model is used
```

`prediction_examples.png` intentionally samples both correctly and incorrectly classified test images, and shows the predicted-class confidence for each example.

`metrics.json` also records model parameter count, training time, dataset split sizes, random seed, hyperparameters, and Python/TensorFlow/NumPy environment versions.

Generated outputs are ignored by Git so experiment artefacts and trained models do not accidentally bloat the repository.

## Compare multiple CNN architectures

Run the documented preset matrix:

```bash
python experiments/run_experiments.py --dataset all --epochs 20
```

For a faster comparison:

```bash
python experiments/run_experiments.py --dataset all --epochs 3 --train-limit 10000
```

The batch runner avoids saving every trained model by default, then automatically creates:

```text
results/experiment_summary.csv
results/experiment_summary.md
```

The Markdown summary ranks models **within each dataset** by test accuracy and includes test loss, best validation accuracy, parameter count, epochs, and training time.

If individual runs already exist, rebuild the comparison table without retraining:

```bash
python experiments/summarize_results.py
```

## Historical course-design results

The archived report recorded the following prediction accuracies. These are **historical results from the 2024 coursework**, not newly regenerated benchmark claims.

| Dataset | Experiment | Reported accuracy |
|---|---|---:|
| CIFAR-10 | Original 3-conv baseline | 58.6% |
| CIFAR-10 | Filters 16 -> 32 -> 16 | 51.5% |
| CIFAR-10 | Filters 32 -> 64 -> 64 | 55.7% |
| CIFAR-10 | Two-convolution model | 55.5% |
| CIFAR-10 | Four-convolution model | 60.3% |
| CIFAR-10 | Four-convolution model, longer training | **64.3%** |
| MNIST | Original baseline | 98.1% |
| MNIST | Four conv / two pool | 98.7% |
| MNIST | Final pooling changed to 1x1 | 98.3% |
| MNIST | Longer training | **99.0%** |

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for historical context and the corresponding reproducible presets.

## Reproducibility notes

- The default seed is `42`.
- TensorFlow deterministic operations are requested where the installed backend supports them.
- The train/validation split is deterministic and class-stratified.
- CIFAR-10/MNIST test sets are kept separate from training and validation.
- Exact floating-point results can still vary slightly across TensorFlow versions, hardware backends, and accelerator kernels.
- Historical report values are kept separate from results produced by the reconstructed code.

## Lightweight quality checks

Without installing TensorFlow, you can still run the dependency-light repository checks:

```bash
python -m compileall -q src experiments tests
python -m unittest discover -s tests -v
```

The same checks run in GitHub Actions on pushes and pull requests. Full TensorFlow training is intentionally not run in CI because it is substantially heavier than the repository-level checks.

## From coursework to portfolio project

The original exported scripts contained syntax/formatting issues, hard-coded local Windows paths, inconsistent test normalisation, and fragile manual evaluation logic. This reconstruction keeps the original learning objectives while improving software and experiment quality:

- no bundled `venv/`, IDE metadata, or local dataset copies;
- no hard-coded `D:\...` paths;
- consistent normalisation for train/test images;
- stratified train/validation split and seed control;
- full test-set evaluation rather than ad-hoc partial loops;
- named model presets and CLI-driven hyperparameters;
- richer evaluation artefacts and automatic architecture comparisons;
- generated models/results separated from tracked source files.

## Original coursework archive

The public report is preserved at [`docs/original/2024-course-report-cn.pdf`](docs/original/2024-course-report-cn.pdf). The student's **student number has been removed with true PDF redaction** for the public repository; the remaining academic content is retained as provenance.

See [`docs/ORIGINAL_WORK.md`](docs/ORIGINAL_WORK.md) for details.

## Tech stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- CIFAR-10
- MNIST

## Author

**Shuoxun Wen (温硕勋)**

Reconstructed from the author's earlier machine-learning coursework for portfolio presentation and reproducible experimentation.
