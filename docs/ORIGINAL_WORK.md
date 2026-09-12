# Original Coursework Archive

This repository was reconstructed from a machine-learning course design completed in early 2024. The original topic was **image classification using convolutional neural networks**, with experiments on CIFAR-10 and MNIST using TensorFlow/Keras.

The public Chinese course-design report is preserved at:

- [`original/2024-course-report-cn.pdf`](original/2024-course-report-cn.pdf)

## Public-release privacy edit

The original submitted report contained the author's student number on the cover page. Because this repository is intended to be public, that number has been removed using **true PDF redaction**, which removes the underlying text rather than merely drawing a white rectangle over it.

The author's name and the technical/academic content remain visible so the report can still serve as provenance for the project.

## Relationship between the archive and `src/`

The old `.py` files are intentionally **not** used as runnable source code in this repository. They contained formatting, indentation, hard-coded path, normalisation and evaluation issues from the original coursework export.

The implementation under `src/` is therefore a cleaned reconstruction of the same experiment ideas. It should be read as:

1. evidence of the original coursework topic and experiments; and
2. a later engineering pass that makes those ideas reproducible and portfolio-ready.

The repository does **not** present the historical screenshots as freshly reproduced benchmark results. Historical values are clearly labelled in the README and [`EXPERIMENTS.md`](EXPERIMENTS.md).
