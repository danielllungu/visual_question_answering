# Visual Question Answering (VQA)

*A vision–language project that answers free‑form questions about images by combining visual perception with natural‑language understanding.*

## Overview
This repository explores **Visual Question Answering (VQA)**: teaching a model to answer a natural‑language question about a given image. The system combines a **vision encoder** for extracting visual features and a **language encoder** for understanding the text question, then fuses both to predict the answer.

The project contains two major parts:
- `VQAExperiments/` — research code for models, training, and evaluation.
- `VqaWeb/` — a simple web interface for interactive VQA demos.


## Features
- Modular pipeline for **vision** and **language** processing.
- Reproducible **training/evaluation** scripts (inside `VQAExperiments/`).
- Lightweight **web demo** (inside `VqaWeb/`) to try the model on your own images/questions.

## Dataset: VQA v2
MS‑COCO VQA v2 is used for training and evaluation.
