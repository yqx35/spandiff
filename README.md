# spandiff
spandiff:Relation-Aware Span Diffusion for Nested Named Entity Recognition

# SpanDiff: Relation-Aware Span Diffusion for Nested Named Entity Recognition

This repository contains the official PyTorch implementation, training scripts, and pretrained weights for **SpanDiff**.

## 1. Requirements

This code requires **Python 3.8**. We recommend setting up a virtual environment using Conda.

**Create and activate the environment:**
```bash
conda create -n spandiff python=3.8
conda activate spandiff

```

**Install Core Dependencies (PyTorch):**
The model requires PyTorch with CUDA support (tested on `1.10.0+cu111`).

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0+rocm4.1 -f [https://download.pytorch.org/whl/cu111/torch_stable.html](https://download.pytorch.org/whl/cu111/torch_stable.html)

```

**Install Other Dependencies:**

```bash
pip install -r requirements.txt

```

## 2. Pre-trained Language Models (PLMs) Setup

Our experiments utilize standard PLMs. Please ensure you have the appropriate weights downloaded or allow the `transformers` library to fetch them automatically:

* **BERT:** `bert-base-uncased`
* *(List any other specific PLMs used, e.g., SciBERT, RoBERTa)*

## 3. Pretrained Weights & Sample Data

To facilitate immediate independent verification, we provide pre-trained model weights and sample input/output files.

* **Download Weights:** [Google Drive Link](https://drive.google.com/drive/folders/17eakwaw0D2AwFoHegQzHSSqdDy7uYnAT?usp=drive_link)
Place the downloaded weights in the `./checkpoints/` directory.

## 4. Evaluation / Inference

To run evaluation or inference using the provided sample data and pretrained weights, execute the following command. This will output the diffusion-based iterative boundary refinement results.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python spandiff.py eval --config configs/eval.conf

```

## 5. Training

To train SpanDiff from scratch on your desired dataset, you can use the provided configuration files. For example, to train on the ACE2004 dataset, run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python spandiff.py train --config configs/ace2004.conf


