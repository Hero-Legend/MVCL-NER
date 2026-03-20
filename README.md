# MVCL-NER: Adaptive Gated Multi-View Contrastive Learning for Lightweight Legal Entity Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
This is the official PyTorch implementation for the paper: **"Adaptive Gated Multi-View Contrastive Learning for Lightweight Legal Entity Recognition"**.

## 💡 Overview

Structured information extraction from judicial documents is a cornerstone of computational law. However, existing methods either suffer from severe feature pollution when integrating external knowledge or face unpredictable boundary drift and high inference latency when deploying Large Language Models (LLMs). 

To resolve the **"CRF Dilemma"** and achieve real-time, privacy-preserving deployment, we propose **MVCL-NER**. 

### ✨ Core Highlights:
- **Adaptive Gated Fusion:** Dynamically integrates explicit Lexical (BMES) and Structural (POS) views to prevent feature pollution.
- **Token-Level Contrastive Alignment:** Employs SupConLoss to tightly cluster heterogeneous judicial features, perfectly handling the extreme long-tail distribution of legal entities.
- **CRF-Free & Lightweight:** Achieves a **3.1$\times$ inference speedup** (14.5ms/seq) over traditional CRF models while maintaining a state-of-the-art Strict F1-score (86.11%).
- **Crushing Generative LLMs:** Comprehensively outperforms leading LLMs (DeepSeek, Qwen, Kimi) in exact-match boundary constraints while requiring $<1$GB VRAM for 100% on-premise privacy-preserving deployment.

## 🏛️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="MVCL-NER Architecture" width="85%">
</p>

> **Figure:** The extraction-fusion-alignment architecture of MVCL-NER. It harmonizes pre-trained semantics with lexical/structural boundaries, enabling a lightweight linear classifier to bypass the heavy CRF decoder entirely.

## 🚀 Main Results

Performance on the **CAIL2021 Judicial Benchmark** (Strict Exact-Match F1):

| Model Paradigm | Parameters | Latency (ms/seq) | F1-Score |
| :--- | :---: | :---: | :---: |
| Generative LLM (DeepSeek) | > 100B | ~850.0 | 51.24% |
| PLM Baseline (RoBERTa) | 110M | 45.8 | 85.06% |
| Domain PLM (BOCNER) | 110M | 46.5 | 85.50% |
| **MVCL-NER + CRF (Ours)** | **110M** | **46.2** | **85.83%** |
| **MVCL-NER (Ours, Full)** | **110M** | **14.5** | **86.11%** |

## 🛠️ Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/Hero-Legend/MVCL-NER.git](https://github.com/Hero-Legend/MVCL-NER.git)
cd MVCL-NER
