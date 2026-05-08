# Information Theory Project

**Bridging Computational Limits and Cognitive Architecture: Working Memory Limits and Representational Collapse in Transformers**

NYU | DS-GA 3001 / PSYCH-GA 3505: Information Theory and Cognition | Spring 2026

## Team

- Yila Cao (NYU Data Science, Graduate)
- Yichen Zhao (NYU Data Science, Undergraduate)

## Overview

This project replicates and extends Barbero et al. (2024), "Transformers need glasses! Information over-squashing in language tasks," and proposes **Miller's Window**, a cognitive prior restricting early-layer attention to mitigate representational collapse.

The repository contains the experimental notebooks for both the replication study (across GPT-2, Pythia-410M, Llama-3.1-8B, and Gemma-7B) and the Miller's Window extension trained on GPT-2.

## Structure

```
notebooks/
├── Yila_Representational_Collapse.ipynb         # Per-head L∞ collapse measurement across four architectures
├── Yila_Counting_complete.ipynb                 # Counting-task evaluation (4 tasks × 3 prompting modes × 4 models)
└── Yichen_reproduction_of_new_metric.ipynb      # Miller's Window prior, training and evaluation

RC_and_counting_results/                         # Cached CSVs and figures from RC + counting replication
├── collapse/
└── counting/
```

## Models

All models loaded under `torch.bfloat16` to match the precision floor of Barbero et al. (2024):

| Model | Variant | Context cap |
|---|---|---|
| GPT-2 | base | 1024 |
| Pythia-410M | base | 2048 |
| Llama-3.1-8B | Instruct | 8192 |
| Gemma-7B | -IT | 8192 |

## Reproducing Results

Notebooks are designed for Google Colab (A100/V100). Each notebook is self-contained — install dependencies in the first cell, then run sequentially. Cached results are saved as CSV alongside each notebook.

Cached results (per-model Δₙ measurements, counting-task accuracies, and generated figures) are stored in `RC_and_counting_results/` for reproducibility, so figures in the report can be regenerated without rerunning model inference.

## Reference

Barbero, F. et al. (2024). *Transformers need glasses! Information over-squashing in language tasks.* NeurIPS 37.
