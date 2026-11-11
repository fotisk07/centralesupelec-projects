# VIC Project: Classical Face Recognition Under Real-World Variations

## Overview

This project studies **classical face recognition methods** and evaluates their robustness under real-world variations such as illumination changes, noise, blur, and occlusions.

The implemented models follow a **Scikit-learn–compatible API**, allowing standardized training, evaluation, and comparison.

## Download the Data

First, download the datasets (ORL and Yale):

```bash
bash data/download.sh
```

## Running the Project

This project uses **uv** as its package manager. To run any script:

```bash
uv run my_script.py
```

## Project Structure

```
src/
├── dataloader.py   # Utilities for loading ORL and Yale datasets  
├── metrics.py      # Evaluation metrics and analysis tools  
├── models.py       # Four implemented models (Scikit-learn API compatible)  
└── transform.py    # Data transformation and augmentation scripts  
```

## Notebooks

The provided notebooks demonstrate:

- How to train and evaluate each model  
- How to generate performance curves  
- How to test robustness under controlled degradations  
- How to compare classical methods against a simple deep learning baseline  
