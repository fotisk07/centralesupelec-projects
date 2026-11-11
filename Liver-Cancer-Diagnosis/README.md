# Liver Cancer Diagnosis Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project aims to classify liver cancer types, specifically **Cholangiocarcinoma (CCK)** and **Hepatocellular Carcinoma (CHC)**, using machine learning techniques such as **Principal Component Analysis (PCA)**, **Logistic Regression**, and **Random Forests**. The dataset includes **147 tumors** across **146 patients** with four different phases: **PORT, ART, VEIN**, and **TARD**.

Our ultimate goal is to enhance diagnostic accuracy and provide insights for medical professionals to better understand liver cancer types.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Project Structure

```
├── data_analysis.ipynb        # Main analysis notebook
├── datasets/                  # Data folder (not included in this repo)
├── README.md                  # Project README
└── images/                    # Folder for visual outputs and figures
```

## Dataset

The dataset contains **147 tumors** across **146 patients**, categorized into different classes based on tumor type. We focus on a binary classification problem distinguishing **CCK** from **CHC** tumors. Each tumor is analyzed across four medical imaging phases:

1. **PORT**
2. **ART**
3. **VEIN**
4. **TARD**

Each tumor has **428 variables**, which are reduced through dimensionality reduction techniques before classification.

## Models

Several machine learning models are used to analyze and classify the liver tumor data:

- **Principal Component Analysis (PCA)**: Used to reduce the dimensionality of the dataset from 428 variables to 2 principal components (PC1 and PC2).
    - Variances explained:
        - PC1: 39.2%
        - PC2: 22.23%
    - Cumulative variance: 61.43%
  
- **Logistic Regression**: A simple and interpretable model for binary classification.
  
- **Random Forest**: A more complex, non-linear model to enhance classification performance and account for variable interactions.

## How to Run

### Prerequisites

1. Install [Python](https://www.python.org/downloads/) (>= 3.7).
2. Install necessary libraries by running:

```bash
pip install -r requirements.txt
```

### Running the Project

To run the analysis, you can open the provided Jupyter notebook and run the cells sequentially:

```bash
jupyter notebook data_analysis.ipynb
```

Make sure to have the dataset in the correct directory path (`datasets/`).

### Key Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib

## Results

### PCA Visualization

Below is a visualization of the data reduced to two dimensions using **PCA**. The distinction between the CCK and CHC tumors is clearly visible.

![PCA Plot](images/pca_visualization.png)

### Logistic Regression vs. Random Forest

- **Logistic Regression**:
    - A more interpretable model but may struggle with complex data interactions.
  
- **Random Forest**:
    - A robust model providing better performance with nonlinear relationships in the dataset.

## Future Work

- **Mixed Tumor Types**: Expand the model to handle mixed tumors (CHC + CCK).
- **Larger Datasets**: Train the models on a larger dataset to improve generalization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can include images by downloading visuals (like a PCA plot) or generating results from your notebook and saving them in the `images/` folder to display in the README.

Let me know if you want specific visuals or customizations for this README!
