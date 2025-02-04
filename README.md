# Fraud Detection System

This project implements a fraud detection system using Python. It leverages several unsupervised anomaly detection algorithms to identify fraudulent transactions in a credit card dataset. The system demonstrates data exploration, visualization, and model evaluation to showcase various skills in data analysis and machine learning.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results & Observations](#results--observations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project uses three anomaly detection methods to flag potential fraudulent credit card transactions:
- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class Support Vector Machine (SVM)**

The project not only applies these models but also includes thorough data exploration and visualization to understand transaction patterns.

## Features

- **Data Exploration:**
  - Loading and inspecting the dataset with pandas.
  - Checking for missing values and summarizing basic statistics.
- **Visualization:**
  - Plotting the transaction class distribution.
  - Creating histograms for transaction amounts in both fraudulent and normal transactions.
  - Scatter plots to analyze the relationship between transaction time and amount.
  - Correlation heatmap to study feature relationships.
- **Anomaly Detection Models:**
  - Implements Isolation Forest, Local Outlier Factor, and One-Class SVM.
  - Converts model predictions to indicate 0 (normal) and 1 (fraudulent).
- **Evaluation:**
  - Generates accuracy scores and detailed classification reports including precision, recall, and F1-score for each model.

## Dataset

The project uses the `creditcard.csv` dataset, which contains credit card transactions labeled as normal or fraudulent. Ensure that the CSV file is in the same directory as the script.

## Installation

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Dependencies

Install the required Python libraries using pip:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn
