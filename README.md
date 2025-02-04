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

Usage:

Place the creditcard.csv file in the same directory as the script.
Run the script from the command line:
bash
Copy
Edit
python fraud_detection.py
The script will load the data, perform analysis and visualization, train the models, and then output the evaluation metrics for each model.
Code Structure:

Data Loading & Exploration:
Loads and inspects the dataset, checks for missing values, and summarizes data statistics.
Visualization:
Plots transaction class distribution.
Creates histograms for transaction amounts for both fraudulent and normal transactions.
Generates scatter plots for transaction time vs. amount.
Displays a correlation heatmap.
Data Sampling:
Uses a 10% random sample of the dataset for model evaluation.
Model Training & Prediction:
Implements three models (Isolation Forest, Local Outlier Factor, and One-Class SVM) and adjusts predictions to classify transactions as normal (0) or fraudulent (1).
Evaluation:
Computes accuracy scores and generates classification reports for each model.
Results & Observations:

Isolation Forest:
Detected 73 misclassifications.
Achieved an accuracy of approximately 99.74%.
Local Outlier Factor:
Detected 97 misclassifications.
Achieved an accuracy of approximately 99.65%.
One-Class SVM:
Performed significantly worse with 8516 misclassifications.
Achieved an accuracy of around 70.09%.
Observations:

The Isolation Forest method outperformed the other models in terms of precision, recall, and overall accuracy.
Detection rates for fraudulent transactions were significantly higher with Isolation Forest compared to LOF and SVM.
Future Work:

Scaling Up: Increase the sample size or use the entire dataset to further validate model performance.
Advanced Models: Experiment with deep learning techniques or more sophisticated anomaly detection algorithms for potentially better performance.
Feature Engineering: Incorporate additional features or advanced preprocessing techniques to enhance model accuracy.
Contributing: Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

License: This project is licensed under the MIT License.

Acknowledgments:

scikit-learn Documentation
pandas Documentation
matplotlib Documentation
seaborn Documentation
Copy
Edit

