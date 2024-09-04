# Comparative Analysis of ML Algorithms for Conversion Predictors of CIS to Multiple Sclerosis

## Overview
This project aims to perform a comparative analysis of various machine learning algorithms to predict the conversion of Clinically Isolated Syndrome (CIS) to Multiple Sclerosis (MS). The analysis includes preprocessing, data visualization, and the evaluation of multiple classification models to determine the most effective predictor.

## Features

- **Preprocessing**: 
  - **Data Cleaning**: Missing values were handled, and outliers were addressed to ensure data quality.
  - **Feature Scaling**: Numerical features were normalized to bring them onto a similar scale, improving model performance.
  - **Encoding Categorical Variables**: Categorical features were converted into numerical values using one-hot encoding, allowing them to be utilized in the machine learning models.

- **Data Visualization**:
  - **Exploratory Data Analysis (EDA)**: Various plots and graphs were created to understand the distribution of the data, identify patterns, and explore relationships between features.
  - **Correlation Matrix**: A heatmap was generated to visualize the correlations between different features, helping to identify the most influential predictors.

- **Model Evaluation Metrics**:
  - **Confusion Matrix**: Used to visualize the performance of each model, showing the breakdown of true positives, false positives, true negatives, and false negatives.
  - **Accuracy**: Calculated as the ratio of correctly predicted instances to the total instances, giving an overall measure of model correctness.
  - **Precision**: Assesses the accuracy of positive predictions, calculated as the ratio of true positives to the sum of true positives and false positives.
  - **Recall**: Measures the model's ability to correctly identify all relevant instances, calculated as the ratio of true positives to the sum of true positives and false negatives.
  - **F1 Score**: Combines precision and recall into a single metric, providing a balanced measure, especially useful for imbalanced datasets.

- **Machine Learning Algorithms**:
  - **Random Forest**: An ensemble learning method that creates multiple decision trees and merges their results for more accurate predictions.
  - **Logistic Regression**: A linear model that predicts the probability of a binary outcome, in this case, the conversion from CIS to MS.
  - **KNeighborsClassifier**: A non-parametric method that classifies based on the majority class among the k-nearest neighbors.
  - **Gaussian Naive Bayes (Gaussian NB)**: A probabilistic classifier based on Bayes' theorem with strong independence assumptions between features.
  - **Support Vector Classifier (SVC)**: A powerful classification technique that finds the optimal hyperplane to separate classes.
  - **XGBClassifier**: A high-performance implementation of the eXtreme Gradient Boosting algorithm, known for its efficiency and accuracy in classification tasks.

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, and other necessary dependencies.

### Installation
Clone the repository and install the required libraries using `pip`.

```bash
git clone <repository-url>
cd CIS-to-MS-conversion-predictors
pip install -r requirements.txt
