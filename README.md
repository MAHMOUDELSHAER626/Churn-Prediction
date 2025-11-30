# Telco Customer Churn Prediction

A machine learning project to predict customer churn in the telecommunications industry using multiple classification algorithms with an interactive Gradio web interface.

## Overview

This project analyzes customer data to predict whether customers will churn (leave the service). It compares 5 different machine learning models and provides an interactive web interface for easy predictions.

## Dataset

- **Source**: Telco Customer Churn Dataset
- **Total Samples**: 10,348 customers
- **Features**: 25 (after preprocessing)
- **Target**: Churn (Yes/No)

## Models Evaluated

| Model | Accuracy | Precision | Recall | CV Score |
|-------|----------|-----------|--------|----------|
| Gradient Boosting | 99.52% | 99.13% | 99.90% | 0.900 |
| Random Forest | 97.44% | 95.17% | 99.90% | 0.899 |
| XGBoost | 96.91% | 94.54% | 99.51% | 0.883 |
| SVM | 84.78% | 81.60% | 89.47% | 0.810 |
| Logistic Regression | 76.28% | 75.60% | 77.00% | 0.759 |

**Best Model**: Gradient Boosting with 99.90% recall

## Key Features

- Comprehensive exploratory data analysis with 15 visualizations
- Data preprocessing with upsampling to handle class imbalance
- Grid search cross-validation for hyperparameter tuning
- Multiple evaluation metrics (Accuracy, Precision, Recall, AUC-ROC)
- Feature importance analysis
- Interactive Streamlit web interface for predictions

## Installation

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost streamlit
```

## Usage

### Training Models

```bash
python Churn_Last.py
```

This will:
- Perform exploratory data analysis
- Train 5 different models
- Generate 15 visualizations
- Save trained models as pickle files

### Running the Web Interface

```bash
python APP.py
```

The Streamlit interface will launch automatically 

### Making Predictions

The Streamlit interface provides two tabs:

**1. Single Customer Prediction:**
- Fill in customer information using interactive dropdowns and sliders
- Click "Predict Churn" to get instant results
- View churn probability and prediction status



## Key Insights

- **Contract Type**: Month-to-month contracts have highest churn rate
- **Tenure**: Customers with shorter tenure are more likely to churn
- **Internet Service**: Fiber optic customers show higher churn rates
- **Services**: Customers without OnlineSecurity and TechSupport churn more
- **Payment Method**: Electronic check users have higher churn rates

## Results

The Gradient Boosting model achieved:
- **99.90% Recall** - Successfully identifies churning customers
- **99.13% Precision** - Minimizes false positives
- **99.52% Accuracy** - Overall performance

## Visualizations Generated

1. Target distribution analysis
2. Binary features vs churn
3. Churn rates by features
4. Internet service analysis
5. Additional services impact
6. Contract and payment analysis
7. Numerical features distribution
8. Correlation heatmap
9. Class resampling comparison
10. Confusion matrices
11. Metrics comparison
12. ROC curves
13. Precision-recall curves
14. Performance radar chart
15. Feature importance

## Web Interface Features

- **User-Friendly**: No coding required to make predictions
- **Single Prediction**: Input individual customer data with interactive controls
- **Real-Time Results**: Instant prediction with probability scores


## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit

