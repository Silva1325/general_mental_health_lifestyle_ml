# general_mental_health_lifestyle_ml

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1+cpu-EE4C2C)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900)

## Project Overview

A machine learning project that uses a linear regression model to predict mental health outcomes based on lifestyle and behavirol factors. With this analysis we can also identify which parameters are the most beneficial and prejudicial for our  emotional well-being.

## Dataset

The dataset chosen for this project is [Mental Health](https://www.kaggle.com/datasets/mabubakrsiddiq/general-mental-health-and-lifestyle-dataset/data) from Kaggle.

This dataset represents synthetic daily mental health and lifestyle data for general working individuals. It captures realistic patterns of sleep, screen time, exercise, work load, and social interactions that affect mood, stress, and overall productivity.

Each record corresponds to a single day for a hypothetical worker, with features including:

- **Sleep Hours** – total hours slept
- **Screen Time –** daily time spent on digital devices
- **Exercise Minutes –** physical activity per day
- **Daily Pending Tasks –** unfinished work items or responsibilities
- **Interruptions –** number of daily interruptions
- **Fatigue Level –** subjective fatigue rating
- **Social Hours –** time spent interacting with family, friends, or coworkers
- **Coffee Cups –** caffeine intake
- **Diet Quality –** categorical rating of meals: poor, average, or good
- **Weather –** categorical: sunny, cloudy, rainy, snowy
- **Mood Score –** daily mood rating (1–10)
- **Stress Level –** daily stress rating (1–10)

## Installation

1. Clone the repository:
```bash
git clone 
cd GENERAL_MENTAL_HEALTH_LIFESTYLE_ML
```

2. Install required dependencies:
```bash
pip install numpy
pip install pandas
pip install -U scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install matplotlib
pip install mpl_toolkits
```

## Usage

1. Run the linear regression model:
```bash
python linear_regression.py
```

## Methodology

### 1. Data Preprocessing

Data preprocessing is a critical foundation for any machine learning project, directly impacting model accuracy, training efficiency, and prediction reliability. My preprocessing pipeline consisted of the following steps:
 1. **Handling Missing Values:** Removed all rows containing missing values to ensure data completeness and prevent errors during model training.
 2. **Removing Duplicates:** Duplicate entries were identified and removed to avoid bias and redundancy in the dataset.
 3. **Outlier Detection and Removal:** Applied the Interquartile Range (IQR) method to detect and remove outliers. This involved calculating the first quartile (Q1) and third quartile (Q3), then computing IQR = Q3 - Q1. Data points falling below Q1 - 1.5×IQR or above Q3 + 1.5×IQR were filtered out to eliminate extreme values that could skew the model.
 4. **Feature Engineering:** Prepared raw data for the model. It involves transforming and preparing features to make them suitable for the algorithm while preserving or enhancing their predictive power.
    1. **Encoding Categorical Variables:** Applied OneHotEncodeing to two categorical features in order to transform text into numerical format so that our ML can use them to detect patterns:
       - diet_quality: [poor, average, good] → [0,0,1], [1,0,0], [0,1,0]
       - weather: [snowy, sunny, rainy, cloudy] → [0,1,0,0], [0,0,1,0], [0,1,0,0], [1,0,0,0]
    2. **Feature Scaling:** Since the model uses Stochastic Gradient Descent (SGD) for optimization, feature scaling was essential. SGD is sensitive to feature magnitudes—unscaled features with different ranges can cause slow or unstable convergence. I standardized all input features using StandardScaler. The target variables were not scaled as they already share the same range (1-10), making scaling unnecessary for the outputs.
   
### 2. Model Selection

Linear regression is a supervised machine learning model that estimates a linear relationship between independent variables and a dependent variable:

$$
y = f[x, φ]
$$
$$
  = φ₀ + φ₁x₁ + φ₂x₂ + φ₃x₃ + ... + φₚxₚ
$$

Where:
- **y** is the output (dependent variable)
- **x₁, x₂, x₃, ..., xₚ** are the inputs (independent variables)
- **φ₀** is the intercept (bias term)
- **φ₁, φ₂, φ₃, ..., φₚ** are the coefficients (weights)
- **φ** represents all parameters {φ₀, φ₁, φ₂, ..., φₚ} 

In this case, I'm predicting two dependent variables, mood score and stress level, based on the independent variables, making this a multiple linear regression. So we would have something like this:

y₁ = f₁[x, φ¹] = φ₀¹ + φ₁¹x₁ + φ₂¹x₂ + ... + φₚ¹xₚ
y₂ = f₂[x, φ²] = φ₀² + φ₁²x₁ + φ₂²x₂ + ... + φₚ²xₚ

Where:
- **y₁** is the output of mood score
- **y₂** is the output of stress level

### 3. Feature Analysis
- **Numerical Features:** Continuous variables (age, sleep hours, exercise frequency, etc.)
- **Categorical Features:** Discrete variables (gender, occupation, lifestyle factors)
- Correlation analysis to identify key predictors

### 4. Model Training & Evaluation
- Training on historical data with lifestyle and behavioral features
- Performance metrics: Mean Squared Error (MSE), R² score
- Visualization of training loss and prediction accuracy

### 5. Visualization & Interpretation
- 2D and 3D loss surface plots for optimization analysis
- Feature relationship plots to understand parameter impacts
- Categorical feature boxplots for group comparisons

















