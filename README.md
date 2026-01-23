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

- Sleep Hours – total hours slept
- Screen Time – daily time spent on digital devices
- Exercise Minutes – physical activity per day
- Daily Pending Tasks – unfinished work items or responsibilities
- Interruptions – number of daily interruptions
- Fatigue Level – subjective fatigue rating
- Social Hours – time spent interacting with family, friends, or coworkers
- Coffee Cups – caffeine intake
- Diet Quality – categorical rating of meals: poor, average, or good
- Weather – categorical: sunny, cloudy, rainy, snowy
- Mood Score – daily mood rating (1–10)
- Stress Level – daily stress rating (1–10)

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
- Handling Missing Values: I removed all rows containing missing values to ensure data completeness and prevent errors during model training.
- Removing Duplicates: Duplicate entries were identified and removed to avoid bias and redundancy in the dataset.
- Outlier Detection and Removal: I applied the Interquartile Range (IQR) method to detect and remove outliers. This involved calculating the first quartile (Q1) and third quartile (Q3), then computing IQR = Q3 - Q1. Data points falling below Q1 - 1.5×IQR or above Q3 + 1.5×IQR were filtered out to eliminate extreme values that could skew the model.
- Encoding Categorical Variables: Two categorical features required transformation into numerical format:
   diet_quality: [poor, average, good]
   weather: [snowy, sunny, rainy, cloudy]
- Feature Scaling: Since the model uses Stochastic Gradient Descent (SGD) for optimization, feature scaling was essential. SGD is sensitive to feature magnitudes—unscaled features with different ranges can cause slow or unstable convergence. I standardized all input features using StandardScaler. The target variables were not scaled as they already share the same range (1-10), making scaling unnecessary for the outputs.
   


















