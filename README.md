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

#### What is Supervised Learning?

A supervised learning model defines a mapping from one or more inputs to one or more outputs. For example, the input might be the mileage and age of a car in order to predict its market value.

The model is just a mathematical function. When the inputs are passed through this function, it computes the output, and this is termed inference. The model equation describes a family of possible relationships between the input and output, and the parameters specify the particular relationship.

When we train a model, we find parameters that describe the true relationship between the inputs and outputs. A learning algorithm takes a training set of input/output pairs and manipulates their parameters until their inputs predict the output as closely as possible.
If the model works well for these training pairs, then we hope for it to work well with new data to make good predictions.

### Linear regression

Linear regression is a supervised machine learning model that estimates a linear relationship between independent variables and a dependent variable:

$$
\begin{aligned}
y &= f[x, φ] \\
  &= φ₀ + φ₁x₁ + φ₂x₂ + φ₃x₃ + ... + φₚxₚ
\end{aligned}
$$

Where:
- **y** is the output (dependent variable)
- **x₁, x₂, x₃, ..., xₚ** are the inputs (independent variables)
- **φ₀** is the intercept (bias term)
- **φ₁, φ₂, φ₃, ..., φₚ** are the coefficients (weights)
- **φ** represents all parameters {φ₀, φ₁, φ₂, ..., φₚ} 

In this case, I'm predicting two dependent variables based on the independent variables, making this a multiple linear regression. 
- **Independent variables:** Sleep hours, screen time, exercise minutes, daily pending tasks, interruptions, fatigue level, social hours, coffee cups, diet quality, weather
- **Dependent variables:** Mood score, stress level

Being that said, mathematically, we would have something like this:

$$
\begin{aligned}
y₁ &= f₁[x, φ¹] \\
   &= φ₀¹ + φ₁¹x₁ + φ₂¹x₂ + ... + φₚ¹xₚ
\end{aligned}
$$

$$
\begin{aligned}
y₂ &= f₂[x, φ²] \\
   &= φ₀² + φ₁²x₁ + φ₂²x₂ + ... + φₚ²xₚ
\end{aligned}
$$

Where:
- **y₁** is the output of mood score
- **y₂** is the output of stress level
- Where each output yᵢ has its own set of parameters φⁱ = {φ₀ⁱ, φ₁ⁱ, φ₂ⁱ, ..., φₚⁱ}.

### 3. Feature Analysis

The feature analysis reveals comprehensive insights into how various lifestyle and environmental factors influence mental health outcomes (mood score and stress level). This multi-dimensional analysis examines both numerical and categorical variables to understand their predictive power.

The model analyzes eight continuous variables that capture daily behavioral patterns and their impact on mental wellbeing:
- **Sleep Hours:** Sleep duration shows a direct correlation with mood scores and inverse relationship with stress levels, highlighting the fundamental role of rest in mental health
- **Exercise Minutes:** Physical activity emerges as a protective factor, with increased exercise correlating with improved mood and reduced stress
- **Interruptions:** Frequency of daily interruptions shows detrimental effects on both mental health metrics
- **Social Hours:** Time spent in social interactions demonstrates positive effects on mood while potentially reducing stress levels
- **Screen Time:** Prolonged screen exposure demonstrates negative associations with mood and positive correlation with stress, reflecting modern digital lifestyle impacts
- **Daily Pending Tasks:** Task burden exhibits strong positive correlation with stress levels and negative association with mood scores
- **Fatigue Level:** Self-reported fatigue serves as both a predictor and outcome variable, closely linked to stress and mood fluctuations
- **Coffee Cups:** Caffeine consumption patterns reveal complex, potentially non-linear relationships with mental health outcomes

<img src="data_analysis\02_numerical_features_relationships.png" width="1000">


#### Sleep Hours

 - Overall there is a lot of density around the [6,8] interval which means that most of the people in this dataset sleep around 6-8 hours a day.
 - **Mood score:** We can clearly see a trend emerging in this graph where the more you sleep the better your mood will be. Individuals that sleep below 6 hours a day tend to have a worse mood. Individuals that sleep more than 6 hours tend to have a better mood.
 - **Stress level:** Here we have a similar behaviour. Individuals with less sleep hours show higher stress levels. As sleep hours increase, stress levels decrease significantly. The pattern shows an inverse relationship where adequate sleep (7-9 hours) is associated with lower stress levels (1-4 range), while sleep deprivation (<5 hours) correlates with elevated stress (4-9 range).

#### Screen Time
 - The data shows a dense concentration of points across various screen time levels (0-12 hours), with most people falling in the 4-8 hour range.
 - **Mood score:** There is a negative correlation between screen time and mood. Individuals with high screen time (8-12 hours) tend to have lower mood scores, predominantly in the 3-6 range. Those with moderate screen time (2-4 hours) show better mood scores, typically in the 6-8 range.
 - **Stress level:** Screen time shows a positive correlation with stress. Higher screen exposure (8-12 hours) is associated with elevated stress levels (4-7 range), while lower screen time corresponds to reduced stress levels (1-3 range).

#### Exercise Minutes
 - The distribution shows wide variation in exercise habits, ranging from sedentary (0-20 minutes) to highly active (120+ minutes).
 - **Mood score:** There is a strong positive correlation between exercise and mood. Individuals who exercise regularly (60-120 minutes) show significantly higher mood scores (7-9 range). Sedentary individuals (<20 minutes) tend to have lower mood scores (3-5 range).
 - **Stress level:** Exercise shows an inverse relationship with stress. Regular physical activity (60+ minutes) is associated with lower stress levels (1-3 range), while minimal exercise correlates with higher stress (5-8 range).




 
### 4. Model Training & Evaluation
- Training on historical data with lifestyle and behavioral features
- Performance metrics: Mean Squared Error (MSE), R² score
- Visualization of training loss and prediction accuracy

### 5. Visualization & Interpretation
- 2D and 3D loss surface plots for optimization analysis
- Feature relationship plots to understand parameter impacts
- Categorical feature boxplots for group comparisons

















