# House Price Prediction using Linear Regression from Scratch

This project implements a complete Linear Regression model from scratch using:

- NumPy
- Pandas
- Linear Algebra
- Gradient Descent
- Normal Equation

The goal is to predict house prices based on multiple house features such as:

- Area
- Bedrooms
- Bathrooms
- Floors
- House Age
- Location
- Condition
- Garage Availability

---

# Project Objective

The main objective of this project is to understand how Linear Regression works internally without using machine learning libraries like:

- Scikit-Learn
- TensorFlow
- PyTorch

This project focuses on:

- Data preprocessing
- One-Hot Encoding
- Matrix representation
- Normal Equation
- Gradient Descent
- Weight optimization
- Manual prediction

---

# Dataset Features

| Feature | Description |
|---|---|
| Area | House area size |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Floors | Total floors |
| HouseAge | CurrentYear - YearBuilt |
| Location | Downtown / Suburban / Urban / Rural |
| Condition | Excellent / Good / Fair / Poor |
| Garage | Yes / No |
| Price | Target variable |

---

# Feature Engineering

## HouseAge Conversion

Instead of using:

```python
YearBuilt
```

we convert it into:

```python
HouseAge = CurrentYear - YearBuilt
```

This improves model understanding.

---

# One-Hot Encoding

Categorical variables are converted into numerical format.

## Location Encoding

Reference category:
- Rural

| Downtown | Suburban | Urban | Actual Location |
|---|---|---|---|
|1|0|0|Downtown|
|0|1|0|Suburban|
|0|0|1|Urban|
|0|0|0|Rural|

---

## Condition Encoding

Reference category:
- Poor

| Excellent | Fair | Good | Actual Condition |
|---|---|---|---|
|1|0|0|Excellent|
|0|1|0|Fair|
|0|0|1|Good|
|0|0|0|Poor|

---

# Final Feature Matrix

| Variable | Feature |
|---|---|
| x1 | Area |
| x2 | Bedrooms |
| x3 | Bathrooms |
| x4 | Floors |
| x5 | HouseAge |
| x6 | Location_Downtown |
| x7 | Location_Suburban |
| x8 | Location_Urban |
| x9 | Condition_Excellent |
| x10 | Condition_Fair |
| x11 | Condition_Good |
| x12 | Garage |

Target:
- y = Price

---

# Mathematical Representation

## Prediction Formula

\[
\hat{y} = b + w_1x_1 + w_2x_2 + \dots + w_{12}x_{12}
\]

Where:

- \( \hat{y} \) = predicted price
- \( b \) = bias
- \( w \) = learned weights

---

# Matrix Representation

## Feature Matrix

\[
X =
\begin{bmatrix}
1 & x_1 & x_2 & \dots & x_{12}
\end{bmatrix}
\]

---

## Weight Matrix

\[
W =
\begin{bmatrix}
b \\
w_1 \\
w_2 \\
\vdots \\
w_{12}
\end{bmatrix}
\]

---

# Normal Equation

The project implements Linear Regression using:

\[
W = (X^TX)^{-1}X^Ty
\]

and stable pseudo inverse:

\[
W = X^{+}y
\]

---

# Gradient Descent

The project also implements Gradient Descent from scratch.

## Cost Function

\[
J(W)=\frac{1}{m}\sum_{i=1}^{m}(\hat{y}_i-y_i)^2
\]

---

## Gradient Formula

\[
\frac{\partial J}{\partial W}
=
\frac{2}{m}X^T(\hat{y}-y)
\]

---

## Weight Update Rule

\[
W := W - \alpha \nabla J(W)
\]

Where:
- \( \alpha \) = learning rate

---

# Technologies Used

- Python
- NumPy
- Pandas

---

# Project Structure

```bash
House-Price-Prediction/
│
├── Dataset/
│   ├── House Price Prediction Dataset.csv
│   └── encoded_house_data.csv
│
├── linear_regression_house_price_prediction_solve_using_normal_methametical_equation.ipynb
├── liner_regression_house_price_prediction_gradient_desent.ipynb
│
└── README.md
```

---

# How to Run

## 1. Clone Repository

```bash
git clone repo link :
```

---

## 2. Install Dependencies

```bash
pip install pandas numpy
```

---

# Example Prediction

```python
Area        = 2200
Bedrooms    = 4
Bathrooms   = 3
Floors      = 2
HouseAge    = 10

Downtown    = 1
Suburban    = 0
Urban       = 0

Excellent   = 1
Fair        = 0
Good        = 0

Garage      = 1
```

Predicted house price is generated using learned weights.

---

# Learning Outcomes

This project helps understand:

- Linear Regression mathematics
- Matrix operations
- Feature engineering
- One-Hot Encoding
- Gradient Descent
- Cost functions
- Optimization
- Model training
- Machine Learning fundamentals

---

# Future Improvements

Possible future upgrades:

- Add Train/Test Split
- Add Evaluation Metrics
- Add R² Score
- Add Regularization
- Add Visualization
- Build Flask/Django Web App
- Deploy Model

---

# Author

Created for learning Machine Learning fundamentals and understanding Linear Regression from scratch using pure mathematics and Python.
