# INFO-B211-Assignment-5-SciKit-Learn-Regression
# Diabetes Progression Regression Model Comparison  
*A Machine Learning Evaluation Project*
---

# Best Performing Model: Lasso Regression

The strongest overall model in this project is the **Lasso Regression** model, which outperformed all Random Forest, Ridge, OLS, and KNN variants across nearly every evaluation metric. It achieved the **highest R² score (0.5304)** and the **highest explained variance (0.5308)** of all models tested. Lasso also produced the **lowest MAE (40.18)**, **lowest MSE (2611.97)**, and **lowest RMSE (51.11)**, indicating that its predictions were consistently closer to the true disease‑progression values than any other model.  

This performance advantage is largely due to Lasso’s **L1 regularization**, which reduces overfitting and handles multicollinearity—both important characteristics of the diabetes dataset. The model’s ability to shrink irrelevant coefficients to zero results in a simpler, more stable, and more generalizable predictor. Overall, Lasso Regression provides the most accurate and reliable performance profile among all models evaluated.

---

## Model Performance Summary

| Model | Variant | R² Score | Explained Variance | MAE | RMSE | MAPE |
|-------|---------|----------|--------------------|------|--------|--------|
| **LinearRegression** | **Lasso** | **0.5304** | **0.5308** | **40.18** | **51.11** | **0.345** |
| LinearRegression | OLS | 0.5234 | 0.5239 | 40.66 | 51.48 | 0.350 |
| LinearRegression | Ridge | 0.4474 | 0.4478 | 44.76 | 55.43 | 0.419 |
| RandomForest | Max Features = sqrt | 0.4748 | 0.4767 | 43.74 | 54.04 | 0.397 |
| RandomForest | More Trees (400) | 0.4427 | 0.4439 | 45.21 | 55.67 | 0.407 |
| RandomForest | Default | 0.4350 | 0.4366 | 45.55 | 56.06 | 0.412 |
| KNN | k=10 | 0.4568 | 0.4609 | 43.09 | 54.96 | 0.363 |
| KNN | Distance Weight | 0.3927 | 0.3995 | 44.34 | 58.12 | 0.374 |
| KNN | k=5 | 0.3885 | 0.3944 | 44.53 | 58.31 | 0.378 |

**Why Lasso Wins:**  
- Highest R² and explained variance  
- Lowest MAE, MSE, RMSE, and MAPE  
- Handles multicollinearity effectively  
- Reduces overfitting through L1 regularization  
- Produces a simpler, more stable model  
- Well‑suited for small, linear datasets like diabetes progression  

---

## Project Overview  
This project builds, trains, and evaluates three machine learning regression model families using the **Diabetes** dataset from `sklearn`. The goal is to compare several model types—**Random Forest**, **Linear Regression (OLS, Ridge, Lasso)**, and **K‑Nearest Neighbors (KNN)**—across different parameter variations to determine which performs best on this dataset.

All models are evaluated using consistent metrics: **R²**, **Explained Variance**, **MAE**, **MSE**, **RMSE**, and **MAPE**. Results are saved to a CSV file (`model_comparison_full.csv`) for external review.

---

## Project Purpose  
The purpose of this project is to:

- Explore how different regression algorithms perform on the same dataset  
- Compare model variants to understand how hyperparameters influence performance  
- Build a reproducible evaluation pipeline using modular functions  
- Produce a clean, interpretable results table for analysis or reporting  

This project demonstrates practical model experimentation, structured evaluation, and clear documentation—skills essential for applied machine learning.

---

# Program Structure and Design

The program is organized into four major components:

1. **Evaluation Function**  
2. **Random Forest Model Variants**  
3. **Linear Regression Model Variants**  
4. **KNN Model Variants**  
5. **Main Execution Block**

Each model family has its own set of functions, and each function returns a dictionary of evaluation metrics. This modular design ensures clarity, reusability, and easy expansion.

---

# Function Documentation

Below is a full explanation of each function and its attributes.

---

## Evaluation Function

### `evaluate_model(model, X_test, y_test)`
**Purpose:**  
Generate predictions using a trained model and compute regression metrics.

**Inputs:**  
- `model` — a trained ML model  
- `X_test` — test feature data  
- `y_test` — true target values  

**Behavior:**  
- Calls `model.predict()`  
- Computes R², explained variance, MAE, MSE, RMSE, MAPE  
- Normalizes RMSE relative to mean and range  

**Output:**  
A dictionary containing all evaluation metrics.

**Limitations:**  
- Assumes numeric regression targets  
- No confidence intervals or uncertainty estimates  

---

# Random Forest Variants

### `rf_default(...)`
Random Forest with default hyperparameters (`random_state=50`).

### `rf_more_trees(...)`
Random Forest with **400 trees**, improving stability at the cost of computation time.

### `rf_max_features(...)`
Random Forest using **max_features='sqrt'** to reduce correlation between trees.

**Shared Inputs:**  
- `X_train`, `X_test`, `y_train`, `y_test`  

**Shared Behavior:**  
- Fit model on training data  
- Return evaluation metrics via `evaluate_model()`  

**Limitations:**  
- No tuning of depth, min samples, or other hyperparameters  
- May underperform on small datasets  

---

# Linear Regression Variants

### `lr_default(...)`
Ordinary Least Squares (OLS) regression.

### `lr_ridge(...)`
Ridge regression with **L2 regularization** to reduce coefficient magnitude.

### `lr_lasso(...)`
Lasso regression with **L1 regularization**, encouraging sparsity and reducing overfitting.

**Shared Inputs:**  
- `X_train`, `X_test`, `y_train`, `y_test`  

**Shared Behavior:**  
- Fit model  
- Evaluate using `evaluate_model()`  

**Limitations:**  
- Linear models cannot capture nonlinear relationships  
- Sensitive to outliers  

---

# K‑Nearest Neighbors Variants

### `knn_5(...)`
KNN with 5 neighbors.

### `knn_10(...)`
KNN with 10 neighbors for smoother predictions.

### `knn_distance(...)`
KNN with distance weighting—closer neighbors have more influence.

**Shared Inputs:**  
- `n_neighbors`  
- Optional `weights="distance"`

**Shared Behavior:**  
- Store training samples  
- Predict based on nearest neighbors  
- Evaluate via `evaluate_model()`  

**Limitations:**  
- Sensitive to feature scaling  
- Performs poorly on small datasets  
- Slow at prediction time  

---

# Main Program Execution

The main execution block:

- Loads the Diabetes dataset  
- Splits data into training/testing sets  
- Runs all model variants  
- Appends results to a list  
- Converts results into a Pandas DataFrame  
- Saves results to `model_comparison_full.csv`  
- Prints the results table to the terminal  

This structure ensures the script can be imported without running the full pipeline.

---

# Project Limitations

- **No feature scaling**, which affects KNN performance  
- **Minimal hyperparameter tuning**  
- **No cross‑validation**, so results depend on a single split  
- **Dataset‑specific performance** — results may not generalize  

---

# Output File

### `model_comparison_full.csv`
Contains:

- Model family  
- Variant name  
- R²  
- Explained Variance  
- MAE  
- MSE  
- RMSE  
- MAPE  

This file allows easy comparison and visualization outside the script.

---

# Summary

This project provides a clean, modular framework for comparing multiple machine learning regression models on the same dataset. Each function is designed for clarity and reusability, and the evaluation pipeline ensures consistent, interpretable results. The README documents the purpose, structure, and limitations of the project for future users or collaborators.

