import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    explained_variance_score as EVS,
    r2_score,
    median_absolute_error as MDAE,   # median absolute error
    mean_absolute_error as MAE,     # standard mean absolute error
    mean_squared_error as MSE,      # used for RMSE as well
    mean_absolute_percentage_error as MAPE,)
import numpy as np  # needed for square root

# -------------------------------------------------
# Model Evaluation Function
# -------------------------------------------------
def evaluate_model(model, x_test, y_test):
    """
    Evaluate a trained regression model using a comprehensive set of metrics.
    This function centralizes evaluation so all models are compared consistently.
    """

    # Generate predictions for the test set
    predictions = model.predict(x_test)

    # Core error metrics (lower = better)
    mse = MSE(y_test, predictions)
    rmse = np.sqrt(mse)  # RMSE penalizes large errors more heavily than MAE

    # Compute normalization constants so we can express error relative to scale
    y_mean = np.mean(y_test)  # helps interpret RMSE relative to typical values
    y_range = y_test.max() - y_test.min()  # helps interpret error relative to spread

    # Return a dictionary so results can be unpacked directly into a DataFrame
    return {
        "Explained Variance Score": EVS(y_test, predictions),  # how much variance is captured
        "R2 Score": r2_score(y_test, predictions),             # proportion of variance explained
        "Median Absolute Error": MDAE(y_test, predictions),    # robust to outliers
        "Mean Absolute Error": MAE(y_test, predictions),       # average absolute deviation
        "Mean Squared Error": mse,
        "Root Mean Squared Error": rmse,
        "RMSE / Mean(y)": rmse / y_mean if y_mean != 0 else np.nan,   # scale-aware error
        "RMSE / Range(y)": rmse / y_range if y_range != 0 else np.nan,
        "Mean Absolute Percentage Error": MAPE(y_test, predictions),   # relative percentage error
    }

# -------------------------------------------------
# Random Forest Variants
# -------------------------------------------------

def rf_default(x_train, y_train, x_test, y_test):
    """
    Baseline Random Forest model.
    Random Forests reduce overfitting by averaging many decision trees.
    """
    model = RandomForestRegressor(random_state=50)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def rf_more_trees(x_train, y_train, x_test, y_test):
    """
    Increase number of trees to 400.
    More trees generally reduce variance and stabilize predictions,
    but increase training time.
    """
    model = RandomForestRegressor(n_estimators=400, random_state=50)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def rf_max_features(x_train, y_train, x_test, y_test):
    """
    Use sqrt(max_features) to reduce correlation between trees.
    This improves generalization by encouraging more diverse splits.
    """
    model = RandomForestRegressor(max_features='sqrt', random_state=50)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

# -------------------------------------------------
# Linear Regression Variants
# -------------------------------------------------

def lr_default(x_train, y_train, x_test, y_test):
    """
    Ordinary Least Squares (OLS) regression.
    Serves as a simple baseline model.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def lr_ridge(x_train, y_train, x_test, y_test):
    """
    Ridge regression adds L2 regularization.
    Helps when features are correlated by shrinking coefficients.
    """
    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def lr_lasso(x_train, y_train, x_test, y_test):
    """
    Lasso regression adds L1 regularization.
    Encourages sparsity (drives some coefficients to zero).
    """
    model = Lasso(alpha=0.01)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

# -------------------------------------------------
# KNN Regression Variants
# -------------------------------------------------

def knn_5(x_train, y_train, x_test, y_test):
    """
    KNN with k=5 neighbors.
    KNN is a non-parametric model that predicts based on local similarity.
    """
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def knn_10(x_train, y_train, x_test, y_test):
    """
    KNN with k=10 neighbors.
    Increasing k smooths predictions and reduces noise,
    but may underfit if k is too large.
    """
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

def knn_distance(x_train, y_train, x_test, y_test):
    """
    Distance-weighted KNN.
    Closer neighbors have more influence, which can improve performance
    when local structure matters.
    """
    model = KNeighborsRegressor(weights="distance")
    model.fit(x_train, y_train)
    return evaluate_model(model, x_test, y_test)

# -------------------------------------------------
# Main Execution
# -------------------------------------------------

# Load the diabetes dataset (10 features, 442 samples)
d_data = datasets.load_diabetes()
x_data = d_data.data
y_data = d_data.target

print("Feature count:", x_data.shape[1])
print("Sample count:", x_data.shape[0])

# Standard 80/20 split for fair evaluation
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=50)

results = []

# Run all nine model variants and store results for comparison
results.append({"Model": "RandomForest", "Variant": "Default", **rf_default(x_train, y_train, x_test, y_test)})
results.append({"Model": "RandomForest", "Variant": "More Trees (400)", **rf_more_trees(x_train, y_train, x_test, y_test)})
results.append({"Model": "RandomForest", "Variant": "Max Features = sqrt", **rf_max_features(x_train, y_train, x_test, y_test)})

results.append({"Model": "LinearRegression", "Variant": "OLS", **lr_default(x_train, y_train, x_test, y_test)})
results.append({"Model": "LinearRegression", "Variant": "Ridge", **lr_ridge(x_train, y_train, x_test, y_test)})
results.append({"Model": "LinearRegression", "Variant": "Lasso", **lr_lasso(x_train, y_train, x_test, y_test)})

results.append({"Model": "KNN", "Variant": "k=5", **knn_5(x_train, y_train, x_test, y_test)})
results.append({"Model": "KNN", "Variant": "k=10", **knn_10(x_train, y_train, x_test, y_test)})
results.append({"Model": "KNN", "Variant": "Distance Weight", **knn_distance(x_train, y_train, x_test, y_test)})

# Convert results to DataFrame for easy viewing and export
df = pd.DataFrame(results)
df.to_csv("model_comparison_full.csv", index=False)

print("\nResults saved to model_comparison_full.csv")
print(df)

# Identify the best model using R² (higher = better)
best_by_r2 = df.loc[df["R2 Score"].idxmax()]
print("\nBest model according to R2 score:")
print(best_by_r2.to_string())
