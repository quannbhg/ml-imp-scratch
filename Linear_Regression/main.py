import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from linear_regression import LinearRegression

# =====================================
# 1. Load dataset
# =====================================
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", diabetes.feature_names)

# =====================================
# 2. Split dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================================
# 3. Scale data
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 4. Train custom Linear Regression
# =====================================

custom_model = LinearRegression(
    learning_rate=0.01,
    n_iters=10000,
    method="gradient_descent"
)

custom_model.fit(X_train, y_train)

custom_predictions = custom_model.predict(X_test)

# =====================================
# 5. Train Scikit-learn Linear Regression
# =====================================

sklearn_model = SkLinearRegression()

sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)

# =====================================
# 6. Evaluation function
# =====================================

def evaluate_model(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# =====================================
# 7. Compare performance
# =====================================

custom_results = evaluate_model(
    y_test,
    custom_predictions
)

sklearn_results = evaluate_model(
    y_test,
    sklearn_predictions
)

results = pd.DataFrame(
    [custom_results, sklearn_results],
    index=["Custom Linear Regression", "Scikit-learn"]
)

print("\n===== MODEL COMPARISON =====")
print(results)

# =====================================
# 8. Compare model parameters
# =====================================

print("\n===== MODEL PARAMETERS =====")

print("Custom weights:")
print(custom_model.weights)

print("Custom bias:")
print(custom_model.bias)

print("\nScikit-learn coefficients:")
print(sklearn_model.coef_)

print("Scikit-learn intercept:")
print(sklearn_model.intercept_)

# =====================================
# 9. Compare predictions
# =====================================

comparison = pd.DataFrame({
    "Actual": y_test,
    "Custom Prediction": custom_predictions,
    "Sklearn Prediction": sklearn_predictions
})

print("\n===== PREDICTION COMPARISON =====")
print(comparison.head(10))

# =====================================
# 10. Plot actual vs predictions
# =====================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    custom_predictions,
    label="Custom Linear Regression",
    alpha=0.7
)

plt.scatter(
    y_test,
    sklearn_predictions,
    label="Scikit-learn",
    alpha=0.7
)

min_value = min(y_test.min(), custom_predictions.min(),
                sklearn_predictions.min())

max_value = max(y_test.max(), custom_predictions.max(),
                sklearn_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# =====================================
# 11. Plot loss curve
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(
    custom_model.loss_history,
    label="Custom Gradient Descent"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()