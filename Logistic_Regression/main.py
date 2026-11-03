import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.preprocessing import StandardScaler

from logistic_regression import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from logistic_regression import LogisticRegression


# =====================================
# 1. Load dataset
# =====================================

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", dataset.feature_names)

print("\nClasses:", dataset.target_names)

# =====================================
# 2. Split dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =====================================
# 3. Scale data
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 4. Train Custom Logistic Regression
# =====================================

custom_model = LogisticRegression(
    learning_rate=0.01,
    n_iters=10000
)

custom_model.fit(X_train, y_train)

custom_predictions = custom_model.predict(X_test)

# Get probability predictions
custom_probabilities = custom_model.predict_probability(X_test)

# =====================================
# 5. Train Scikit-learn Logistic Regression
# =====================================

sklearn_model = SkLogisticRegression(
    max_iter=10000,
    solver="lbfgs"
)

sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)

sklearn_probabilities = sklearn_model.predict_proba(X_test)[:, 1]

# =====================================
# 6. Evaluation function
# =====================================

def evaluate_model(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    auc = roc_auc_score(y_true, y_prob)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    }


# =====================================
# 7. Compare performance
# =====================================

custom_results = evaluate_model(
    y_test,
    custom_predictions,
    custom_probabilities
)

sklearn_results = evaluate_model(
    y_test,
    sklearn_predictions,
    sklearn_probabilities
)

results = pd.DataFrame(
    [custom_results, sklearn_results],
    index=[
        "Custom Logistic Regression",
        "Scikit-learn"
    ]
)

print("\n===== MODEL COMPARISON =====")
print(results)


# =====================================
# 8. Compare model parameters
# =====================================

print("\n===== MODEL PARAMETERS =====")

print("Custom weights:")
print(custom_model.weights)

print("\nCustom bias:")
print(custom_model.bias)

print("\nScikit-learn coefficients:")
print(sklearn_model.coef_)

print("\nScikit-learn intercept:")
print(sklearn_model.intercept_)


# =====================================
# 9. Classification reports
# =====================================

print("\n===== CUSTOM CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        custom_predictions,
        target_names=dataset.target_names
    )
)

print("\n===== SKLEARN CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        sklearn_predictions,
        target_names=dataset.target_names
    )
)


# =====================================
# 10. Compare predictions
# =====================================

comparison = pd.DataFrame({
    "Actual": y_test,
    "Custom Prediction": custom_predictions,
    "Sklearn Prediction": sklearn_predictions,
    "Custom Probability": custom_probabilities,
    "Sklearn Probability": sklearn_probabilities
})

print("\n===== PREDICTION COMPARISON =====")
print(comparison.head(10))


# =====================================
# 11. Plot confusion matrices
# =====================================

custom_cm = confusion_matrix(
    y_test,
    custom_predictions
)

sklearn_cm = confusion_matrix(
    y_test,
    sklearn_predictions
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(
    axes,
    [custom_cm, sklearn_cm],
    ["Custom Logistic Regression", "Scikit-learn"]
):

    ax.imshow(cm, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(dataset.target_names)
    ax.set_yticklabels(dataset.target_names)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12
            )

plt.tight_layout()
plt.show()


# =====================================
# 12. Plot ROC Curve
# =====================================

custom_fpr, custom_tpr, _ = roc_curve(
    y_test,
    custom_probabilities
)

sklearn_fpr, sklearn_tpr, _ = roc_curve(
    y_test,
    sklearn_probabilities
)

custom_auc = roc_auc_score(
    y_test,
    custom_probabilities
)

sklearn_auc = roc_auc_score(
    y_test,
    sklearn_probabilities
)

plt.figure(figsize=(10, 6))

plt.plot(
    custom_fpr,
    custom_tpr,
    label=f"Custom Logistic Regression (AUC = {custom_auc:.4f})"
)

plt.plot(
    sklearn_fpr,
    sklearn_tpr,
    label=f"Scikit-learn (AUC = {sklearn_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "r--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(True)
plt.show()


# =====================================
# 13. Plot Training Loss Curve
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(
    custom_model.loss_history,
    label="Custom Logistic Regression"
)

plt.xlabel("Epoch")
plt.ylabel("Binary Cross Entropy Loss")

plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)
plt.show()