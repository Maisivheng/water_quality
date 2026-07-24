# =====================================
# WATER QUALITY PREDICTION
# Decision Tree Classification
# =====================================

# ---------- Import Libraries ----------
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================
# Create Images Folder
# =====================================
os.makedirs("images", exist_ok=True)

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("data/water_potability.csv")

print("=" * 50)
print("First 5 Rows")
print(df.head())

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("=" * 60)
print("Missing Values")
print(df.isnull().sum())

# =====================================
# Handle Missing Values
# =====================================
imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

print("=" * 60)
print("Missing Values After Cleaning")
print(df.isnull().sum())

# =====================================
# Exploratory Data Analysis (EDA)
# =====================================
print(df.describe())

# Target Distribution
plt.figure(figsize=(5,4))
sns.countplot(x="Potability", data=df)
plt.title("Water Potability Distribution")
plt.tight_layout()
plt.savefig("images/potability_distribution.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()

# =====================================
# Split Features and Target
# =====================================
X = df.drop("Potability", axis=1)
y = df["Potability"]

# =====================================
# Train/Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# Decision Tree Model
# =====================================
tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)

# Prediction
y_pred = tree.predict(X_test)

# =====================================
# Evaluation
# =====================================
print("\n========== Decision Tree Results ==========")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# =====================================
# Confusion Matrix
# =====================================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("images/confusion_matrix_tree.png")
plt.show()

# =====================================
# Hyperparameter Tuning
# =====================================
params = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n========== Best Model ==========")
print("Best Parameters :", grid.best_params_)
print("Best CV Score   :", grid.best_score_)

# =====================================
# Best Model Evaluation
# =====================================
best_tree = grid.best_estimator_

y_best = best_tree.predict(X_test)

print("\n========== Best Decision Tree ==========")
print("Accuracy :", accuracy_score(y_test, y_best))
print("Precision:", precision_score(y_test, y_best))
print("Recall   :", recall_score(y_test, y_best))
print("F1 Score :", f1_score(y_test, y_best))

# =====================================
# Feature Importance
# =====================================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_tree.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.show()

# =====================================
# Decision Tree Visualization
# =====================================
plt.figure(figsize=(18,10))
plot_tree(
    best_tree,
    feature_names=X.columns,
    class_names=["Not Potable", "Potable"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.tight_layout()
plt.savefig("images/decision_tree.png")
plt.show()

# print("\nProject Completed Successfully!")