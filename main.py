import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("data/water_potability.csv")

print("="*50)
print("First 5 Rows")
print("="*50)
print(df.head())

print("="*50)
print("Dataset Shape")
print(df.shape)

print("="*50)
print(df.info())

# =====================================
# Missing Values
# =====================================

print("="*50)
print("Missing Values")
print(df.isnull().sum())

imputer = SimpleImputer(strategy="median")

df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

print("="*50)
print("Missing Values After Cleaning")
print(df.isnull().sum())

# =====================================
# Exploratory Data Analysis
# =====================================

print(df.describe())

plt.figure(figsize=(5,4))
sns.countplot(x="Potability", data=df)
plt.title("Water Potability Distribution")
plt.savefig("images/potability_distribution.png")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png")
plt.show()

# =====================================
# Split Features & Target
# =====================================

X = df.drop("Potability", axis=1)
y = df["Potability"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =====================================
# Feature Scaling
# =====================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# Logistic Regression
# =====================================

print("="*50)
print("LOGISTIC REGRESSION")

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred_log))

log_accuracy = accuracy_score(y_test, y_pred_log)
log_precision = precision_score(y_test, y_pred_log)
log_recall = recall_score(y_test, y_pred_log)
log_f1 = f1_score(y_test, y_pred_log)

print("Accuracy :", log_accuracy)
print("Precision:", log_precision)
print("Recall   :", log_recall)
print("F1 Score :", log_f1)

cm = confusion_matrix(y_test, y_pred_log)

ConfusionMatrixDisplay(cm).plot()

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("images/confusion_matrix_logistic.png")
plt.show()

# ROC Curve

y_prob = log_model.predict_proba(X_test_scaled)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label="ROC Curve")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("images/roc_curve.png")
plt.show()

print("AUC Score:",roc_auc_score(y_test,y_prob))

# =====================================
# Logistic Regression GridSearchCV
# =====================================

print("="*50)
print("Grid Search Logistic Regression")

param_log = {
    "C":[0.01,0.1,1,10],
    "solver":["liblinear","lbfgs"]
}

grid_log = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_log,
    cv=5,
    scoring="accuracy"
)

grid_log.fit(X_train_scaled,y_train)

print("Best Parameters:",grid_log.best_params_)
print("Best Score:",grid_log.best_score_)

# =====================================
# Decision Tree
# =====================================

print("="*50)
print("DECISION TREE")

tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train,y_train)

y_pred_tree = tree.predict(X_test)

print(classification_report(y_test,y_pred_tree))

tree_accuracy = accuracy_score(y_test,y_pred_tree)
tree_precision = precision_score(y_test,y_pred_tree)
tree_recall = recall_score(y_test,y_pred_tree)
tree_f1 = f1_score(y_test,y_pred_tree)

print("Accuracy :",tree_accuracy)
print("Precision:",tree_precision)
print("Recall   :",tree_recall)
print("F1 Score :",tree_f1)

cm = confusion_matrix(y_test,y_pred_tree)

ConfusionMatrixDisplay(cm).plot()

plt.title("Decision Tree Confusion Matrix")
plt.savefig("images/confusion_matrix_tree.png")
plt.show()

# =====================================
# Decision Tree Grid Search
# =====================================

print("="*50)
print("Grid Search Decision Tree")

param_tree = {
    "criterion":["gini","entropy"],
    "max_depth":[3,5,7,10,None],
    "min_samples_split":[2,5,10]
}

grid_tree = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_tree,
    cv=5,
    scoring="accuracy"
)

grid_tree.fit(X_train,y_train)

print("Best Parameters:",grid_tree.best_params_)
print("Best Score:",grid_tree.best_score_)

# =====================================
# Model Comparison
# =====================================

results = pd.DataFrame({

    "Model":[
        "Logistic Regression",
        "Decision Tree"
    ],

    "Accuracy":[
        log_accuracy,
        tree_accuracy
    ],

    "Precision":[
        log_precision,
        tree_precision
    ],

    "Recall":[
        log_recall,
        tree_recall
    ],

    "F1 Score":[
        log_f1,
        tree_f1
    ]

})

print("="*50)
print("Comparison")
print(results)

results.set_index("Model").plot(kind="bar",figsize=(8,5))

plt.title("Model Comparison")
plt.ylabel("Score")
plt.savefig("images/model_comparison.png")
plt.show()

print("="*50)
print("PROJECT FINISHED")
print("="*50)