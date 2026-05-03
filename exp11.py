# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

# ===============================
# 2. LOAD DATA
# ===============================
train_df = pd.read_csv("archive/train_features.csv")
val_df = pd.read_csv("archive/val_features.csv")
test_df = pd.read_csv("archive/test_features.csv")

print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

# ===============================
# 3. PREPROCESSING
# ===============================
target_col = "image"

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_val = val_df.drop(target_col, axis=1)
y_val = val_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# Handle missing values
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# ===============================
# 4. EDA (on train only)
# ===============================
plt.figure()
sns.countplot(x=y_train)
plt.title("Train Data Class Distribution")
plt.show()

# ===============================
# 5. FEATURE GROUP ANALYSIS
# ===============================
features = X_train.columns

intensity = [c for c in features if "intensity" in c.lower()]
glcm = [c for c in features if "glcm" in c.lower()]
fft = [c for c in features if "fft" in c.lower()]
edge = [c for c in features if "edge" in c.lower()]
lbp = [c for c in features if "lbp" in c.lower()]

groups = {
    "Intensity": intensity,
    "GLCM": glcm,
    "FFT": fft,
    "Edge": edge,
    "LBP": lbp
}

print("\nFeature Groups:")
for k, v in groups.items():
    print(k, ":", len(v))

# ===============================
# 6. PIPELINES
# ===============================
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=20)),
    ('pca', PCA(n_components=10)),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])

models = {
    "Random Forest": pipeline_rf,
    "Logistic Regression": pipeline_lr,
    "SVM": pipeline_svm
}

# ===============================
# 7. TRAIN + VALIDATE
# ===============================
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🔹 Training {name}")

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    val_acc = accuracy_score(y_val, val_pred)
    print("Validation Accuracy:", val_acc)

    if val_acc > best_score:
        best_score = val_acc
        best_model = model
        best_name = name

print(f"\n✅ Best Model: {best_name}")

# ===============================
# 8. FINAL TEST EVALUATION
# ===============================
y_pred = best_model.predict(X_test)

print("\n===== FINAL TEST RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Final Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()