# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 2. Load Dataset
# =========================
train = pd.read_csv("train.csv", low_memory=False)
test = pd.read_csv("test.csv", low_memory=False)

df = pd.concat([train, test], axis=0)

# =========================
# 3. Drop unwanted columns
# =========================
drop_cols = ['ID', 'Customer_ID', 'Name', 'SSN']

for col in drop_cols:
    if col in df.columns:
        df = df.drop(col, axis=1)

# =========================
# 4. Handle Missing Values
# =========================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# =========================
# 5. Convert categorical → string
# =========================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str)

# =========================
# 6. Encoding
# =========================
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# =========================
# 7. Split
# =========================
train_df = df[:len(train)]
target = "Credit_Score"

X = train_df.drop(target, axis=1)
y = train_df[target]

# =========================
# 8. Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 9. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# =========================
# 10. Models (Improved)
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),

    "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced'),

    "Random Forest": RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "SVM": SVC(kernel='rbf', C=2, class_weight='balanced')
}

# =========================
# 11. Train + Evaluate
# =========================
results = {}

for name, model in models.items():
    print("\n=========================")
    print(name)
    print("=========================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(name)
    plt.show()

# =========================
# 12. Best Model
# =========================
best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Accuracy:", results[best_model])