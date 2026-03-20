# ===============================
# 1. Import Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ===============================
# 2. Load Dataset (Objective 1)
# ===============================
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nStatistical Summary:\n", df.describe())


# ===============================
# 3. Data Preprocessing (Objective 2)
# ===============================
X = df.iloc[:, :-1]   # features
y = df['species']     # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 4. Exploratory Data Analysis (EDA)
# ===============================

# Feature-wise plots
df.hist(figsize=(8,6))
plt.suptitle("Feature Distribution")
plt.show()

# Scatter plot (Sepal Length vs Width)
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['species'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Scatter Plot")
plt.show()


# ===============================
# 5. Models Implementation (Objective 3)
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}


# ===============================
# 6. Training + Evaluation (Objective 4 & 5)
# ===============================



for name, model in models.items():
    print("\n==============================")
    print("Model:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    # Confusion Matrix Display (Visual)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=iris.target_names
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.show()