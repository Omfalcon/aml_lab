import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================================
# 1. LOAD DATASET
# ==========================================
df0 = pd.read_csv("0.csv")
df1 = pd.read_csv("1.csv")
df2 = pd.read_csv("2.csv")
df3 = pd.read_csv("3.csv")

# Add labels
df0['label'] = 0
df1['label'] = 1
df2['label'] = 2
df3['label'] = 3

# Combine dataset
df = pd.concat([df0, df1, df2, df3], ignore_index=True)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nFirst Rows:\n", df.head())

# ==========================================
# VISUALIZATION (Point 1)
# ==========================================
plt.figure()
sns.countplot(x='label', data=df)
plt.title("Gesture Class Distribution")
plt.show()

# ==========================================
# 2. PREPROCESSING + EDA
# ==========================================

# Drop unwanted columns
for col in ['time', 'timestamp']:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Handle missing values
df = df.fillna(df.mean())

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Sample EMG signal
plt.figure()
plt.plot(df.iloc[0][:-1])
plt.title("Sample EMG Signal")
plt.xlabel("Sensor Channels")
plt.ylabel("Signal Value")
plt.show()

# ==========================================
# FEATURE ENGINEERING (Important for EMG)
# ==========================================
def extract_features(data):
    features = pd.DataFrame()
    features['mean'] = data.mean(axis=1)
    features['std'] = data.std(axis=1)
    features['max'] = data.max(axis=1)
    features['min'] = data.min(axis=1)
    features['range'] = features['max'] - features['min']
    features['rms'] = np.sqrt((data**2).mean(axis=1))
    return features

X_raw = df.drop('label', axis=1)
X = extract_features(X_raw)
y = df['label']

# ==========================================
# SPLIT + SCALING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 3. FIVE MODELS
# ==========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(kernel='rbf')
}

labels = ["Rock", "Scissors", "Paper", "OK"]
gesture_map = {0: "Rock", 1: "Scissors", 2: "Paper", 3: "OK"}

results = {}


for name, model in models.items():
    print("\n============================")
    print(f"Model: {name}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Show sample predictions
    print("\nSample Predictions:")
    for i in range(3):
        print("Actual:", gesture_map[y_test.iloc[i]],
              "| Predicted:", gesture_map[y_pred[i]])

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ==========================================
# MODEL COMPARISON
# ==========================================
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()

# ==========================================
# BEST MODEL
# ==========================================
best_model = max(results, key=results.get)

print("\n🔥 BEST MODEL:", best_model)
print("Accuracy:", results[best_model])