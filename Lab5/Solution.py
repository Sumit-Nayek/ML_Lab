import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load raw data (no column names)
df = pd.read_excel(r'C:\Users\Student\Downloads\ML_Lab5_data.xlsx', header=None)

# First column is target, rest are features
X = df.iloc[:, 1:].values  # All columns except first
y = df.iloc[:, 0].values    # First column (target: 1,2,3)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN with different metrics (k=5)
metrics = {'Euclidean': 'euclidean', 'Manhattan': 'manhattan', 'Minkowski': 'minkowski'}
results = []

print("="*60)
print("KNN Classification Results (k=5)")
print("="*60)
print(f"{'Metric':<12} {'Accuracy':<10} {'F1-Score':<10}")
print("-"*60)

for name, metric in metrics.items():
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"{name:<12} {acc:<10.4f} {f1:<10.4f}")
    results.append([name, acc, f1])

print("="*60)

print("\nFeature scaling applied (StandardScaler - mean=0, std=1).")

# --- Step 5: Implement KNN with different distance metrics ---
metrics = {
    'Euclidean': 'euclidean',
    'Manhattan': 'manhattan',
    'Minkowski': 'minkowski'  # Note: for p=2, Minkowski is same as Euclidean. Default p=2 in scikit-learn.
}

results = []

# Let's find a good k value (number of neighbors). You can experiment.
# A simple starting point is k = int(sqrt(n_samples)) or just try k=5.
k = 5
print(f"\n--- Evaluating KNN (k={k}) with different distance metrics ---")

for name, metric in metrics.items():
    # Create KNN classifier with the specified metric
    # For Minkowski, p=2 gives Euclidean, p=1 gives Manhattan. We'll use default p=2.
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

    # Train the model on the SCALED training data
    knn.fit(X_train_scaled, y_train)

    # Predict on the SCALED test data
    y_pred = knn.predict(X_test_scaled)

    # Calculate Accuracy and F1-Score
    # For multi-class, F1-score needs an averaging method. 'weighted' is common.
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use 'macro' if classes are balanced

    results.append({
        'Distance Metric': name,
        'Accuracy': acc,
        'F1-Score (weighted)': f1
    })

    # Optional: Print detailed report
    # print(f"\nClassification Report for {name}:")
    # print(classification_report(y_test, y_pred))

# --- Step 6: Compare results in a table ---
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("Comparison Table for KNN with Different Distance Metrics")
print("="*60)
# Format the float columns for better readability
pd.options.display.float_format = '{:.4f}'.format
print(results_df.to_string(index=False))
print("="*60)

# Optional: If you want to experiment with different k values
print("\n--- Experimenting with different k values (optional) ---")
for k_val in [3, 5, 7, 9]:
    print(f"\nResults for k={k_val}:")
    for name, metric in metrics.items():
        knn = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"  {name:10s}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
