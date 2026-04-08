# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             classification_report)

# Load the data (assuming file is in working directory)
df = pd.read_excel('ML_Lab7_data.xlsx')

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nTarget distribution:")
print(df['target'].value_counts())

# Data Pre-processing

# Since there are no missing values (as seen from output), we proceed

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget distribution:\n{y.value_counts()}")

# Normalize/Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier viewing (optional)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nAfter standardization:")
print(f"Mean of each feature: {X_scaled_df.mean().values[:5]}...")
print(f"Std of each feature: {X_scaled_df.std().values[:5]}...")

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training target distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test target distribution:\n{pd.Series(y_test).value_counts()}")

# Train SVM with multiple kernels

# Define kernels to test
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Dictionary to store models and their predictions
models = {}
predictions = {}
probabilities = {}

print("Training SVM models with different kernels...\n")

for kernel in kernels:
    print(f"\nTraining with {kernel} kernel...")
    
    # Create SVM model
    if kernel == 'poly':
        svm_model = SVC(kernel=kernel, degree=3, probability=True, random_state=42)
    else:
        svm_model = SVC(kernel=kernel, probability=True, random_state=42)
    
    # Train the model
    svm_model.fit(X_train, y_train)
    
    # Store model
    models[kernel] = svm_model
    
    # Make predictions
    y_pred = svm_model.predict(X_test)
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    
    predictions[kernel] = y_pred
    probabilities[kernel] = y_prob
    
    print(f"  {kernel} kernel training completed")

print("All models trained successfully!")

# Compare performance metrics

# Create results dataframe
results = []
for kernel in kernels:
    y_pred = predictions[kernel]
    y_prob = probabilities[kernel]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results.append({
        'Kernel': kernel,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': roc_auc
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.round(4)




print(results_df.to_string(index=False))

# Find best kernel for each metric
print("\n BEST PERFORMING KERNEL FOR EACH METRIC:")

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
    best_kernel = results_df.loc[results_df[metric].idxmax(), 'Kernel']
    best_score = results_df[metric].max()
    print(f"{metric}: {best_kernel} kernel ({best_score:.4f})")


# Display confusion matrices



fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    cm = confusion_matrix(y_test, predictions[kernel])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Benign (0)', 'Malignant (1)'],
                yticklabels=['Benign (0)', 'Malignant (1)'])
    axes[idx].set_title(f'{kernel.capitalize()} Kernel - Confusion Matrix')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Detailed classification report for best kernel (RBF usually performs best)
best_kernel = results_df.loc[results_df['Accuracy'].idxmax(), 'Kernel']
print(f"\nDetailed Classification Report for {best_kernel.upper()} Kernel:")
print(classification_report(y_test, predictions[best_kernel], 
                           target_names=['Benign (0)', 'Malignant (1)']))

# Plot ROC curves (FPR vs TPR)

plt.figure(figsize=(10, 8))

colors = ['blue', 'green', 'red', 'orange']
linestyles = ['-', '--', '-.', ':']
for idx, kernel in enumerate(kernels):
    fpr, tpr, _ = roc_curve(y_test, probabilities[kernel])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=colors[idx], linestyle=linestyles[idx],
             lw=2, label=f'{kernel.capitalize()} (AUC = {roc_auc:.3f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curves for Different SVM Kernels', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

