# Chunk 1: Import necessary libraries
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
import warnings
warnings.filterwarnings('ignore')

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

# Chunk 2: Data Pre-processing

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

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
