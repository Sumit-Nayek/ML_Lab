# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------

# If loading directly from URL
url = "https://www.nitttrkol.ac.in/kinsuk/sample_day2.txt"

# Try reading with automatic separator detection
data = pd.read_csv(url, sep=None, engine='python', header=None)

# Rename columns (Assuming 2 columns: X and Y)
data.columns = ['X', 'Y']

print("\nDataset Preview:")
print(data.head())

# -------------------------------
# Part (a): Least Squares Method
# -------------------------------

X = data['X'].values
Y = data['Y'].values

# Calculate slope and intercept manually (Least Squares Formula)
n = len(X)

slope = (n*np.sum(X*Y) - np.sum(X)*np.sum(Y)) / (n*np.sum(X**2) - (np.sum(X))**2)
intercept = (np.sum(Y) - slope*np.sum(X)) / n

print("\nLeast Squares Regression Line:")
print(f"Y = {slope:.4f}X + {intercept:.4f}")

# -------------------------------
# Part (b): Plot Data + Regression Line
# -------------------------------

plt.figure()
plt.scatter(X, Y)
plt.plot(X, slope*X + intercept)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Data Points and Regression Line")
plt.show()

# -------------------------------
# Part (c): Train-Test Split + Regression Model
# -------------------------------

# Reshape for sklearn
X_reshaped = X.reshape(-1, 1)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X_reshaped, Y, test_size=0.2, random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict Test Data
Y_pred = model.predict(X_test)

# Predict Y for X = 2026
future_value = model.predict([[2026]])
print("\nPredicted Y for X = 2026:", future_value[0])

# -------------------------------
# Part (d): RMSE and R2 Score
# -------------------------------

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)
