# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Load Dataset
# ---------------------------
url = "https://www.nitttrkol.ac.in/kinsuk/ML_lab3_data.xlsx"
data = pd.read_excel(url)

print("Original Dataset:")
print(data.head())

# ---------------------------
# a) Label Encoding
# ---------------------------
le = LabelEncoder()

encoded_data = data.copy()

for col in encoded_data.columns:
    if encoded_data[col].dtype == 'object':
        encoded_data[col] = le.fit_transform(encoded_data[col])

print("\nEncoded Dataset:")
print(encoded_data.head())

# ---------------------------
# Split Features and Target
# (Assuming last column is target Play Game)
# ---------------------------
X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# b) Decision Tree Classifier
# ---------------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# Predict New Weather Condition
# Example input (change values based on dataset encoding)
# ---------------------------
sample = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)

print("\nPrediction for sample:", prediction)