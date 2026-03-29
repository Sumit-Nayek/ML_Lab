import pandas as pd

# Load dataset
url = "https://www.nitttrkol.ac.in/kinsuk/lab/ML_Lab7_data.xlsx"
df = pd.read_excel(url)

print(df.head())
print(df.info())