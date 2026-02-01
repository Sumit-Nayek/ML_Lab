import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

st.title(" Dataset Statistics Analyzer")

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # -----------------------------
    # Missing Values
    # -----------------------------
    total_missing = df.isnull().sum().sum()
    st.subheader("Missing Values")
    st.write(f"Total Missing Values: **{total_missing}**")

    # -----------------------------
    # Numeric Columns Only
    # -----------------------------
    num_df = df.select_dtypes(include=[np.number])
    num_df_no_missing = num_df.dropna()

    # -----------------------------
    # Stats Function
    # -----------------------------
    def get_stats(data):
        stats = pd.DataFrame({
            "Mean": data.mean(),
            "Median": data.median(),
            "Mode": data.mode().iloc[0],
            "Range": data.max() - data.min(),
            "Variance": data.var(),
            "Std Dev": data.std(),
            "Skewness": data.apply(lambda x: skew(x.dropna())),
            "Kurtosis": data.apply(lambda x: kurtosis(x.dropna()))
        })
        return stats

    # -----------------------------
    # Show Results
    # -----------------------------
    st.subheader(" Statistics WITH Missing Values")
    st.dataframe(get_stats(num_df))

    st.subheader(" Statistics WITHOUT Missing Values")
    st.dataframe(get_stats(num_df_no_missing))

else:
    st.info("Please upload a CSV file to continue.")
