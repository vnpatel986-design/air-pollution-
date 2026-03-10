import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Air Pollution Prediction", layout="wide")

# Title
st.title("🌍 Air Pollution Prediction System")
st.write("Upload your dataset to predict pollution levels using Machine Learning.")

# Load trained model
@st.cache_resource
def load_model():
    with open("air_pol_pipe_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    # Select numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    st.subheader("Numeric Features Used For Prediction")
    st.write(numeric_df.columns)

    if st.button("🚀 Run Prediction"):

        prediction = model.predict(numeric_df)

        df["Prediction"] = prediction

        st.subheader("Prediction Results")
        st.dataframe(df)

        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="air_pollution_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to start prediction.")
