import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io

# Load the dataset with caching
@st.cache_data
def load_data():
    try:
        # Try loading from local path (for local use)
        data = pd.read_csv("C:/Users/chris/OneDrive/Desktop/GCU/a DSC 580/car prediction/car.csv", encoding='ISO-8859-1')
    except FileNotFoundError:
        # If file is not found, load from GitHub (for Streamlit Cloud)
        url = "https://raw.githubusercontent.com/your-username/streamlit-car-prediction/main/car.csv"
        data = pd.read_csv(url, encoding='ISO-8859-1')
    
    data.dropna(inplace=True)  # Handle missing values
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    return data

# Streamlit UI
st.title("Car Purchase Prediction Data Product")
st.markdown("""
This tool predicts car purchase amounts based on financial and demographic inputs.

**Features:**
- Real-time predictions using machine learning
- Interactive data analysis
- Custom report generation
- Supports multiple users simultaneously
""")

# Load data
data = load_data()

# Select relevant features and target
if 'car_purchase_amount' in data.columns:
    features = data[['age', 'annual_salary', 'credit_card_debt', 'net_worth']]
    target = data['car_purchase_amount']

    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train the linear regression model with caching
    @st.cache_resource
    def train_model():
        model = LinearRegression()
        model.fit(features_scaled, target)
        return model

    model = train_model()

    # Function to predict car purchase amount
    def predict_car_purchase(age, annual_salary, credit_card_debt, net_worth):
        input_data = pd.DataFrame({
            'age': [age],
            'annual_salary': [annual_salary],
            'credit_card_debt': [credit_card_debt],
            'net_worth': [net_worth]
        })
        input_scaled = scaler.transform(input_data)
        predicted_amount = model.predict(input_scaled)
        return predicted_amount[0]

    # Tabs
    tabs = st.tabs(["Prediction", "Data Analysis", "Reports"])

    # Prediction Tab
    with tabs[0]:
        st.header("Car Purchase Prediction")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        annual_salary = st.number_input("Annual Salary", min_value=10000, max_value=200000, value=50000)
        credit_card_debt = st.number_input("Credit Card Debt", min_value=0, max_value=50000, value=10000)
        net_worth = st.number_input("Net Worth", min_value=0, max_value=1000000, value=100000)
        if st.button("Predict Car Purchase Amount"):
            prediction = predict_car_purchase(age, annual_salary, credit_card_debt, net_worth)
            st.success(f"Predicted Car Purchase Amount: ${prediction:,.2f}")

    # Data Analysis Tab
    with tabs[1]:
        st.header("Interactive Data Analysis")
        age_range = st.slider("Select Age Range", int(data['age'].min()), int(data['age'].max()), 
                              (int(data['age'].min()), int(data['age'].max())))
        filtered_data = data[(data['age'] >= age_range[0]) & (data['age'] <= age_range[1])]
        st.dataframe(filtered_data)
        column_to_plot = st.selectbox("Select column for histogram", data.columns)
        fig, ax = plt.subplots()
        filtered_data[column_to_plot].hist(bins=20, edgecolor='black', ax=ax)
        st.pyplot(fig)

    # Reports Tab
    with tabs[2]:
        st.header("Web-Based Report Generation")
        if st.button("Generate Sample Report"):
            report = data.describe().transpose()
            st.dataframe(report)
            buffer = io.StringIO()
            report.to_csv(buffer)
            st.download_button("Download Report", buffer.getvalue(), "report.csv", "text/csv")

    st.info("This app supports multiple users and runs in a cloud environment.")
else:
    st.error("The dataset does not contain the required columns. Please check the dataset.")
