import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
BACKEND_URL = "http://backend:7860"

# Set the title of the Streamlit app
st.title("SuperKart Store Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for SuperKart features
product_weight = st.number_input("Product Weight", min_value=0.0, step=0.1, value=12.5)
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, step=0.01, value=0.05)
product_mrp = st.number_input("Product MRP", min_value=0.0, step=1.0, value=150.0)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2025, step=1, value=2009)

product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar", "reg"])
product_type = st.selectbox("Product Type", [
    "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy",
    "Household", "Baking Goods", "Canned", "Health and Hygiene",
    "Meat", "Soft Drinks", "Breads", "Hard Drinks", "Others",
    "Starchy Foods", "Breakfast", "Seafood"
])

store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

# Convert user input into a dictionary matching expected model payload
input_data = {
    'Product_Weight': product_weight,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Establishment_Year': int(store_establishment_year),
    'Product_Sugar_Content': product_sugar_content,
    'Product_Type': product_type,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data)  # Send data to Flask API
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales']
        st.success(f"Predicted Total Store Sales: ${prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/salesbatch", files={"file": uploaded_file})  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
