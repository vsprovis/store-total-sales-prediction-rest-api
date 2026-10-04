import streamlit as st
import pandas as pd
import requests

# Public URL of the backend Hugging Face Space (replace with your own Space URL)
BACKEND_URL = "https://miniature-carnival-wv5q4p56j9gxf59g9-7860.app.github.dev/"

# Same reference year and mappings used during feature engineering in training
REFERENCE_YEAR = 2025
PERISHABLES = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breads',
               'Breakfast', 'Seafood', 'Frozen Foods']
DRINKS = ['Soft Drinks', 'Hard Drinks']
NON_CONSUMABLES = ['Health and Hygiene', 'Household', 'Others']

st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")
st.title("SuperKart Store Sales Prediction")

# ---------------- Online (single) prediction ----------------
st.subheader("Online Prediction")

st.markdown("**Product details**")
product_type = st.selectbox("Product Type", sorted(PERISHABLES + DRINKS + NON_CONSUMABLES + [
    'Snack Foods', 'Baking Goods', 'Canned', 'Starchy Foods']))
product_weight = st.number_input("Product Weight", min_value=0.0, step=0.1, value=12.66)
product_allocated_area = st.number_input("Product Allocated Area (share of display area)",
                                         min_value=0.0, max_value=1.0, step=0.001,
                                         value=0.056, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0, step=1.0, value=147.0)

# Non-consumable products have no sugar content in the training data
if product_type in NON_CONSUMABLES:
    product_sugar_content = st.selectbox("Product Sugar Content", ["No Sugar"])
else:
    product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular"])

st.markdown("**Store details**")
store_establishment_year = st.number_input("Store Establishment Year", min_value=1950,
                                           max_value=REFERENCE_YEAR, step=1, value=2009)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1",
                                         "Supermarket Type2", "Supermarket Type3", "Food Mart"])
if store_type == "Supermarket Type3":
    st.caption("Note: Supermarket Type3 was not present in the training data, "
               "so predictions for this store type are less reliable.")

# Derive the engineered features exactly as in training
if product_type in DRINKS:
    product_id_char = "DR"
elif product_type in NON_CONSUMABLES:
    product_id_char = "NC"
else:
    product_id_char = "FD"
product_type_category = "Perishables" if product_type in PERISHABLES else "Non Perishables"
store_age_years = REFERENCE_YEAR - int(store_establishment_year)

# Payload with the 10 features the model expects
input_data = {
    'Product_Weight': product_weight,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Age_Years': store_age_years,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Id_char': product_id_char,
    'Product_Type_Category': product_type_category,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}

with st.expander("Features sent to the model"):
    st.json(input_data)

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/sales", json=input_data, timeout=60)
        if response.status_code == 200:
            prediction = response.json()['Predicted Sales']
            st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
        else:
            st.error(f"Prediction failed: {response.json().get('error', response.text)}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect to the prediction API: {e}")

# ---------------- Batch prediction ----------------
st.subheader("Batch Prediction")
st.caption("Upload a CSV with these columns: " + ", ".join(input_data.keys()))

uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(batch_df.head())

    if st.button("Predict Batch", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/salesbatch",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                timeout=120,
            )
            if response.status_code == 200:
                batch_df['Predicted_Sales'] = response.json()['predictions']
                st.success(f"Batch predictions completed for {len(batch_df)} rows.")
                st.dataframe(batch_df)
                st.download_button("Download predictions as CSV",
                                   batch_df.to_csv(index=False).encode("utf-8"),
                                   file_name="superkart_batch_predictions.csv",
                                   mime="text/csv")
            else:
                st.error(f"Batch prediction failed: {response.json().get('error', response.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to connect to the prediction API: {e}")
