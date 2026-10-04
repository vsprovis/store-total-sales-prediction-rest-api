# Import necessary libraries
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
store_sales_predictor_api = Flask("SuperKart Total Sales Predictor")

# Load the trained model pipeline (the file sits next to app.py inside the container)
model = joblib.load("total_sales_prediction_model_v1_0.joblib")

# Features the model was trained on, after feature engineering
NUMERIC_FEATURES = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age_Years']
CATEGORICAL_FEATURES = ['Product_Sugar_Content', 'Product_Id_char', 'Product_Type_Category',
                        'Store_Size', 'Store_Location_City_Type', 'Store_Type']
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_input(df):
    """Check required fields, apply the training-time label cleaning,
    and return the features in the order used during training."""
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    df = df[FEATURES].copy()
    # Same cleaning as in training: 'reg' is a duplicate label of 'Regular'
    df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    return df


# Home route (GET request)
@store_sales_predictor_api.get('/')
def home():
    """Return a welcome message to confirm the API is running."""
    return "Welcome to the SuperKart Store Sales Prediction API!"


# Single prediction endpoint (POST request)
@store_sales_predictor_api.post('/v1/sales')
def predict_store_sales():
    """Accept a JSON payload with the 10 model features and return the predicted sales."""
    sales_data = request.get_json()
    if not sales_data:
        return jsonify({'error': 'Request body must be JSON'}), 400
    try:
        input_data = prepare_input(pd.DataFrame([sales_data]))
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    predicted_sales = round(float(model.predict(input_data)[0]), 2)
    return jsonify({'Predicted Sales': predicted_sales})


# Batch prediction endpoint (POST request)
@store_sales_predictor_api.post('/v1/salesbatch')
def predict_store_sales_batch():
    """Accept a CSV file with the 10 model features and return one prediction per row."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        input_data = prepare_input(pd.read_csv(request.files['file']))
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    predictions = [round(float(p), 2) for p in model.predict(input_data)]
    return jsonify({'predictions': predictions})


# Run the Flask app locally in debug mode
if __name__ == '__main__':
    store_sales_predictor_api.run(debug=True)
