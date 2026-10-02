# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
store_sales_predictor_api = Flask("SuperKart Total Sales Predictor")

# Load the trained machine learning model
model = joblib.load("deployment_files/total_sales_prediction_model_v1_0.joblib")

# Define a route for the home page (GET request)
@store_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Store Sales Prediction API!"

# Define an endpoint for single store sales prediction (POST request)
@store_sales_predictor_api.post('/v1/sales')
def predict_store_sales():
    """
    This function handles POST requests to the '/v1/sales' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted store sales as a JSON response.
    """
    # Get the JSON data from the request body
    sales_data = request.get_json()

    # Extract relevant features from the JSON data matching SuperKart fields
    sample = {
        'Product_Weight': sales_data['Product_Weight'],
        'Product_Allocated_Area': sales_data['Product_Allocated_Area'],
        'Product_MRP': sales_data['Product_MRP'],
        'Store_Establishment_Year': sales_data['Store_Establishment_Year'],
        'Product_Sugar_Content': sales_data['Product_Sugar_Content'],
        'Product_Type': sales_data['Product_Type'],
        'Store_Size': sales_data['Store_Size'],
        'Store_Location_City_Type': sales_data['Store_Location_City_Type'],
        'Store_Type': sales_data['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert prediction to float and round to 2 decimal places
    predicted_sales = round(float(predicted_sales), 2)

    # Return the predicted sales
    return jsonify({'Predicted Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@store_sales_predictor_api.post('/v1/salesbatch')
def predict_store_sales_batch():
    """
    This function handles POST requests to the '/v1/salesbatch' endpoint.
    It expects a CSV file containing store and product details
    and returns the predicted sales as a list in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all rows in the DataFrame
    predicted_sales_list = model.predict(input_data).tolist()
    predicted_sales_list = [round(float(sales), 2) for sales in predicted_sales_list]

    # Return the predictions
    return jsonify({'predictions': predicted_sales_list})

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    store_sales_predictor_api.run(debug=True)
