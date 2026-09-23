# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
sales_forecast_predictor_api = Flask("Sales Forcast Predictor")

# Load the trained machine learning model
model = joblib.load("sales_forecast_prediction_model_v2_0.joblib")

# Define a route for the home page (GET request)
@sales_forecast_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the Sales Forecast Prediction API!"


# ============================================
# SINGLE SALES FORECAST API
# ============================================

@sales_forecast_predictor_api.post('/v1/salesforecast')
def sales_forecast_single():

    """
    Handles a single sales forecast prediction.

    Expected JSON input:
        Product_Weight
        Product_Sugar_Content
        Product_Allocated_Area
        Product_MRP
        Store_Size
        Store_Location_City_Type
        Store_Type
        Product_Id_char
        Store_Age_Years
        Product_Type_Category
    """

    # Get JSON request data
    salesforecast_data = request.get_json()

    # ----------------------------------------
    # Validate request
    # ----------------------------------------

    if not salesforecast_data:
        return jsonify({
            'error': 'No JSON data provided'
        }), 400

    # ----------------------------------------
    # Required model features
    # ----------------------------------------

    required_features = [
        'Product_Weight',
        'Product_Sugar_Content',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Id_char',
        'Store_Age_Years',
        'Product_Type_Category'
    ]

    # Check for missing fields
    missing_features = [
        feature
        for feature in required_features
        if feature not in salesforecast_data
    ]

    if missing_features:
        return jsonify({
            'error': 'Missing required fields',
            'missing_fields': missing_features
        }), 400

    # ----------------------------------------
    # Create model input
    # ----------------------------------------

    sample = {
        'Product_Weight': salesforecast_data['Product_Weight'],
        'Product_Sugar_Content': salesforecast_data['Product_Sugar_Content'],
        'Product_Allocated_Area': salesforecast_data['Product_Allocated_Area'],
        'Product_MRP': salesforecast_data['Product_MRP'],
        'Store_Size': salesforecast_data['Store_Size'],
        'Store_Location_City_Type': salesforecast_data['Store_Location_City_Type'],
        'Store_Type': salesforecast_data['Store_Type'],
        'Product_Id_char': salesforecast_data['Product_Id_char'],
        'Store_Age_Years': salesforecast_data['Store_Age_Years'],
        'Product_Type_Category': salesforecast_data['Product_Type_Category']
    }

    # Convert to DataFrame
    input_data = pd.DataFrame([sample])

    # ----------------------------------------
    # Make prediction
    # ----------------------------------------

    predicted_sales_forecast = model.predict(input_data)[0]

    # Convert NumPy value to Python float
    predicted_sales_forecast = round(
        float(predicted_sales_forecast),
        2
    )

    # ----------------------------------------
    # Return response
    # ----------------------------------------

    return jsonify({
        'Predicted Sales forecast (in dollars)': predicted_sales_forecast
    })


# ============================================
# BATCH SALES FORECAST API
# ============================================

@sales_forecast_predictor_api.post('/v1/salesforecastbatch')
def sales_forecast_batch():

    """
    Handles batch sales forecast prediction.

    The uploaded CSV must contain:

    Product_Weight
    Product_Sugar_Content
    Product_Allocated_Area
    Product_MRP
    Store_Size
    Store_Location_City_Type
    Store_Type
    Product_Id_char
    Store_Age_Years
    Product_Type_Category

    Returns predicted Product_Store_Sales_Total
    for each row.
    """

    # ----------------------------------------
    # 1. Check if a file was uploaded
    # ----------------------------------------

    if 'file' not in request.files:
        return jsonify({
            'error': 'No CSV file uploaded'
        }), 400

    file = request.files['file']

    # ----------------------------------------
    # 2. Check filename
    # ----------------------------------------

    if file.filename == '':
        return jsonify({
            'error': 'No file selected'
        }), 400

    # ----------------------------------------
    # 3. Read CSV
    # ----------------------------------------

    try:
        input_data = pd.read_csv(file)

    except Exception as e:
        return jsonify({
            'error': 'Unable to read CSV file',
            'details': str(e)
        }), 400

    # ----------------------------------------
    # 4. Remove accidental index column
    # ----------------------------------------

    if 'Unnamed: 0' in input_data.columns:
        input_data = input_data.drop(
            columns=['Unnamed: 0']
        )

    # ----------------------------------------
    # 5. Expected model features
    # ----------------------------------------

    prediction_columns = [
        'Product_Weight',
        'Product_Sugar_Content',
        'Product_Allocated_Area',
        'Product_MRP',
        'Store_Size',
        'Store_Location_City_Type',
        'Store_Type',
        'Product_Id_char',
        'Store_Age_Years',
        'Product_Type_Category'
    ]

    # ----------------------------------------
    # 6. Check for missing columns
    # ----------------------------------------

    missing_columns = [
        column
        for column in prediction_columns
        if column not in input_data.columns
    ]

    if missing_columns:
        return jsonify({
            'error': 'Missing required columns',
            'missing_columns': missing_columns
        }), 400

    # ----------------------------------------
    # 7. Prepare model input
    # ----------------------------------------

    prediction_data = input_data[
        prediction_columns
    ].copy()

    # ----------------------------------------
    # 8. Make predictions
    # ----------------------------------------

    try:
        predicted_forecast = model.predict(
            prediction_data
        ).tolist()

    except Exception as e:
        return jsonify({
            'error': 'Model prediction failed',
            'details': str(e)
        }), 500

    # ----------------------------------------
    # 9. Create response
    # ----------------------------------------

    predictions = []

    for i, prediction in enumerate(predicted_forecast):

        predictions.append({
            'row': i,
            'Predicted Sales forecast (in dollars)': round(
                float(prediction),
                2
            )
        })

    # ----------------------------------------
    # 10. Return JSON response
    # ----------------------------------------

    return jsonify({
        'number_of_predictions': len(predictions),
        'predictions': predictions
    })

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    sales_forecast_predictor_api.run(debug=True)
