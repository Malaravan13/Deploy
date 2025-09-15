from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and feature names
try:
    model = joblib.load('gbr_jaundice_model.joblib')
    feature_names = joblib.load('gbr_feature_names.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model files not found. Please train the model first.")
    model = None
    feature_names = []

@app.route('/')
def home():
    """Simple API info"""
    return jsonify({
        'message': 'Jaundice Level Prediction API',
        'endpoints': {
            '/predict': 'POST - Send baby data to get jaundice prediction',
            '/health': 'GET - Health check'
        },
        'required_fields': [
            'gender', 'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
            'birth_head_circumference_cm', 'age_days', 'weight_kg', 'length_cm',
            'temperature_c', 'heart_rate_bpm', 'feeding_type'
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for jaundice level prediction"""
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    try:
        # Get input data
        data = request.get_json()
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([data])
        
        # Add year, month, day columns (default values)
        input_df['year'] = 2024
        input_df['month'] = 12
        input_df['day'] = 15
        
        # One-hot encode categorical variables to match training data
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'message': f'Predicted jaundice level: {prediction:.2f} mg/dL'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
