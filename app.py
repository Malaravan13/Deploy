from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

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
        data = request.get_json()
        
        input_df = pd.DataFrame([data])
        
        input_df['year'] = 2024
        input_df['month'] = 12
        input_df['day'] = 15
        
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        
        input_df = input_df[feature_names]
        
        
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
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

