import json
import joblib
import pandas as pd
import numpy as np

# Load model (this will be packaged with the Lambda)
try:
    model = joblib.load('gbr_jaundice_model.joblib')
    feature_names = joblib.load('gbr_feature_names.joblib')
except FileNotFoundError:
    model = None
    feature_names = []

def lambda_handler(event, context):
    if model is None:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'Model not loaded'})
        }
    
    try:
        # Get data from event body
        if 'body' in event:
            data = json.loads(event['body'])
        else:
            data = event
            
        # Create DataFrame
        input_df = pd.DataFrame([data])
        input_df['year'] = 2024
        input_df['month'] = 12
        input_df['day'] = 15
        
        # One-hot encode
        input_df = pd.get_dummies(input_df, drop_first=True)
        
        # Ensure all features present
        missing_features = set(feature_names) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
            
        input_df = input_df[feature_names]
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'prediction': float(prediction),
                'message': f'Predicted jaundice level: {prediction:.2f} mg/dL'
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }
