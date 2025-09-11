#!/usr/bin/env python3
"""
Test model predictions on the sample data
"""

import pandas as pd
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_utils import AlzheimerModelLoader

def test_predictions():
    print('🧪 TESTING MODEL PREDICTIONS')
    print('=' * 50)
    
    # Load the model
    print('📥 Loading model...')
    loader = AlzheimerModelLoader()
    
    if loader.model is None:
        print('❌ No model found. Please train a model first.')
        return
    
    print('✅ Model loaded successfully!')
    
    # Test files
    test_files = [
        'test_healthy_sample.csv',
        'test_patient_sample.csv', 
        'test_samples.csv'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f'\n🔍 Testing {test_file}...')
            
            # Load test data
            test_data = pd.read_csv(test_file)
            print(f'   • Loaded {len(test_data)} sample(s)')
            
            # Make predictions
            try:
                predictions = loader.predict(test_data)
                probabilities = loader.predict_proba(test_data)
                
                print(f'   • Predictions: {predictions}')
                print(f'   • Probabilities: {probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities}')
                
                # Show results for each sample
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    pred_label = "Patient" if pred == 1 else "Healthy"
                    confidence = prob[1] if len(prob) > 1 else prob[0]
                    true_label = test_data['class'].iloc[i]
                    
                    print(f'   Sample {i+1}:')
                    print(f'     True Label: {true_label}')
                    print(f'     Prediction: {pred_label}')
                    print(f'     Confidence: {confidence:.3f}')
                    print(f'     Correct: {"✅" if (pred == 1 and true_label == "P") or (pred == 0 and true_label == "H") else "❌"}')
                
            except Exception as e:
                print(f'   ❌ Error making predictions: {e}')
        else:
            print(f'   ⚠️  File not found: {test_file}')
    
    print('\n🎯 PREDICTION TESTING COMPLETE!')
    print('   You can now test these files in the web interface at: http://localhost:5000')

if __name__ == "__main__":
    test_predictions()
