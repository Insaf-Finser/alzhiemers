#!/usr/bin/env python3
"""
Test script for Alzheimer's handwriting classification model
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_model_loading():
    """Test model loading functionality"""
    print("🧪 Testing Model Loading...")
    
    try:
        from model_utils import AlzheimerModelLoader
        
        # Try to load the latest model
        loader = AlzheimerModelLoader()
        
        if loader.model is not None:
            print("✅ Model loaded successfully!")
            print(f"   • Model type: {type(loader.model).__name__}")
            print(f"   • Features: {len(loader.feature_names) if loader.feature_names else 'Unknown'}")
            return loader
        else:
            print("❌ No model found. Please train a model first.")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_prediction(loader):
    """Test model prediction functionality"""
    print("\n🧪 Testing Model Prediction...")
    
    if loader is None or loader.model is None:
        print("❌ No model available for testing")
        return False
    
    try:
        # Load test data
        if os.path.exists('data/data.csv'):
            data = pd.read_csv('data/data.csv')
            
            # Clean data (same as training)
            data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
            data = data.drop(columns=['ID'])
            
            # Use first 5 samples for testing
            test_data = data.drop(columns=['class']).head(5)
            true_labels = data['class'].head(5)
            
            # Make predictions
            predictions = loader.predict(test_data)
            probabilities = loader.predict_proba(test_data)
            
            print("✅ Predictions made successfully!")
            print("\nTest Results:")
            print("-" * 50)
            
            for i, (pred, prob, true_label) in enumerate(zip(predictions, probabilities, true_labels)):
                pred_label = "Patient" if pred == 1 else "Healthy"
                true_label_text = "Patient" if true_label == 1 else "Healthy"
                confidence = prob[1] if len(prob) > 1 else prob[0]
                
                print(f"Sample {i+1}:")
                print(f"  True Label: {true_label_text}")
                print(f"  Prediction: {pred_label}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Correct: {'✅' if pred == true_label else '❌'}")
                print()
            
            # Calculate accuracy
            accuracy = accuracy_score(true_labels, predictions)
            print(f"Test Accuracy: {accuracy:.3f}")
            
            return True
            
        else:
            print("❌ Test data not found. Please ensure data/data.csv exists.")
            return False
            
    except Exception as e:
        print(f"❌ Error during prediction testing: {e}")
        return False

def test_web_interface():
    """Test if web interface can start"""
    print("\n🧪 Testing Web Interface...")
    
    try:
        import flask
        print("✅ Flask is available")
        
        # Check if app.py exists
        if os.path.exists('app.py'):
            print("✅ app.py found")
            print("💡 To start web interface: python app.py")
            return True
        else:
            print("❌ app.py not found")
            return False
            
    except ImportError:
        print("❌ Flask not installed. Run: pip install flask")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("\n🧪 Testing Data Processing...")
    
    try:
        from data_processor import DataProcessor
        
        if os.path.exists('data/data.csv'):
            processor = DataProcessor()
            result = processor.process_pipeline('data/data.csv')
            
            if result:
                print("✅ Data processing successful!")
                print(f"   • Training data: {result['X_train'].shape}")
                print(f"   • Test data: {result['X_test'].shape}")
                print(f"   • Features: {len(result['feature_names'])}")
                return True
            else:
                print("❌ Data processing failed")
                return False
        else:
            print("❌ Data file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Alzheimer's Classification Model Tests")
    print("=" * 60)
    
    # Test 1: Model loading
    loader = test_model_loading()
    
    # Test 2: Prediction functionality
    if loader:
        test_prediction(loader)
    
    # Test 3: Data processing
    test_data_processing()
    
    # Test 4: Web interface
    test_web_interface()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("\n💡 Next steps:")
    print("   1. Train model: python train_model.py")
    print("   2. Start web interface: python app.py")
    print("   3. Open Jupyter notebook: jupyter lab")

if __name__ == "__main__":
    main()
