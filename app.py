#!/usr/bin/env python3
"""
Flask Web Application for Alzheimer's Handwriting Classification
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and data
model = None
feature_names = None
model_metadata = None

def load_latest_model():
    """Load the most recent trained model"""
    global model, feature_names, model_metadata
    
    try:
        # Find the latest model file
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return False, "Models directory not found. Please train a model first."
        
        # Look for model files
        model_files = [f for f in os.listdir(models_dir) if f.startswith('alzheimer_rf_model_') and f.endswith('.joblib')]
        if not model_files:
            return False, "No trained models found. Please train a model first."
        
        # Get the latest model file
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)
        
        # Load model
        model = joblib.load(model_path)
        
        # Load feature names
        feature_files = [f for f in os.listdir(models_dir) if f.startswith('feature_names_') and f.endswith('.pkl')]
        if feature_files:
            import pickle
            latest_features = sorted(feature_files)[-1]
            with open(os.path.join(models_dir, latest_features), 'rb') as f:
                feature_names = pickle.load(f)
        
        # Load metadata
        metadata_files = [f for f in os.listdir(models_dir) if f.startswith('model_metadata_') and f.endswith('.json')]
        if metadata_files:
            latest_metadata = sorted(metadata_files)[-1]
            with open(os.path.join(models_dir, latest_metadata), 'r') as f:
                model_metadata = json.load(f)
        
        return True, "Model loaded successfully"
        
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def preprocess_data(data):
    """Preprocess the input data to match training format"""
    try:
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        # Check if ID column exists and remove it
        if 'ID' in data.columns:
            data = data.drop(columns=['ID'])
        
        # Ensure we have the right number of features
        if len(data.columns) != len(feature_names):
            # Try to match feature names
            if feature_names:
                # Keep only columns that match feature names
                available_features = [col for col in data.columns if col in feature_names]
                if len(available_features) < len(feature_names) * 0.8:  # At least 80% of features
                    return None, f"Data has {len(data.columns)} features, but model expects {len(feature_names)} features"
                data = data[available_features]
        
        # Handle missing values
        data = data.fillna(data.mean())
        
        return data, None
        
    except Exception as e:
        return None, f"Error preprocessing data: {str(e)}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on uploaded data"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train a model first.'}), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read the uploaded file
        try:
            data = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}'}), 400
        
        # Parse optional probability threshold (default 0.5)
        try:
            threshold = float(request.form.get('threshold', '0.5'))
            if not (0.0 <= threshold <= 1.0):
                threshold = 0.5
        except Exception:
            threshold = 0.5

        # Preprocess data
        processed_data, error = preprocess_data(data)
        if error:
            return jsonify({'error': error}), 400
        
        # Make predictions
        try:
            probabilities = model.predict_proba(processed_data)
            # Apply custom threshold for patient (class 1)
            predictions = (probabilities[:, 1] >= threshold).astype(int) if probabilities.shape[1] > 1 else (probabilities[:, 0] >= threshold).astype(int)
            
            # Prepare results
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'sample_id': i + 1,
                    'prediction': 'Patient' if pred == 1 else 'Healthy',
                    'confidence': float(prob[1]) if len(prob) > 1 else float(prob[0]),
                    'probability_healthy': float(prob[0]) if len(prob) > 1 else float(prob[0]),
                    'probability_patient': float(prob[1]) if len(prob) > 1 else float(prob[0])
                }
                results.append(result)
            
            # Calculate summary statistics
            total_samples = len(results)
            patient_count = sum(1 for r in results if r['prediction'] == 'Patient')
            healthy_count = total_samples - patient_count
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            summary = {
                'total_samples': total_samples,
                'patient_count': patient_count,
                'healthy_count': healthy_count,
                'average_confidence': round(avg_confidence, 3)
            }
            
            return jsonify({
                'success': True,
                'results': results,
                'summary': summary,
                'model_info': model_metadata,
                'threshold': threshold
            })
            
        except Exception as e:
            return jsonify({'error': f'Error making predictions: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        # Import training function
        from train_model import main as train_model_main
        
        # Train the model
        model_result, X_test, y_test = train_model_main()
        
        if model_result is None:
            return jsonify({'error': 'Model training failed'}), 500
        
        # Model is already saved by the training script
        # Reload the model
        success, message = load_latest_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model trained and saved successfully',
                'model_info': model_metadata
            })
        else:
            return jsonify({'error': f'Model trained but failed to load: {message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error training model: {str(e)}'}), 500

@app.route('/train_simple', methods=['POST'])
def train_simple_model():
    """Train a simple model using the data processor"""
    try:
        from src.data_processor import DataProcessor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        import joblib
        import json
        import pickle
        
        # Process data
        processor = DataProcessor()
        result = processor.process_pipeline('data/data.csv', apply_scaling=False)
        
        if result is None:
            return jsonify({'error': 'Data processing failed'}), 500
        
        X_train = result['X_train']
        X_test = result['X_test']
        y_train = result['y_train']
        y_test = result['y_test']
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/alzheimer_rf_model_{timestamp}.joblib'
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_path)
        
        # Save feature names
        feature_path = f'models/feature_names_{timestamp}.pkl'
        with open(feature_path, 'wb') as f:
            pickle.dump(X_train.columns.tolist(), f)
        
        # Save metadata
        metadata = {
            "model_type": "RandomForestClassifier",
            "timestamp": timestamp,
            "training_date": datetime.now().isoformat(),
            "n_estimators": 100,
            "n_features": X_train.shape[1],
            "n_samples_train": X_train.shape[0],
            "n_samples_test": X_test.shape[0],
            "feature_names": X_train.columns.tolist(),
            "target_classes": ["Healthy", "Patient"],
            "performance_metrics": {
                "train_accuracy": float(train_score),
                "test_accuracy": float(test_score)
            }
        }
        
        metadata_path = f'models/model_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Reload the model
        success, message = load_latest_model()
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Simple model trained successfully! Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}',
                'model_info': model_metadata
            })
        else:
            return jsonify({'error': f'Model trained but failed to load: {message}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error training simple model: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the current model"""
    if model_metadata:
        return jsonify(model_metadata)
    else:
        return jsonify({'error': 'No model metadata available'}), 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Try to load the latest model
    success, message = load_latest_model()
    if success:
        print(f"✅ {message}")
    else:
        print(f"⚠️  {message}")
        print("You can train a model using the web interface or run: python train_model.py")
    
    # Start the Flask app
    print("🚀 Starting Alzheimer's Classification Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
