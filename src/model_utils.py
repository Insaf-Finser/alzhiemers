#!/usr/bin/env python3
"""
Model utilities for Alzheimer's handwriting classification
"""

import os
import joblib
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

class AlzheimerModelLoader:
    """Utility class to load and use trained Alzheimer's handwriting classification models"""
    
    def __init__(self, model_path=None, feature_path=None, scaler_path=None):
        """
        Initialize the model loader
        
        Args:
            model_path (str): Path to the trained model file
            feature_path (str): Path to the feature names file
            scaler_path (str): Path to the scaler file (optional)
        """
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.metadata = None
        
        # Auto-detect latest model if no path provided
        if model_path is None:
            model_path = self._find_latest_model()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        if feature_path and os.path.exists(feature_path):
            self.load_feature_names(feature_path)
        elif model_path:
            # Try to find feature names file
            feature_path = self._find_latest_features()
            if feature_path:
                self.load_feature_names(feature_path)
        
        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)
    
    def _find_latest_model(self):
        """Find the latest trained model file"""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return None
        
        model_files = [f for f in os.listdir(models_dir) 
                      if f.startswith('alzheimer_rf_model_') and f.endswith('.joblib')]
        
        if not model_files:
            return None
        
        latest_model = sorted(model_files)[-1]
        return os.path.join(models_dir, latest_model)
    
    def _find_latest_features(self):
        """Find the latest feature names file"""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return None
        
        feature_files = [f for f in os.listdir(models_dir) 
                        if f.startswith('feature_names_') and f.endswith('.pkl')]
        
        if not feature_files:
            return None
        
        latest_features = sorted(feature_files)[-1]
        return os.path.join(models_dir, latest_features)
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_feature_names(self, feature_path):
        """Load feature names for consistency"""
        try:
            with open(feature_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Feature names loaded from: {feature_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading feature names: {e}")
            return False
    
    def load_scaler(self, scaler_path):
        """Load scaler if used during training"""
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded from: {scaler_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            return False
    
    def preprocess_data(self, data):
        """Preprocess the input data to match training format"""
        try:
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Check if ID column exists and remove it
            if 'ID' in data.columns:
                data = data.drop(columns=['ID'])
            
            # Ensure we have the right number of features
            if self.feature_names and len(data.columns) != len(self.feature_names):
                # Try to match feature names
                available_features = [col for col in data.columns if col in self.feature_names]
                if len(available_features) < len(self.feature_names) * 0.8:  # At least 80% of features
                    raise ValueError(f"Data has {len(data.columns)} features, but model expects {len(self.feature_names)} features")
                data = data[available_features]
            
            # Handle missing values
            data = data.fillna(data.mean())
            
            # Apply scaler if available
            if self.scaler is not None:
                data = self.scaler.transform(data)
            
            return data, None
            
        except Exception as e:
            return None, str(e)
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Preprocess data
        processed_data, error = self.preprocess_data(X)
        if error:
            raise ValueError(f"Error preprocessing data: {error}")
        
        # Make predictions
        return self.model.predict(processed_data)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Preprocess data
        processed_data, error = self.preprocess_data(X)
        if error:
            raise ValueError(f"Error preprocessing data: {error}")
        
        # Get probabilities
        return self.model.predict_proba(processed_data)
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "n_features": len(self.feature_names) if self.feature_names else "Unknown",
            "feature_names_loaded": self.feature_names is not None,
            "scaler_loaded": self.scaler is not None
        }
        
        # Add model-specific parameters
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info["max_depth"] = self.model.max_depth
        if hasattr(self.model, 'random_state'):
            info["random_state"] = self.model.random_state
        
        return info

def load_model(model_path=None):
    """Convenience function to load a model"""
    loader = AlzheimerModelLoader(model_path)
    return loader.model if loader.model else None

def predict(model, X):
    """Convenience function to make predictions"""
    if model is None:
        raise ValueError("Model is None")
    
    # Ensure data is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    return model.predict(X)

def predict_proba(model, X):
    """Convenience function to get prediction probabilities"""
    if model is None:
        raise ValueError("Model is None")
    
    # Ensure data is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    return model.predict_proba(X)

def predict_batch(model, X):
    """Make predictions on batch data and return detailed results"""
    if model is None:
        raise ValueError("Model is None")
    
    # Ensure data is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'prediction_label': ['Patient' if p == 1 else 'Healthy' for p in predictions],
        'confidence': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0],
        'probability_healthy': probabilities[:, 0] if probabilities.shape[1] > 1 else probabilities[:, 0],
        'probability_patient': probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
    })
    
    return results

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    if model is None:
        raise ValueError("Model is None")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] > 1 else model.predict_proba(X_test)[:, 0]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'Patient'], output_dict=True)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': report
    }

def save_model(model, filepath, feature_names=None, metadata=None):
    """Save a trained model with metadata"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        joblib.dump(model, filepath)
        
        # Save feature names if provided
        if feature_names is not None:
            feature_path = filepath.replace('.joblib', '_features.pkl')
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_names, f)
        
        # Save metadata if provided
        if metadata is not None:
            metadata_path = filepath.replace('.joblib', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved to: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Load model
    loader = AlzheimerModelLoader()
    
    if loader.model is not None:
        print("Model loaded successfully!")
        print("Model info:", loader.get_model_info())
        
        # Example prediction (you would load your actual data)
        # sample_data = pd.read_csv('your_data.csv')
        # predictions = loader.predict(sample_data)
        # print("Predictions:", predictions)
    else:
        print("No model found. Please train a model first.")
