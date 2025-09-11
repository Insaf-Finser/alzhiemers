
# Model Loading Utility
# Generated on 2025-09-12 04:46:07

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

class AlzheimerModelLoader:
    """Utility class to load and use trained Alzheimer's handwriting classification models"""

    def __init__(self, model_path="../models/alzheimer_rf_model_20250912_044605.joblib"):
        self.model = joblib.load(model_path)
        self.feature_names = None
        self.scaler = None

    def load_feature_names(self, feature_path="../models/feature_names_20250912_044605.pkl"):
        """Load feature names for consistency"""
        import pickle
        with open(feature_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        return self.feature_names

    def load_scaler(self, scaler_path="../models/scaler_20250912_044605.joblib"):
        """Load scaler if used during training"""
        try:
            self.scaler = joblib.load(scaler_path)
            return self.scaler
        except FileNotFoundError:
            print("No scaler found, using raw features")
            return None

    def predict(self, X):
        """Make predictions on new data"""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None

# Example usage:
# loader = AlzheimerModelLoader()
# predictions = loader.predict(new_data)
# probabilities = loader.predict_proba(new_data)
