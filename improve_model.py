#!/usr/bin/env python3
"""
Improved model training with better hyperparameters and feature selection
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
import joblib
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    """Load and preprocess the data with better feature engineering"""
    print("📊 Loading and preprocessing data...")
    
    # Load data
    data = pd.read_csv('data/data.csv')
    print(f"   • Raw data shape: {data.shape}")
    
    # Clean data
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    data = data.drop(columns=['ID'])
    
    # Separate features and target
    X = data.drop(columns=['class'])
    y = data['class']
    
    print(f"   • Cleaned data shape: {data.shape}")
    print(f"   • Class distribution: {y.value_counts().to_dict()}")
    
    return X, y

def advanced_feature_engineering(X):
    """Apply advanced feature engineering"""
    print("\n🔧 Applying advanced feature engineering...")
    
    X_processed = X.copy()
    
    # 1. Handle skewness more aggressively
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    skewness = X_processed[numeric_cols].apply(lambda x: skew(x.dropna()))
    
    # Apply different transformations based on skewness level
    for col in numeric_cols:
        if abs(skewness[col]) > 1.0:  # Highly skewed
            if (X_processed[col] <= 0).sum() == 0:
                X_processed[col] = np.log1p(X_processed[col])
        elif abs(skewness[col]) > 0.5:  # Moderately skewed
            if (X_processed[col] <= 0).sum() == 0:
                X_processed[col] = np.sqrt(X_processed[col])
    
    print(f"   • Applied transformations to skewed features")
    
    # 2. Create new features
    # Time-based ratios
    time_cols = [col for col in X_processed.columns if 'total_time' in col]
    air_time_cols = [col for col in X_processed.columns if 'air_time' in col]
    paper_time_cols = [col for col in X_processed.columns if 'paper_time' in col]
    
    # Create efficiency features
    for i in range(1, 26):  # Assuming 25 tasks
        if f'total_time{i}' in X_processed.columns and f'air_time{i}' in X_processed.columns:
            X_processed[f'efficiency_{i}'] = X_processed[f'paper_time{i}'] / (X_processed[f'total_time{i}'] + 1e-8)
    
    # Create pressure stability features
    pressure_mean_cols = [col for col in X_processed.columns if 'pressure_mean' in col]
    pressure_var_cols = [col for col in X_processed.columns if 'pressure_var' in col]
    
    for i in range(1, 26):
        if f'pressure_mean{i}' in X_processed.columns and f'pressure_var{i}' in X_processed.columns:
            X_processed[f'pressure_stability_{i}'] = X_processed[f'pressure_var{i}'] / (X_processed[f'pressure_mean{i}'] + 1e-8)
    
    print(f"   • Created {len(X_processed.columns) - len(X.columns)} new features")
    print(f"   • Total features: {X_processed.shape[1]}")
    
    return X_processed

def feature_selection(X, y, k=200):
    """Select the most important features"""
    print(f"\n🎯 Selecting top {k} features...")
    
    # Use SelectKBest with f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"   • Selected {len(selected_features)} features from {X.shape[1]}")
    
    return X_selected, selected_features, selector

def train_improved_model(X, y):
    """Train an improved model with better hyperparameters"""
    print("\n🤖 Training improved model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature selection
    X_train_selected, selected_features, selector = feature_selection(X_train, y_train, k=150)
    X_test_selected = selector.transform(X_test)
    
    # Define improved hyperparameters
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True, False],
        'random_state': [42]
    }
    
    # Train with grid search
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
    )
    
    print("   • Starting grid search...")
    grid_search.fit(X_train_selected, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"   • Best parameters: {best_params}")
    print(f"   • Best CV score: {best_score:.4f}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_selected)
    y_prob = best_model.predict_proba(X_test_selected)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📈 IMPROVED MODEL PERFORMANCE:")
    print(f"   • Test Accuracy: {test_accuracy:.4f}")
    print(f"   • Best CV Score: {best_score:.4f}")
    
    print(f"\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Patient']))
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return best_model, X_train_selected, X_test_selected, y_train, y_test, selected_features, selector

def save_improved_model(model, X_train, X_test, y_train, y_test, selected_features, selector):
    """Save the improved model and metadata"""
    print("\n💾 Saving improved model...")
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"models/improved_alzheimer_rf_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    print(f"   • Model saved: {model_filename}")
    
    # Save feature selector
    selector_filename = f"models/feature_selector_{timestamp}.joblib"
    joblib.dump(selector, selector_filename)
    print(f"   • Feature selector saved: {selector_filename}")
    
    # Save selected feature names
    feature_names_filename = f"models/selected_features_{timestamp}.pkl"
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(selected_features, f)
    print(f"   • Selected features saved: {feature_names_filename}")
    
    # Save metadata
    metadata = {
        "model_type": "ImprovedRandomForestClassifier",
        "timestamp": timestamp,
        "training_date": datetime.now().isoformat(),
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "max_features": model.max_features,
        "bootstrap": model.bootstrap,
        "n_features_selected": len(selected_features),
        "n_features_original": X_train.shape[1],
        "n_samples_train": X_train.shape[0],
        "n_samples_test": X_test.shape[0],
        "selected_features": selected_features,
        "target_classes": ["Healthy", "Patient"]
    }
    
    metadata_filename = f"models/improved_model_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   • Metadata saved: {metadata_filename}")
    
    return timestamp

def test_improved_model(model, selector, selected_features):
    """Test the improved model on our sample data"""
    print("\n🧪 Testing improved model on sample data...")
    
    # Load test samples
    healthy = pd.read_csv('test_healthy_sample.csv')
    patient = pd.read_csv('test_patient_sample.csv')
    
    # Preprocess samples (same as training)
    healthy_processed = advanced_feature_engineering(healthy.drop(columns=['ID', 'class']))
    patient_processed = advanced_feature_engineering(patient.drop(columns=['ID', 'class']))
    
    # Apply feature selection
    healthy_selected = selector.transform(healthy_processed)
    patient_selected = selector.transform(patient_processed)
    
    # Make predictions
    healthy_pred = model.predict(healthy_selected)
    healthy_prob = model.predict_proba(healthy_selected)
    patient_pred = model.predict(patient_selected)
    patient_prob = model.predict_proba(patient_selected)
    
    print(f"\n📊 IMPROVED MODEL PREDICTIONS:")
    print(f"Healthy Sample:")
    print(f"  True Label: Healthy (H)")
    print(f"  Prediction: {'Patient' if healthy_pred[0] == 1 else 'Healthy'}")
    print(f"  Confidence: {healthy_prob[0][1]:.3f}")
    print(f"  Correct: {'✅' if healthy_pred[0] == 0 else '❌'}")
    
    print(f"\nPatient Sample:")
    print(f"  True Label: Patient (P)")
    print(f"  Prediction: {'Patient' if patient_pred[0] == 1 else 'Healthy'}")
    print(f"  Confidence: {patient_prob[0][1]:.3f}")
    print(f"  Correct: {'✅' if patient_pred[0] == 1 else '❌'}")

def main():
    """Main function to train improved model"""
    print("🚀 TRAINING IMPROVED ALZHEIMER'S CLASSIFICATION MODEL")
    print("=" * 70)
    
    try:
        # 1. Load and preprocess data
        X, y = load_and_preprocess_data()
        
        # 2. Apply advanced feature engineering
        X_processed = advanced_feature_engineering(X)
        
        # 3. Train improved model
        model, X_train, X_test, y_train, y_test, selected_features, selector = train_improved_model(X_processed, y)
        
        # 4. Save improved model
        timestamp = save_improved_model(model, X_train, X_test, y_train, y_test, selected_features, selector)
        
        # 5. Test on sample data
        test_improved_model(model, selector, selected_features)
        
        print(f"\n✅ IMPROVED MODEL TRAINING COMPLETED!")
        print(f"   • Model saved with timestamp: {timestamp}")
        print(f"   • Use this model for better predictions")
        
        return model, selector, selected_features
        
    except Exception as e:
        print(f"\n❌ Error training improved model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, selector, selected_features = main()
