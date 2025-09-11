#!/usr/bin/env python3
"""
Quick model training script for Alzheimer's classification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def train_model():
    print("🚀 Starting Alzheimer's Classification Model Training")
    print("=" * 60)
    
    # Load data
    print("📊 Loading data...")
    data = pd.read_csv('data/data.csv')
    
    # Clean data
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    data = data.drop(columns=['ID'])
    
    print(f"Dataset shape: {data.shape}")
    print(f"Class distribution: {data['class'].value_counts().to_dict()}")
    
    # Prepare features and target
    X = data.drop(columns=['class'])
    y = data['class']
    
    # Split data
    print("\n🔄 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Define hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'random_state': [42]
    }
    
    # Train model
    print("\n🤖 Training Random Forest with Grid Search...")
    rf = RandomForestClassifier()
    grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_rf.best_estimator_
    print(f"✅ Training completed!")
    print(f"Best parameters: {grid_rf.best_params_}")
    print(f"Best CV score: {grid_rf.best_score_:.4f}")
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    y_pred = best_rf.predict(X_test)
    y_prob = best_rf.predict_proba(X_test)[:,1]
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Patient']))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Feature importance
    importances = best_rf.feature_importances_
    top_features = np.argsort(importances)[-10:][::-1]
    
    print(f"\n🔍 Top 10 Most Important Features:")
    for i, idx in enumerate(top_features):
        print(f"{i+1:2d}. {X.columns[idx]:20s}: {importances[idx]:.4f}")
    
    print(f"\n✅ Model training completed successfully!")
    return best_rf, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()

