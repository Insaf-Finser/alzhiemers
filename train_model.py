#!/usr/bin/env python3
"""
Comprehensive model training script for Alzheimer's classification
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import skew
import joblib
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_notebook_best_params(path: str = 'models/notebook_best_params.json'):
    """Optionally load best hyperparameters exported from the notebook."""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                params = json.load(f)
            print(f"   • Using notebook best params from {path}: {params}")
            return params
    except Exception as e:
        print(f"   • Could not load notebook params: {e}")
    return None

def load_and_clean_data(filepath='data/data.csv'):
    """Load and clean the data"""
    print("📊 Loading and cleaning data...")
    
    # Load data
    data = pd.read_csv(filepath)
    print(f"   • Raw data shape: {data.shape}")
    
    # Clean data
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    data = data.drop(columns=['ID'])
    
    print(f"   • Cleaned data shape: {data.shape}")
    print(f"   • Class distribution: {data['class'].value_counts().to_dict()}")
    
    return data

def analyze_and_transform_features(data):
    """Analyze feature skewness and apply transformations"""
    print("\n🔍 Analyzing feature skewness...")
    
    # Calculate skewness for all numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    skewness = data[numeric_cols].apply(lambda x: skew(x.dropna()))
    
    # Identify highly skewed features
    high_skew = skewness[abs(skewness) > 0.5]
    print(f"   • Total features: {len(numeric_cols)}")
    print(f"   • Highly skewed features: {len(high_skew)}")
    
    # Apply log transformation to skewed features
    data_transformed = data.copy()
    transformed_count = 0
    
    for feature in high_skew.index:
        if (data_transformed[feature] <= 0).sum() == 0:  # Only if no zero or negative values
            data_transformed[feature] = np.log1p(data_transformed[feature])
            transformed_count += 1
    
    print(f"   • Log transformation applied to {transformed_count} features")
    
    return data_transformed, high_skew

def prepare_data(data):
    """Prepare features and target variables"""
    print("\n🔄 Preparing data for training...")
    
    # Separate features and target
    X = data.drop(columns=['class'])
    y = data['class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Correlation pruning (highly correlated > 0.95)
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    if to_drop:
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    # SelectKBest to focus on most predictive features
    selector = SelectKBest(score_func=f_classif, k=min(150, X_train.shape[1]))
    selector.fit(X_train, y_train)
    X_train = pd.DataFrame(selector.transform(X_train), index=X_train.index)
    X_test = pd.DataFrame(selector.transform(X_test), index=X_test.index)
    
    print(f"   • Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   • Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the Random Forest model with hyperparameter tuning"""
    print("\n Training Random Forest model...")
    
    # Try to use parameters exported by the notebook
    nb_params = load_notebook_best_params()
    if nb_params is not None:
        # Ensure mandatory keys exist/fallbacks
        nb_params.setdefault('random_state', 42)
        # Train directly with notebook params
        model = RandomForestClassifier(**nb_params)
        model.fit(X_train, y_train)
        print("   • Trained with notebook-provided best params")
        return model, nb_params

    # Define hyperparameters (wider search for better generalization)
    param_dist = {
        'n_estimators': np.arange(250, 551),
        'max_depth': np.append(np.arange(8, 21), None),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 6),
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'class_weight': [None, 'balanced'],
        'bootstrap': [True]
    }

    rf = RandomForestClassifier(random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=120,
        scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)

    best_rf = search.best_estimator_
    print(f"   • Best parameters: {search.best_params_}")
    print(f"   • Best CV score: {search.best_score_:.4f}")
    return best_rf, search.best_params_

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Patient']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def analyze_feature_importance(model, X_train, top_n=20):
    """Analyze and display feature importance"""
    print(f"\n🔍 Top {top_n} Most Important Features:")
    print("="*60)
    
    importances = model.feature_importances_
    top_features = np.argsort(importances)[-top_n:][::-1]
    
    for i, idx in enumerate(top_features):
        col_name = X_train.columns[idx]
        try:
            col_str = f"{col_name:25s}"
        except Exception:
            col_str = f"{str(col_name):25s}"
        print(f"{i+1:2d}. {col_str}: {importances[idx]:.4f}")
    
    return top_features, importances

def save_model_and_metadata(model, X_train, y_train, X_test, y_test, performance_metrics, best_params, skewed_features):
    """Save the trained model and metadata"""
    print("\n Saving model and metadata...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"models/alzheimer_rf_model_{timestamp}.joblib"
    joblib.dump(model, model_filename)
    print(f"   • Model saved: {model_filename}")
    
    # Save feature names
    feature_names_filename = f"models/feature_names_{timestamp}.pkl"
    with open(feature_names_filename, 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)
    print(f"   • Feature names saved: {feature_names_filename}")
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "timestamp": timestamp,
        "training_date": datetime.now().isoformat(),
        "best_parameters": best_params,
        "n_features": X_train.shape[1],
        "n_samples_train": X_train.shape[0],
        "n_samples_test": X_test.shape[0],
        "feature_names": X_train.columns.tolist(),
        "target_classes": ["Healthy", "Patient"],
        "performance_metrics": performance_metrics,
        "skewed_features": skewed_features.to_dict() if skewed_features is not None else None
    }
    
    # Convert numpy/scalar types recursively for JSON serialization
    def _to_jsonable(obj):
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_to_jsonable(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    metadata = _to_jsonable(metadata)
    
    metadata_filename = f"models/model_metadata_{timestamp}.json"
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   • Metadata saved: {metadata_filename}")
    
    # Save training and test data
    X_train.to_csv(f"models/X_train_{timestamp}.csv", index=False)
    y_train.to_csv(f"models/y_train_{timestamp}.csv", index=False)
    X_test.to_csv(f"models/X_test_{timestamp}.csv", index=False)
    y_test.to_csv(f"models/y_test_{timestamp}.csv", index=False)
    print(f"   • Training data saved: models/X_train_{timestamp}.csv")
    print(f"   • Test data saved: models/X_test_{timestamp}.csv")
    
    return timestamp

def cross_validate_model(model, X_train, y_train):
    """Perform cross-validation to assess model stability"""
    print("\n Performing cross-validation...")
    
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    print(f"   • CV Scores: {cv_scores}")
    print(f"   • Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores

def main():
    """Main training function"""
    print("Starting Comprehensive Alzheimer's Classification Model Training")
    print("=" * 80)
    
    try:
        # 1. Load and clean data
        data = load_and_clean_data()
        
        # 2. Analyze and transform features
        data_transformed, skewed_features = analyze_and_transform_features(data)
        
        # 3. Prepare data
        X_train, X_test, y_train, y_test = prepare_data(data_transformed)
        
        # 4. Train model (with class weighting search)
        model, best_params = train_model(X_train, y_train)

        # 4b. Calibrate probabilities for better decision quality
        print("\nCalibrating probabilities with cross-validation (sigmoid)...")
        calibrated_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train, y_train)
        
        # 5. Evaluate model
        performance_metrics = evaluate_model(calibrated_model, X_test, y_test)
        
        # 6. Analyze feature importance
        top_features, importances = analyze_feature_importance(model, X_train)
        
        # 7. Cross-validation
        cv_scores = cross_validate_model(model, X_train, y_train)
        
        # 8. Save model and metadata
        timestamp = save_model_and_metadata(
            calibrated_model, X_train, y_train, X_test, y_test, 
            performance_metrics, best_params, skewed_features
        )
        
        print("\n" + "="*80)
        print(" MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f" Model files saved with timestamp: {timestamp}")
        print(f"Model accuracy: {performance_metrics['accuracy']:.4f}")
        print(f"ROC AUC score: {performance_metrics['auc']:.4f}")
        print(f" CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\n💡 Next steps:")
        print("   1. Start web interface: python app.py")
        print("   2. Open Jupyter notebook: jupyter lab")
        print("   3. Test with new data using the web interface")
        
        return calibrated_model, X_test, y_test
        
    except Exception as e:
        print(f"\n Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model, X_test, y_test = main()

