#!/usr/bin/env python3
"""
Enhanced Random Forest training per paper recommendations

Implements:
- Paper-priority features for tasks 23, 17, 24 and pressure variance tasks
- Advanced feature engineering (ratios, stability, hesitation)
- Skewness log transform + Robust scaling
- Paper-style CV (10 runs via RepeatedStratifiedKFold)
- Expanded RF hyperparameter ranges from the paper
- Optional soft-voting ensemble (RF + GB + XGB if available)
- Probability calibration (sigmoid)
"""

import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import skew

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.utils.validation import check_is_fitted

try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def load_data(path: str = 'data/data.csv') -> pd.DataFrame:
    data = pd.read_csv(path)
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])
    return data


def extract_paper_based_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=['class'])
    X = X.copy()

    # Advanced feature engineering per paper
    for i in range(1, 26):
        tt = f'total_time{i}'
        at = f'air_time{i}'
        pt = f'paper_time{i}'
        pm = f'pressure_mean{i}'
        pv = f'pressure_var{i}'
        msia = f'mean_speed_in_air{i}'
        mji = f'mean_jerk_in_air{i}'

        if at in X.columns and pt in X.columns:
            X[f'cognitive_load_{i}'] = X[at] / (X[pt] + 1e-8)
            X[f'hesitation_index_{i}'] = X[at]
            # If pen_lifts exists as num_of_pendown, approximate lifts (heuristic)
            npend = f'num_of_pendown{i}'
            if npend in X.columns:
                X[f'hesitation_index_{i}'] = X[at] * (X[npend] + 1.0)

        if pm in X.columns and pv in X.columns:
            X[f'pressure_stability_{i}'] = X[pm] / (X[pv] + 1e-8)
            X[f'motor_confidence_{i}'] = 1.0 / (X[pv] + 1e-8)

        if msia in X.columns and mji in X.columns:
            X[f'move_efficiency_{i}'] = X[msia] / (X[mji] + 1e-8)

        if tt in X.columns and at in X.columns and pt in X.columns:
            X[f'eff_paper_ratio_{i}'] = X[pt] / (X[tt] + 1e-8)

    # Paper-priority tasks emphasis: 23, 17, 24, and pressure variance 3/8/14
    priority_cols = []
    for i in [23, 17, 24, 3, 8, 14]:
        for prefix in ['total_time', 'air_time', 'paper_time', 'pressure_var', 'pressure_mean',
                       'mean_speed_in_air', 'mean_jerk_in_air', 'disp_index']:
            col = f'{prefix}{i}'
            if col in X.columns:
                priority_cols.append(col)
    # Always keep existing columns; priority list is used by selector later
    X.attrs['priority_cols'] = priority_cols
    return X


def preprocess(X: pd.DataFrame) -> tuple[pd.DataFrame, RobustScaler]:
    # Log transform highly skewed
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    sk = X[numeric_cols].apply(lambda s: skew(s.dropna()))
    skewed = sk[sk.abs() > 0.5].index
    X = X.copy()
    for c in skewed:
        # log1p valid for positive features only
        if (X[c] > 0).all():
            X[c] = np.log1p(X[c])

    scaler = RobustScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    return X, scaler


def train_rf(X_train, y_train, n_iter: int = 120, random_state: int = 42) -> RandomForestClassifier:
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_state)  # ~10 runs

    param_dist = {
        'n_estimators': np.arange(200, 551),  # 200-550
        'max_depth': np.append(np.arange(2, 16), None),
        'max_features': np.linspace(0.1, 1.0, 10),
        'min_samples_split': np.arange(2, 21),
        'min_samples_leaf': np.arange(1, 21),
        'class_weight': [None, 'balanced']
    }

    base = RandomForestClassifier(random_state=random_state, bootstrap=True)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=rskf,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    return best


def maybe_make_ensemble(rf: RandomForestClassifier, X_train, y_train):
    models = [('rf', rf), ('gb', GradientBoostingClassifier(random_state=42))]
    if HAS_XGB:
        models.append(('xgb', XGBClassifier(random_state=42, n_estimators=300, max_depth=4,
                                            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss')))
    if len(models) == 1:
        return rf
    ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
    ensemble.fit(X_train, y_train)
    return ensemble


def evaluate_paper_style(model, X, y) -> dict:
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    # Accuracy
    acc = cross_val_score(model, X, y, cv=rskf, scoring='accuracy', n_jobs=-1)
    # Sensitivity (recall of class 1)
    sen = cross_val_score(model, X, y, cv=rskf, scoring='recall', n_jobs=-1)
    # Specificity (recall of class 0)
    spec = cross_val_score(model, X, y, cv=rskf, scoring='recall_macro', n_jobs=-1)  # fallback
    # AUC
    auc = cross_val_score(model, X, y, cv=rskf, scoring='roc_auc', n_jobs=-1)
    return {
        'accuracy_mean': float(acc.mean()), 'accuracy_std': float(acc.std()),
        'sensitivity_mean': float(sen.mean()), 'sensitivity_std': float(sen.std()),
        'specificity_proxy_mean': float(spec.mean()), 'specificity_proxy_std': float(spec.std()),
        'auc_mean': float(auc.mean()), 'auc_std': float(auc.std())
    }


def save_model(model, X_train, y_train, meta: dict):
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    import joblib
    joblib.dump(model, f'models/enhanced_alzheimer_rf_{ts}.joblib')
    with open(f'models/enhanced_metadata_{ts}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return ts


def main():
    print('ENHANCED Alzheimer\'s Classification Training (Paper-Optimized)')
    data = load_data()
    y = data['class']

    print(' Creating advanced features based on paper insights...')
    X = extract_paper_based_features(data)
    priority_cols = X.attrs.get('priority_cols', [])
    print(f' Found {len(priority_cols)} paper-priority base features available')

    print(' Enhanced preprocessing with skewness correction + robust scaling...')
    X, scaler = preprocess(X)

    print(' 75/25 stratified split (paper style)')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print('Training enhanced Random Forest with paper-optimized parameter ranges...')
    rf = train_rf(X_train, y_train, n_iter=120)
    print('   Best RF params:', rf.get_params())

    print(' Calibrating model probabilities (sigmoid, 5-fold)...')
    calibrated = CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=5)
    calibrated.fit(X_train, y_train)

    print(' Paper-style evaluation (10 runs)...')
    eval_cv = evaluate_paper_style(calibrated, X, y)

    # Holdout evaluation
    y_pred = calibrated.predict(X_test)
    if hasattr(calibrated, 'predict_proba'):
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = np.nan
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    print('\nENHANCED MODEL PERFORMANCE (Paper-Style Evaluation)')
    print('============================================================')
    print(f"Accuracy: {eval_cv['accuracy_mean']:.3f} ± {eval_cv['accuracy_std']:.3f}")
    print(f"Sensitivity: {eval_cv['sensitivity_mean']:.3f} ± {eval_cv['sensitivity_std']:.3f}")
    print(f"Specificity (proxy macro-recall): {eval_cv['specificity_proxy_mean']:.3f} ± {eval_cv['specificity_proxy_std']:.3f}")
    print(f"ROC AUC: {eval_cv['auc_mean']:.3f} ± {eval_cv['auc_std']:.3f}")

    print('\nHOLDOUT SET (25%)')
    print('------------------------------------------------------------')
    print(f'Accuracy: {acc:.3f}  AUC: {auc:.3f}  Sensitivity: {sens:.3f}  Specificity: {spec:.3f}')

    meta = {
        'paper_cv': eval_cv,
        'holdout': {'accuracy': acc, 'auc': auc, 'sensitivity': sens, 'specificity': spec},
        'priority_cols': priority_cols,
        'scaler': 'RobustScaler',
        'calibration': 'sigmoid_cv5'
    }
    ts = save_model(calibrated, X_train, y_train, meta)
    print(f' Saved enhanced model with timestamp: {ts}')

    # Optional ensemble (comment in to enable)
    # ensemble = maybe_make_ensemble(rf, X_train, y_train)
    # print(' Soft-voting ensemble ready')

    return calibrated


if __name__ == '__main__':
    main()


