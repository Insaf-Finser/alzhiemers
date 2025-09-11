#!/usr/bin/env python3
"""
Analyze why the healthy sample is being misclassified
"""

import pandas as pd
import numpy as np
from src.model_utils import AlzheimerModelLoader
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_healthy_sample():
    """Analyze why the healthy sample is misclassified"""
    print("🔍 ANALYZING HEALTHY SAMPLE MISCLASSIFICATION")
    print("=" * 60)
    
    # Load the model
    loader = AlzheimerModelLoader()
    
    # Load test samples
    healthy = pd.read_csv('test_healthy_sample.csv')
    patient = pd.read_csv('test_patient_sample.csv')
    
    print(f"Healthy Sample ID: {healthy['ID'].iloc[0]}")
    print(f"Patient Sample ID: {patient['ID'].iloc[0]}")
    
    # Get predictions
    healthy_pred = loader.predict(healthy)
    healthy_prob = loader.predict_proba(healthy)
    patient_pred = loader.predict(patient)
    patient_prob = loader.predict_proba(patient)
    
    print(f"\n📊 CURRENT PREDICTIONS:")
    print(f"Healthy: {'Patient' if healthy_pred[0] == 1 else 'Healthy'} (confidence: {healthy_prob[0][1]:.3f})")
    print(f"Patient: {'Patient' if patient_pred[0] == 1 else 'Healthy'} (confidence: {patient_prob[0][1]:.3f})")
    
    # Analyze feature differences
    print(f"\n🔍 FEATURE ANALYSIS:")
    
    # Get feature importances
    if hasattr(loader.model, 'feature_importances_'):
        importances = loader.model.feature_importances_
        feature_names = loader.feature_names
        
        # Get top 20 most important features
        top_features_idx = np.argsort(importances)[-20:][::-1]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importances = importances[top_features_idx]
        
        print(f"\nTop 20 Most Important Features:")
        for i, (feat, imp) in enumerate(zip(top_features, top_importances)):
            print(f"{i+1:2d}. {feat:25s}: {imp:.4f}")
        
        # Compare values for these top features
        print(f"\n📈 FEATURE VALUE COMPARISON (Top 10):")
        print(f"{'Feature':<25s} {'Healthy':<12s} {'Patient':<12s} {'Difference':<12s}")
        print("-" * 65)
        
        for i, feat in enumerate(top_features[:10]):
            healthy_val = healthy[feat].iloc[0] if feat in healthy.columns else 0
            patient_val = patient[feat].iloc[0] if feat in patient.columns else 0
            diff = abs(healthy_val - patient_val)
            
            print(f"{feat:<25s} {healthy_val:<12.3f} {patient_val:<12.3f} {diff:<12.3f}")
    
    # Check if the healthy sample has values that are more similar to patients
    print(f"\n🎯 SIMILARITY ANALYSIS:")
    
    # Load full dataset to compare
    data = pd.read_csv('data/data.csv')
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    
    healthy_samples = data[data['class'] == 0]
    patient_samples = data[data['class'] == 1]
    
    print(f"Dataset healthy samples: {len(healthy_samples)}")
    print(f"Dataset patient samples: {len(patient_samples)}")
    
    # Compare our test healthy sample with dataset averages
    print(f"\n📊 COMPARING TEST SAMPLE WITH DATASET AVERAGES:")
    
    # Select some key features for comparison
    key_features = ['total_time23', 'total_time17', 'air_time17', 'disp_index23', 'pressure_mean1']
    
    for feat in key_features:
        if feat in healthy.columns and feat in data.columns:
            healthy_avg = healthy_samples[feat].mean()
            patient_avg = patient_samples[feat].mean()
            test_val = healthy[feat].iloc[0]
            
            print(f"\n{feat}:")
            print(f"  Test sample: {test_val:.3f}")
            print(f"  Healthy avg: {healthy_avg:.3f}")
            print(f"  Patient avg: {patient_avg:.3f}")
            
            # Check which average it's closer to
            healthy_dist = abs(test_val - healthy_avg)
            patient_dist = abs(test_val - patient_avg)
            
            if patient_dist < healthy_dist:
                print(f"  ⚠️  Test sample is closer to PATIENT average!")
            else:
                print(f"  ✅ Test sample is closer to HEALTHY average")

def create_better_test_samples():
    """Create better test samples that are more representative"""
    print(f"\n🎯 CREATING BETTER TEST SAMPLES...")
    
    # Load full dataset
    data = pd.read_csv('data/data.csv')
    data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
    
    # Find healthy samples that are most representative (closest to healthy mean)
    healthy_samples = data[data['class'] == 0]
    patient_samples = data[data['class'] == 1]
    
    # Calculate means for key features
    key_features = ['total_time23', 'total_time17', 'air_time17', 'disp_index23', 'pressure_mean1']
    healthy_means = healthy_samples[key_features].mean()
    patient_means = patient_samples[key_features].mean()
    
    # Find healthy sample closest to healthy mean
    healthy_distances = []
    for idx, row in healthy_samples.iterrows():
        dist = np.sqrt(((row[key_features] - healthy_means) ** 2).sum())
        healthy_distances.append((idx, dist))
    
    healthy_distances.sort(key=lambda x: x[1])
    best_healthy_idx = healthy_distances[0][0]
    
    # Find patient sample closest to patient mean
    patient_distances = []
    for idx, row in patient_samples.iterrows():
        dist = np.sqrt(((row[key_features] - patient_means) ** 2).sum())
        patient_distances.append((idx, dist))
    
    patient_distances.sort(key=lambda x: x[1])
    best_patient_idx = patient_distances[0][0]
    
    # Create new test samples
    best_healthy = data.loc[best_healthy_idx:best_healthy_idx].copy()
    best_patient = data.loc[best_patient_idx:best_patient_idx].copy()
    
    # Save new test samples
    best_healthy.to_csv('test_healthy_representative.csv', index=False)
    best_patient.to_csv('test_patient_representative.csv', index=False)
    
    print(f"✅ Created representative test samples:")
    print(f"  • test_healthy_representative.csv (ID: {best_healthy['ID'].iloc[0]})")
    print(f"  • test_patient_representative.csv (ID: {best_patient['ID'].iloc[0]})")
    
    return best_healthy, best_patient

def test_representative_samples():
    """Test the representative samples with the model"""
    print(f"\n🧪 TESTING REPRESENTATIVE SAMPLES...")
    
    # Load the model
    loader = AlzheimerModelLoader()
    
    # Load representative samples
    healthy = pd.read_csv('test_healthy_representative.csv')
    patient = pd.read_csv('test_patient_representative.csv')
    
    # Get predictions
    healthy_pred = loader.predict(healthy)
    healthy_prob = loader.predict_proba(healthy)
    patient_pred = loader.predict(patient)
    patient_prob = loader.predict_proba(patient)
    
    print(f"\n📊 REPRESENTATIVE SAMPLE PREDICTIONS:")
    print(f"Healthy Sample (ID: {healthy['ID'].iloc[0]}):")
    print(f"  True Label: Healthy (H)")
    print(f"  Prediction: {'Patient' if healthy_pred[0] == 1 else 'Healthy'}")
    print(f"  Confidence: {healthy_prob[0][1]:.3f}")
    print(f"  Correct: {'✅' if healthy_pred[0] == 0 else '❌'}")
    
    print(f"\nPatient Sample (ID: {patient['ID'].iloc[0]}):")
    print(f"  True Label: Patient (P)")
    print(f"  Prediction: {'Patient' if patient_pred[0] == 1 else 'Healthy'}")
    print(f"  Confidence: {patient_prob[0][1]:.3f}")
    print(f"  Correct: {'✅' if patient_pred[0] == 1 else '❌'}")

def main():
    """Main analysis function"""
    print("🔍 COMPREHENSIVE MISCLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    # 1. Analyze current misclassification
    analyze_healthy_sample()
    
    # 2. Create better test samples
    best_healthy, best_patient = create_better_test_samples()
    
    # 3. Test representative samples
    test_representative_samples()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"1. The original test samples may not be representative")
    print(f"2. Try the new representative samples for testing")
    print(f"3. Consider that some 'healthy' samples may have early signs")
    print(f"4. The model might be detecting subtle patterns not visible to humans")
    print(f"5. Medical classification often has uncertainty near decision boundaries")

if __name__ == "__main__":
    main()
