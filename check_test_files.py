#!/usr/bin/env python3
"""
Check the test data files created
"""

import pandas as pd

print('📊 TEST DATA FILES CREATED')
print('=' * 50)

# Check healthy sample
healthy = pd.read_csv('test_healthy_sample.csv')
print('\n✅ test_healthy_sample.csv')
print(f'   • Shape: {healthy.shape}')
print(f'   • Class: {healthy["class"].iloc[0]}')
print(f'   • ID: {healthy["ID"].iloc[0]}')

# Check patient sample  
patient = pd.read_csv('test_patient_sample.csv')
print('\n✅ test_patient_sample.csv')
print(f'   • Shape: {patient.shape}')
print(f'   • Class: {patient["class"].iloc[0]}')
print(f'   • ID: {patient["ID"].iloc[0]}')

# Check combined samples
combined = pd.read_csv('test_samples.csv')
print('\n✅ test_samples.csv')
print(f'   • Shape: {combined.shape}')
print(f'   • Classes: {combined["class"].tolist()}')
print(f'   • IDs: {combined["ID"].tolist()}')

print('\n🎯 READY FOR TESTING!')
print('   Upload any of these files to the web interface at: http://localhost:5000')
