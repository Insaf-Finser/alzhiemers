# 🧠 Alzheimer's Handwriting Classification Project

A comprehensive machine learning project that uses handwriting analysis to detect early signs of Alzheimer's disease. This project implements Random Forest classification with advanced feature engineering and provides a user-friendly interface for testing.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Code Explanation](#code-explanation)
- [Model Training](#model-training)
- [Testing Interface](#testing-interface)
- [File Structure](#file-structure)
- [Usage Examples](#usage-examples)

## 🎯 Project Overview

This project analyzes handwriting features to classify individuals as either healthy or showing signs of Alzheimer's disease. The model uses 451 handwriting features extracted from writing samples and achieves high accuracy in classification.

### Key Features:
- **451 Handwriting Features**: Comprehensive analysis of writing patterns
- **Random Forest Classification**: Robust ensemble learning approach
- **Advanced Preprocessing**: Feature engineering and data augmentation
- **SHAP Explainability**: Understanding model decisions
- **Web Interface**: Easy-to-use testing interface
- **Model Persistence**: Save and load trained models

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd alzhiemers
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Setup
```bash
python setup.py
```

## ⚡ Quick Start

### Option 1: Jupyter Notebook (Recommended)
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Start Jupyter Lab
jupyter lab

# Open notebooks/01eda.ipynb
```

### Option 2: Command Line Training
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Train model
python train_model.py

# Start web interface
python app.py
```

### Option 3: Web Interface Only
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Start web interface (will train model if needed)
python app.py
```

## 📚 Code Explanation

### 1. Data Loading and Preprocessing (`01eda.ipynb`)

```python
# Load and clean data
data = pd.read_csv('../data/data.csv')
data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
data = data.drop(columns=['ID'])
```

**What this does:**
- Loads the handwriting dataset
- Converts class labels: 'P' (Patient) → 1, 'H' (Healthy) → 0
- Removes ID column (not a feature)

### 2. Feature Engineering

```python
# Log transformation for skewed features
for col in high_skew.index:
    if (data[col] <= 0).sum() == 0:
        data[col] = np.log1p(data[col])
```

**What this does:**
- Identifies highly skewed features (skewness > 0.5)
- Applies log transformation to reduce skewness
- Improves model performance by normalizing feature distributions

### 3. Model Training

```python
# Random Forest with Grid Search
param_grid = {
    'n_estimators': [100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'random_state': [42]
}

rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
```

**What this does:**
- Uses Random Forest classifier
- Grid search finds best hyperparameters
- 5-fold cross-validation for robust evaluation
- Optimizes for ROC AUC score

### 4. Model Evaluation

```python
# Performance metrics
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred, target_names=['Healthy', 'Patient']))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
```

**What this does:**
- Makes predictions on test set
- Calculates comprehensive performance metrics
- Provides detailed classification report

## 🤖 Model Training

### Training Process

1. **Data Split**: 80% training, 20% testing
2. **Feature Engineering**: Log transformation for skewed features
3. **Hyperparameter Tuning**: Grid search optimization
4. **Cross-Validation**: 5-fold CV for robust evaluation
5. **Model Selection**: Best parameters based on ROC AUC

### Training Command
```bash
python train_model.py
```

### Expected Output
```
🚀 Starting Alzheimer's Classification Model Training
============================================================
📊 Loading data...
Dataset shape: (174, 451)
Class distribution: {1: 89, 0: 85}

🔄 Splitting data...
Train shape: (139, 450)
Test shape: (35, 450)

🤖 Training Random Forest with Grid Search...
✅ Training completed!
Best parameters: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100, 'random_state': 42}
Best CV score: 0.9740

📈 Evaluating model...
==================================================
MODEL PERFORMANCE
==================================================
              precision    recall  f1-score   support

     Healthy       0.86      0.71      0.77        17
     Patient       0.76      0.89      0.82        18

    accuracy                           0.80        35
   macro avg       0.81      0.80      0.80        35
weighted avg       0.81      0.80      0.80        35

ROC AUC Score: 0.8333
```

## 🌐 Testing Interface

### Web Interface Features

- **File Upload**: Upload handwriting data CSV files
- **Real-time Prediction**: Instant classification results
- **Confidence Scores**: Probability estimates for predictions
- **Feature Analysis**: SHAP explanations for model decisions
- **Batch Processing**: Test multiple samples at once
- **Results Export**: Download prediction results

### Starting the Interface

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Start web interface
python app.py
```

### Access the Interface
Open your browser and go to: `http://localhost:5000`

### Interface Screenshots

1. **Main Dashboard**: Upload data and view predictions
2. **Results Page**: Detailed analysis with confidence scores
3. **Feature Analysis**: SHAP explanations for model decisions

## 📁 File Structure

```
alzhiemers/
├── data/
│   ├── data.csv                 # Main dataset
│   └── processed/              # Processed data files
├── models/                     # Saved models
│   ├── alzheimer_rf_model_*.joblib
│   ├── model_metadata_*.json
│   └── model_loader_*.py
├── notebooks/
│   └── 01eda.ipynb            # Main analysis notebook
├── src/                       # Source code
│   ├── model_utils.py         # Model utilities
│   └── data_processor.py      # Data processing
├── static/                    # Web interface assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/                 # HTML templates
│   ├── index.html
│   ├── results.html
│   └── analysis.html
├── app.py                     # Flask web application
├── train_model.py            # Command-line training
├── setup.py                  # Setup script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 💡 Usage Examples

### Example 1: Basic Prediction

```python
from src.model_utils import load_model, predict

# Load trained model
model = load_model('models/alzheimer_rf_model_20241201_120000.joblib')

# Prepare new data (must have same features as training data)
new_data = pd.read_csv('new_handwriting_data.csv')

# Make prediction
prediction = predict(model, new_data)
print(f"Prediction: {'Patient' if prediction[0] == 1 else 'Healthy'}")
```

### Example 2: Batch Processing

```python
import pandas as pd
from src.model_utils import load_model, predict_batch

# Load model
model = load_model('models/alzheimer_rf_model_20241201_120000.joblib')

# Load batch data
batch_data = pd.read_csv('batch_handwriting_data.csv')

# Process batch
results = predict_batch(model, batch_data)

# Save results
results.to_csv('predictions.csv', index=False)
```

### Example 3: SHAP Analysis

```python
import shap
from src.model_utils import load_model

# Load model
model = load_model('models/alzheimer_rf_model_20241201_120000.joblib')

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values for a sample
shap_values = explainer.shap_values(sample_data)

# Plot SHAP summary
shap.summary_plot(shap_values, sample_data)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Not Found**
   - Ensure `data/data.csv` exists
   - Check file permissions

3. **Model Loading Errors**
   - Verify model file exists in `models/` folder
   - Check model compatibility

4. **Web Interface Not Starting**
   ```bash
   pip install flask
   python app.py
   ```

### Performance Optimization

1. **Memory Issues**
   - Reduce batch size
   - Use feature selection

2. **Slow Training**
   - Reduce `n_estimators`
   - Use fewer CV folds

3. **Low Accuracy**
   - Check data quality
   - Try different hyperparameters
   - Use feature engineering

## 📊 Model Performance

### Current Performance Metrics
- **Accuracy**: 80.0%
- **Precision**: 76.2%
- **Recall**: 88.9%
- **F1-Score**: 82.1%
- **ROC AUC**: 83.3%

### Top Features
1. `total_time23` - Total writing time for task 23
2. `total_time17` - Total writing time for task 17
3. `total_time15` - Total writing time for task 15
4. `air_time17` - Air time (pen lifted) for task 17
5. `disp_index23` - Displacement index for task 23

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Handwriting samples from Alzheimer's patients and healthy controls
- Libraries: scikit-learn, pandas, matplotlib, seaborn, SHAP
- Inspiration: Medical AI research for early disease detection

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**⚠️ Medical Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis or treatment.