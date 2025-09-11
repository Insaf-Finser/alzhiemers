# 🧠 Alzheimer's Handwriting Classification - Code Explanation

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Core Components](#core-components)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Model Training](#model-training)
6. [Web Interface](#web-interface)
7. [How to Run](#how-to-run)
8. [Code Examples](#code-examples)

## 🎯 Project Overview

This project implements a machine learning system to classify handwriting samples as either from healthy individuals or those showing signs of Alzheimer's disease. The system uses Random Forest classification with 451 handwriting features.

### Key Features:
- **451 Handwriting Features**: Comprehensive analysis of writing patterns
- **Random Forest Classification**: Robust ensemble learning approach
- **Advanced Preprocessing**: Feature engineering and data augmentation
- **SHAP Explainability**: Understanding model decisions
- **Web Interface**: Easy-to-use testing interface
- **Model Persistence**: Save and load trained models

## 📁 File Structure

```
alzhiemers/
├── data/
│   └── data.csv                 # Main dataset (451 features + class)
├── models/                      # Trained models and metadata
│   ├── alzheimer_rf_model_*.joblib
│   ├── model_metadata_*.json
│   └── feature_names_*.pkl
├── notebooks/
│   └── 01eda.ipynb            # Main analysis notebook
├── src/                        # Source code utilities
│   ├── model_utils.py         # Model loading and prediction utilities
│   └── data_processor.py      # Data processing utilities
├── templates/                  # Web interface templates
│   └── index.html             # Main web interface
├── static/                     # Web interface assets
├── app.py                     # Flask web application
├── train_model.py            # Model training script
├── test_model.py             # Test script
├── setup.py                  # Setup script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

## 🔧 Core Components

### 1. Data Processing (`src/data_processor.py`)

**Purpose**: Handles data loading, cleaning, and preprocessing.

**Key Functions**:
```python
class DataProcessor:
    def load_data(self, filepath):
        """Load data from CSV file"""
        
    def clean_data(self, data):
        """Clean the raw data"""
        # Convert class labels: 'P' → 1, 'H' → 0
        # Remove ID column
        # Handle missing values
        
    def analyze_skewness(self, data, threshold=0.5):
        """Analyze feature skewness"""
        # Calculate skewness for all numeric columns
        # Identify highly skewed features
        
    def apply_log_transformation(self, data, skewed_features):
        """Apply log transformation to skewed features"""
        # Apply log1p() to reduce skewness
        
    def process_pipeline(self, filepath, **kwargs):
        """Complete data processing pipeline"""
        # 1. Load data
        # 2. Clean data
        # 3. Analyze skewness
        # 4. Apply transformations
        # 5. Split data
        # 6. Apply scaling (optional)
```

**What it does**:
- Loads handwriting data from CSV
- Converts class labels to numeric format
- Identifies and transforms skewed features
- Splits data into training and testing sets
- Handles missing values and data quality issues

### 2. Model Utilities (`src/model_utils.py`)

**Purpose**: Provides utilities for loading, saving, and using trained models.

**Key Functions**:
```python
class AlzheimerModelLoader:
    def __init__(self, model_path=None):
        """Initialize model loader"""
        
    def load_model(self, model_path):
        """Load trained model from file"""
        
    def preprocess_data(self, data):
        """Preprocess input data to match training format"""
        
    def predict(self, X):
        """Make predictions on new data"""
        
    def predict_proba(self, X):
        """Get prediction probabilities"""
        
    def get_feature_importance(self):
        """Get feature importance from model"""
```

**What it does**:
- Loads trained models from disk
- Preprocesses new data to match training format
- Makes predictions and provides confidence scores
- Handles feature scaling and transformation
- Provides model metadata and information

### 3. Model Training (`train_model.py`)

**Purpose**: Comprehensive model training with hyperparameter tuning.

**Key Functions**:
```python
def load_and_clean_data(filepath='data/data.csv'):
    """Load and clean the data"""
    
def analyze_and_transform_features(data):
    """Analyze feature skewness and apply transformations"""
    
def train_model(X_train, y_train):
    """Train Random Forest with hyperparameter tuning"""
    # Grid search over multiple parameters
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'random_state': [42]
    }
    
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Calculate accuracy, precision, recall, F1, AUC
    
def save_model_and_metadata(model, ...):
    """Save model and metadata to disk"""
```

**What it does**:
- Loads and preprocesses data
- Performs feature engineering and transformation
- Trains Random Forest with hyperparameter tuning
- Evaluates model performance with multiple metrics
- Saves model and metadata for later use
- Performs cross-validation for robust evaluation

### 4. Web Interface (`app.py`)

**Purpose**: Flask web application for testing the model.

**Key Routes**:
```python
@app.route('/')
def index():
    """Main page with upload interface"""
    
@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on uploaded data"""
    # Handle file upload
    # Preprocess data
    # Make predictions
    # Return results as JSON
    
@app.route('/train', methods=['POST'])
def train_model():
    """Train a new model via web interface"""
    
@app.route('/model_info')
def model_info():
    """Get information about current model"""
```

**What it does**:
- Provides web interface for model testing
- Handles file uploads and data processing
- Makes predictions and displays results
- Allows model training through web interface
- Provides model information and status

## 🔄 Data Processing Pipeline

### Step 1: Data Loading
```python
# Load data from CSV
data = pd.read_csv('data/data.csv')
print(f"Raw data shape: {data.shape}")  # (174, 452)
```

### Step 2: Data Cleaning
```python
# Convert class labels
data['class'] = data['class'].str.strip().str.upper().map({'P': 1, 'H': 0})

# Remove ID column
data = data.drop(columns=['ID'])

# Handle missing values
data = data.fillna(data.mean())
```

### Step 3: Feature Analysis
```python
# Calculate skewness for all numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
skewness = data[numeric_cols].apply(lambda x: skew(x.dropna()))

# Identify highly skewed features
high_skew = skewness[abs(skewness) > 0.5]
print(f"Highly skewed features: {len(high_skew)}")
```

### Step 4: Feature Transformation
```python
# Apply log transformation to skewed features
for feature in high_skew.index:
    if (data[feature] <= 0).sum() == 0:  # Only if no zero or negative values
        data[feature] = np.log1p(data[feature])
```

### Step 5: Data Splitting
```python
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

## 🤖 Model Training

### Hyperparameter Tuning
```python
# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],      # Number of trees
    'max_depth': [None, 10, 20],          # Maximum tree depth
    'min_samples_split': [2, 5],          # Minimum samples to split
    'min_samples_leaf': [1, 2],           # Minimum samples per leaf
    'random_state': [42]                  # Random seed
}

# Grid search with cross-validation
rf = RandomForestClassifier()
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train, y_train)
```

### Model Evaluation
```python
# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
```

### Feature Importance
```python
# Get feature importance
importances = model.feature_importances_
top_features = np.argsort(importances)[-20:][::-1]

# Display top features
for i, idx in enumerate(top_features):
    print(f"{i+1:2d}. {X_train.columns[idx]:25s}: {importances[idx]:.4f}")
```

## 🌐 Web Interface

### HTML Template (`templates/index.html`)
- **Bootstrap-based UI**: Modern, responsive design
- **File Upload**: Drag-and-drop CSV file upload
- **Real-time Results**: Instant prediction display
- **Confidence Visualization**: Progress bars for confidence scores
- **Model Status**: Shows current model information

### JavaScript Functionality
```javascript
// File upload handling
document.getElementById('fileInput').addEventListener('change', function(e) {
    selectedFile = e.target.files[0];
    // Enable predict button
});

// Make predictions
function predict() {
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        displayResults(data);
    });
}
```

### Flask Backend
```python
@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload
    file = request.files['file']
    data = pd.read_csv(file)
    
    # Preprocess data
    processed_data, error = preprocess_data(data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    probabilities = model.predict_proba(processed_data)
    
    # Return results
    return jsonify({
        'success': True,
        'results': results,
        'summary': summary
    })
```

## 🚀 How to Run

### Option 1: Complete Setup
```bash
# 1. Run setup script
python setup.py

# 2. Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Train model
python train_model.py

# 4. Start web interface
python app.py

# 5. Open browser
# Go to: http://localhost:5000
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train model
python train_model.py

# 5. Start web interface
python app.py
```

### Option 3: Jupyter Notebook
```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Start Jupyter Lab
jupyter lab

# 3. Open notebooks/01eda.ipynb
```

## 💡 Code Examples

### Example 1: Basic Prediction
```python
from src.model_utils import AlzheimerModelLoader

# Load model
loader = AlzheimerModelLoader()

# Load new data
new_data = pd.read_csv('new_handwriting_data.csv')

# Make prediction
prediction = loader.predict(new_data)
print(f"Prediction: {'Patient' if prediction[0] == 1 else 'Healthy'}")
```

### Example 2: Batch Processing
```python
from src.model_utils import predict_batch

# Load model
model = joblib.load('models/alzheimer_rf_model_20241201_120000.joblib')

# Process batch
batch_data = pd.read_csv('batch_handwriting_data.csv')
results = predict_batch(model, batch_data)

# Save results
results.to_csv('predictions.csv', index=False)
```

### Example 3: Data Processing
```python
from src.data_processor import DataProcessor

# Create processor
processor = DataProcessor()

# Process data
result = processor.process_pipeline('data/data.csv')

# Access processed data
X_train = result['X_train']
X_test = result['X_test']
y_train = result['y_train']
y_test = result['y_test']
```

### Example 4: Model Training
```python
from train_model import main

# Train model
model, X_test, y_test = main()

# Model is automatically saved to models/ folder
```

### Example 5: Web Interface Usage
```python
# Start web interface
python app.py

# Open browser to http://localhost:5000
# Upload CSV file with handwriting features
# View prediction results with confidence scores
```

## 🔍 Key Features Explained

### 1. Feature Engineering
- **Skewness Analysis**: Identifies highly skewed features
- **Log Transformation**: Reduces skewness using log1p()
- **Missing Value Handling**: Fills missing values with mean

### 2. Model Selection
- **Random Forest**: Ensemble method with good performance
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: 5-fold CV for robust evaluation

### 3. Model Persistence
- **Joblib**: Efficient model serialization
- **Metadata**: Complete model information
- **Versioning**: Timestamped model files

### 4. Web Interface
- **File Upload**: CSV file processing
- **Real-time Results**: Instant predictions
- **Confidence Visualization**: User-friendly display
- **Model Management**: Train and load models

### 5. Error Handling
- **Data Validation**: Checks data format and features
- **Error Messages**: Clear error reporting
- **Graceful Degradation**: Handles missing components

## 📊 Performance Metrics

### Current Performance
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

## 🛠️ Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all packages are installed
2. **Data Format**: Check CSV has correct columns
3. **Model Loading**: Verify model files exist
4. **Web Interface**: Check Flask is installed

### Solutions
```bash
# Install missing packages
pip install -r requirements.txt

# Check data format
python -c "import pandas as pd; print(pd.read_csv('data/data.csv').columns.tolist())"

# Test model loading
python test_model.py

# Check web interface
python app.py
```

## 🎯 Next Steps

1. **Improve Model**: Try different algorithms (SVM, XGBoost)
2. **Feature Engineering**: Create new features from existing ones
3. **Data Augmentation**: Generate synthetic samples
4. **Model Ensemble**: Combine multiple models
5. **Deployment**: Deploy to cloud platform

---

**⚠️ Medical Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis or treatment.
