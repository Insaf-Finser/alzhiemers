# Alzheimer's Handwriting Classification

This project uses Random Forest classification with SHAP explainability to detect Alzheimer's disease from handwriting features in the DARWIN dataset.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (already set up in `venv/`)

### Setup Instructions

1. **Activate Virtual Environment**
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**
   - Place your `data.csv` file in the `data/` folder
   - The dataset should have columns: ID, class (P/H), and feature columns
   - A sample dataset is already provided for testing

4. **Run the Notebook**
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
   Then open `notebooks/01eda.ipynb`

## 📊 Dataset Format

The expected CSV format:
- `ID`: Unique identifier
- `class`: Target variable ('P' for Patient, 'H' for Healthy)
- `feature_1` to `feature_N`: Handwriting features

## 🔧 Project Structure

```
alzhiemers/
├── data/           # Dataset files
├── notebooks/      # Jupyter notebooks
├── src/           # Source code
├── models/        # Trained models
├── docs/          # Documentation
├── venv/          # Virtual environment
└── requirements.txt
```

## 📈 What the Notebook Does

1. **Data Loading & Exploration**: Loads and explores the handwriting dataset
2. **Data Cleaning**: Standardizes class labels and handles missing data
3. **Feature Engineering**: Applies transformations to skewed features
4. **Model Training**: Trains Random Forest with hyperparameter tuning
5. **Evaluation**: Comprehensive model evaluation with metrics and visualizations
6. **SHAP Analysis**: Explains model predictions using SHAP values
7. **Feature Importance**: Identifies most important handwriting features

## 🎯 Expected Results

- High accuracy classification of Alzheimer's from handwriting
- Interpretable feature importance rankings
- SHAP explanations for individual predictions
- Clinical insights for early detection

## 📝 Notes

- The notebook is designed to run locally with your own dataset
- SHAP analysis provides model interpretability for clinical applications
- All visualizations are optimized for medical research presentations
