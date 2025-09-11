#!/usr/bin/env python3
"""
Data processing utilities for Alzheimer's handwriting classification
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    """Data processing class for Alzheimer's handwriting data"""
    
    def __init__(self):
        self.scaler = None
        self.feature_names = None
        self.skewed_features = None
        
    def load_data(self, filepath):
        """Load data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"✅ Data loaded: {data.shape[0]} samples, {data.shape[1]} features")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self, data):
        """Clean the raw data"""
        try:
            # Create a copy to avoid modifying original
            data_clean = data.copy()
            
            # Clean class column
            if 'class' in data_clean.columns:
                data_clean['class'] = data_clean['class'].str.strip().str.upper().map({'P': 1, 'H': 0})
            
            # Remove ID column if it exists
            if 'ID' in data_clean.columns:
                data_clean = data_clean.drop(columns=['ID'])
            
            # Handle missing values
            data_clean = data_clean.fillna(data_clean.mean())
            
            print(f"✅ Data cleaned: {data_clean.shape[0]} samples, {data_clean.shape[1]} features")
            return data_clean
            
        except Exception as e:
            print(f"❌ Error cleaning data: {e}")
            return None
    
    def analyze_skewness(self, data, threshold=0.5):
        """Analyze feature skewness"""
        try:
            # Calculate skewness for all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            skewness = data[numeric_cols].apply(lambda x: skew(x.dropna()))
            
            # Identify highly skewed features
            high_skew = skewness[abs(skewness) > threshold]
            self.skewed_features = high_skew
            
            print(f"✅ Skewness analysis completed:")
            print(f"   • Total features: {len(numeric_cols)}")
            print(f"   • Highly skewed features: {len(high_skew)}")
            print(f"   • Skewness threshold: {threshold}")
            
            return skewness, high_skew
            
        except Exception as e:
            print(f"❌ Error analyzing skewness: {e}")
            return None, None
    
    def apply_log_transformation(self, data, skewed_features=None):
        """Apply log transformation to skewed features"""
        try:
            data_transformed = data.copy()
            
            if skewed_features is None:
                skewed_features = self.skewed_features
            
            if skewed_features is None:
                print("⚠️  No skewed features identified. Run analyze_skewness() first.")
                return data_transformed
            
            # Apply log transformation to skewed features
            transformed_count = 0
            for feature in skewed_features.index:
                if (data_transformed[feature] <= 0).sum() == 0:  # Only if no zero or negative values
                    data_transformed[feature] = np.log1p(data_transformed[feature])
                    transformed_count += 1
            
            print(f"✅ Log transformation applied to {transformed_count} features")
            return data_transformed
            
        except Exception as e:
            print(f"❌ Error applying log transformation: {e}")
            return data
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"✅ Data split completed:")
            print(f"   • Training set: {X_train.shape[0]} samples")
            print(f"   • Test set: {X_test.shape[0]} samples")
            print(f"   • Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"❌ Error splitting data: {e}")
            return None, None, None, None
    
    def fit_scaler(self, X_train):
        """Fit scaler on training data"""
        try:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            print("✅ Scaler fitted on training data")
            return True
        except Exception as e:
            print(f"❌ Error fitting scaler: {e}")
            return False
    
    def transform_data(self, X):
        """Transform data using fitted scaler"""
        if self.scaler is None:
            print("⚠️  No scaler fitted. Call fit_scaler() first.")
            return X
        
        try:
            X_transformed = self.scaler.transform(X)
            print("✅ Data transformed using scaler")
            return X_transformed
        except Exception as e:
            print(f"❌ Error transforming data: {e}")
            return X
    
    def process_pipeline(self, filepath, test_size=0.2, random_state=42, apply_scaling=True):
        """Complete data processing pipeline"""
        try:
            print("🚀 Starting data processing pipeline...")
            
            # 1. Load data
            data = self.load_data(filepath)
            if data is None:
                return None
            
            # 2. Clean data
            data_clean = self.clean_data(data)
            if data_clean is None:
                return None
            
            # 3. Separate features and target
            if 'class' not in data_clean.columns:
                raise ValueError("No 'class' column found in data")
            
            X = data_clean.drop(columns=['class'])
            y = data_clean['class']
            
            # 4. Analyze skewness
            skewness, high_skew = self.analyze_skewness(X)
            
            # 5. Apply log transformation
            X_transformed = self.apply_log_transformation(X, high_skew)
            
            # 6. Split data
            X_train, X_test, y_train, y_test = self.split_data(
                X_transformed, y, test_size, random_state
            )
            
            if X_train is None:
                return None
            
            # 7. Apply scaling if requested
            if apply_scaling:
                self.fit_scaler(X_train)
                X_train_scaled = self.transform_data(X_train)
                X_test_scaled = self.transform_data(X_test)
                
                # Convert back to DataFrames to preserve column names
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Store feature names
            self.feature_names = X_train.columns.tolist()
            
            print("✅ Data processing pipeline completed successfully!")
            
            return {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': self.feature_names,
                'skewed_features': high_skew,
                'scaler': self.scaler
            }
            
        except Exception as e:
            print(f"❌ Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_data_summary(self, data):
        """Get summary statistics of the data"""
        try:
            summary = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'numeric_summary': data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else None
            }
            
            if 'class' in data.columns:
                summary['class_distribution'] = data['class'].value_counts().to_dict()
            
            return summary
            
        except Exception as e:
            print(f"❌ Error getting data summary: {e}")
            return None

def process_data(filepath, **kwargs):
    """Convenience function to process data"""
    processor = DataProcessor()
    return processor.process_pipeline(filepath, **kwargs)

# Example usage
if __name__ == "__main__":
    # Process data
    processor = DataProcessor()
    
    # Example usage
    # result = processor.process_pipeline('data/data.csv')
    # if result:
    #     print("Data processing completed successfully!")
    #     print(f"Training data shape: {result['X_train'].shape}")
    #     print(f"Test data shape: {result['X_test'].shape}")
    # else:
    #     print("Data processing failed!")
    
    print("DataProcessor class ready for use!")
