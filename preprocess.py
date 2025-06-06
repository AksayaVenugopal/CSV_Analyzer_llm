import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class UniversalPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.column_types = {}
        self.preprocessing_steps = []
        
    def analyze_data(self, df):
        """Analyze dataset to understand structure and data types"""
        analysis = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        # Detect potential datetime columns that are stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    analysis['datetime_columns'].append(col)
                    analysis['categorical_columns'].remove(col)
                except:
                    continue
        
        return analysis
    
    def handle_missing_values(self, df, strategy='auto'):
        """Handle missing values with different strategies"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)
            
            if missing_pct == 0:
                continue
            elif missing_pct > 0.5:
                print(f"Warning: {col} has {missing_pct:.1%} missing values. Consider dropping.")
                continue
            
            if df_clean[col].dtype in ['int64', 'float64']:
                if strategy == 'auto':
                    # Use median for skewed data, mean for normal data
                    if abs(df_clean[col].skew()) > 1:
                        fill_value = df_clean[col].median()
                        method = 'median'
                    else:
                        fill_value = df_clean[col].mean()
                        method = 'mean'
                else:
                    if strategy == 'knn':
                        imputer = KNNImputer(n_neighbors=5)
                        df_clean[col] = imputer.fit_transform(df_clean[[col]]).ravel()
                        self.imputers[col] = imputer
                        continue
                    else:
                        fill_value = df_clean[col].median() if strategy == 'median' else df_clean[col].mean()
                        method = strategy
                
                df_clean[col].fillna(fill_value, inplace=True)
                self.imputers[col] = {'method': method, 'value': fill_value}
                
            else:  # Categorical
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_value, inplace=True)
                self.imputers[col] = {'method': 'mode', 'value': mode_value}
        
        return df_clean
    
    def detect_outliers(self, df, columns=None, method='iqr'):
        """Detect outliers in numeric columns"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col]))
                outliers[col] = df[z_scores > 3].index
        
        return outliers
    
    def handle_outliers(self, df, outliers, method='cap'):
        """Handle outliers with different strategies"""
        df_clean = df.copy()
        
        for col, outlier_indices in outliers.items():
            if len(outlier_indices) == 0:
                continue
                
            if method == 'remove':
                df_clean = df_clean.drop(outlier_indices)
            elif method == 'cap':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def encode_categorical(self, df, encoding_type='auto'):
        """Encode categorical variables"""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if encoding_type == 'auto':
                # Use label encoding for high cardinality, one-hot for low cardinality
                if unique_values > 10:
                    encoder = LabelEncoder()
                    df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = {'type': 'label', 'encoder': encoder}
                else:
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
            
            elif encoding_type == 'label':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = {'type': 'label', 'encoder': encoder}
                
            elif encoding_type == 'onehot':
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                self.encoders[col] = {'type': 'onehot', 'columns': dummies.columns.tolist()}
        
        return df_encoded
    
    def scale_features(self, df, scaling_type='standard'):
        """Scale numerical features"""
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
        
        if scaling_type == 'standard':
            scaler = StandardScaler()
        elif scaling_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df_scaled
        
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        self.scalers['numeric'] = scaler
        
        return df_scaled
    
    def handle_datetime(self, df):
        """Extract features from datetime columns"""
        df_processed = df.copy()
        
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                try:
                    df_processed[col] = pd.to_datetime(df_processed[col])
                except:
                    continue
            
            if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
                # Extract datetime features
                df_processed[f'{col}_year'] = df_processed[col].dt.year
                df_processed[f'{col}_month'] = df_processed[col].dt.month
                df_processed[f'{col}_day'] = df_processed[col].dt.day
                df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
                df_processed[f'{col}_hour'] = df_processed[col].dt.hour
                
                # Drop original datetime column
                df_processed = df_processed.drop(col, axis=1)
        
        return df_processed
    
    def preprocess(self, df, config=None):
        """Main preprocessing pipeline"""
        if config is None:
            config = {
                'handle_missing': True,
                'missing_strategy': 'auto',
                'handle_outliers': True,
                'outlier_method': 'cap',
                'encode_categorical': True,
                'encoding_type': 'auto',
                'scale_features': True,
                'scaling_type': 'standard',
                'handle_datetime': True
            }
        
        print("Starting data preprocessing...")
        print(f"Original shape: {df.shape}")
        
        # Analyze data
        analysis = self.analyze_data(df)
        print(f"Numeric columns: {len(analysis['numeric_columns'])}")
        print(f"Categorical columns: {len(analysis['categorical_columns'])}")
        print(f"Datetime columns: {len(analysis['datetime_columns'])}")
        
        df_processed = df.copy()
        
        # Handle datetime columns
        if config['handle_datetime']:
            df_processed = self.handle_datetime(df_processed)
            print("✓ Datetime features extracted")
        
        # Handle missing values
        if config['handle_missing']:
            df_processed = self.handle_missing_values(df_processed, config['missing_strategy'])
            print("✓ Missing values handled")
        
        # Handle outliers
        if config['handle_outliers']:
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            outliers = self.detect_outliers(df_processed, numeric_cols)
            df_processed = self.handle_outliers(df_processed, outliers, config['outlier_method'])
            print("✓ Outliers handled")
        
        # Encode categorical variables
        if config['encode_categorical']:
            df_processed = self.encode_categorical(df_processed, config['encoding_type'])
            print("✓ Categorical variables encoded")
        
        # Scale features
        if config['scale_features']:
            df_processed = self.scale_features(df_processed, config['scaling_type'])
            print("✓ Features scaled")
        
        print(f"Final shape: {df_processed.shape}")
        print("Preprocessing completed!")
        
        return df_processed
    
    def transform(self, df):
        """Transform new data using fitted preprocessors"""
        df_transformed = df.copy()
        
        # Apply stored transformations
        for col, imputer_info in self.imputers.items():
            if col in df_transformed.columns:
                if isinstance(imputer_info, dict):
                    df_transformed[col].fillna(imputer_info['value'], inplace=True)
                else:
                    df_transformed[col] = imputer_info.transform(df_transformed[[col]]).ravel()
        
        # Apply encoders
        for col, encoder_info in self.encoders.items():
            if col in df_transformed.columns:
                if encoder_info['type'] == 'label':
                    df_transformed[col] = encoder_info['encoder'].transform(df_transformed[col].astype(str))
                elif encoder_info['type'] == 'onehot':
                    dummies = pd.get_dummies(df_transformed[col], prefix=col)
                    df_transformed = pd.concat([df_transformed.drop(col, axis=1), dummies], axis=1)
        
        # Apply scalers
        if 'numeric' in self.scalers:
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
            df_transformed[numeric_cols] = self.scalers['numeric'].transform(df_transformed[numeric_cols])
        
        return df_transformed

    def load_csv(self, file_path, **kwargs):
        """Load CSV file with intelligent parsing"""
        try:
            # Try different encodings if needed
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    print(f"✓ CSV loaded successfully with {encoding} encoding")
                    print(f"Shape: {df.shape}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read CSV with any common encoding")
            
            return df
        
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def save_processed_data(self, df, output_path):
        """Save processed data to CSV"""
        try:
            df.to_csv(output_path, index=False)
            print(f"✓ Processed data saved to {output_path}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def process_csv(self, input_path, output_path=None, config=None):
        """Complete pipeline: load CSV -> preprocess -> save"""
        print(f"Processing CSV file: {input_path}")
        
        # Load CSV
        df = self.load_csv(input_path)
        if df is None:
            return None
        
        print(f"\nOriginal data info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Preprocess
        df_processed = self.preprocess(df, config)
        
        # Save if output path provided
        if output_path:
            self.save_processed_data(df_processed, output_path)
        
        return df_processed
    
    def generate_report(self, df_original, df_processed, save_path=None):
        """Generate preprocessing report"""
        report = []
        report.append("=" * 50)
        report.append("DATA PREPROCESSING REPORT")
        report.append("=" * 50)
        
        report.append(f"\nORIGINAL DATA:")
        report.append(f"Shape: {df_original.shape}")
        report.append(f"Memory usage: {df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append(f"Missing values: {df_original.isnull().sum().sum()}")
        
        report.append(f"\nPROCESSED DATA:")
        report.append(f"Shape: {df_processed.shape}")
        report.append(f"Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        report.append(f"Missing values: {df_processed.isnull().sum().sum()}")
        
        report.append(f"\nCHANGES:")
        report.append(f"Rows: {df_original.shape[0]} → {df_processed.shape[0]} ({df_processed.shape[0] - df_original.shape[0]:+d})")
        report.append(f"Columns: {df_original.shape[1]} → {df_processed.shape[1]} ({df_processed.shape[1] - df_original.shape[1]:+d})")
        
        report.append(f"\nPREPROCESSING STEPS APPLIED:")
        for step in self.preprocessing_steps:
            report.append(f"✓ {step}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {save_path}")
        
        return report_text

# Example usage functions
def process_single_csv(input_file, output_file=None):
    """Simple function to process a single CSV file"""
    preprocessor = UniversalPreprocessor()
    
    # Process the CSV
    processed_data = preprocessor.process_csv(input_file, output_file)
    
    if processed_data is not None:
        print(f"\n✓ Processing completed successfully!")
        print(f"Processed data shape: {processed_data.shape}")
        return preprocessor, processed_data
    else:
        print("✗ Processing failed!")
        return None, None

def process_csv_with_custom_config(input_file, output_file=None):
    """Process CSV with custom configuration"""
    # Custom configuration example
    custom_config = {
        'handle_missing': True,
        'missing_strategy': 'median',  # Use median for all numeric columns
        'handle_outliers': True,
        'outlier_method': 'remove',    # Remove outliers instead of capping
        'encode_categorical': True,
        'encoding_type': 'onehot',     # Force one-hot encoding
        'scale_features': True,
        'scaling_type': 'minmax',      # Use min-max scaling
        'handle_datetime': True
    }
    
    preprocessor = UniversalPreprocessor()
    processed_data = preprocessor.process_csv(input_file, output_file, custom_config)
    
    return preprocessor, processed_data

# Interactive usage examples
def demo_with_sample_data():
    """Create sample data and demonstrate the preprocessor"""
    # Create sample data
    np.random.seed(42)
    data = {
        'numeric_col1': np.random.normal(100, 15, 1000),
        'numeric_col2': np.random.exponential(2, 1000),
        'category_col1': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'category_col2': np.random.choice(['Type1', 'Type2'], 1000),
        'date_col': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'target': np.random.randint(0, 2, 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values and outliers
    df.loc[np.random.choice(df.index, 50), 'numeric_col1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'category_col1'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'numeric_col1'] = np.random.normal(200, 10, 20)
    
    # Save as CSV for demonstration
    df.to_csv('sample_data.csv', index=False)
    print("✓ Sample data created and saved as 'sample_data.csv'")
    
    # Process the CSV
    preprocessor = UniversalPreprocessor()
    processed_data = preprocessor.process_csv('sample_data.csv', 'processed_sample.csv')
    
    return preprocessor, processed_data

# Easy-to-use wrapper functions
def quick_preprocess(csv_file_path, save_output=True):
    """Quick preprocessing with default settings"""
    preprocessor = UniversalPreprocessor()
    
    output_path = None
    if save_output:
        output_path = csv_file_path.replace('.csv', '_processed.csv')
    
    processed_data = preprocessor.process_csv(csv_file_path, output_path)
    return processed_data

def preprocess_with_options(csv_file_path, **options):
    """Preprocess with custom options"""
    default_config = {
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'cap',
        'encode_categorical': True,
        'encoding_type': 'auto',
        'scale_features': True,
        'scaling_type': 'standard',
        'handle_datetime': True
    }
    
    # Update with user options
    config = {**default_config, **options}
    
    preprocessor = UniversalPreprocessor()
    output_path = csv_file_path.replace('.csv', '_processed.csv')
    processed_data = preprocessor.process_csv(csv_file_path, output_path, config)
    
    return processed_data

# Command line interface (optional)
def main():
    """Command line interface - only runs if script is executed directly"""
    import sys
    import os
    
    # Hardcoded file paths for your specific use case
    input_file = r"D:\current projects\llm+rag\data\sales_data.csv"
    output_file = r"D:\current projects\llm+rag\data\psales_data.csv"
    
    # Check if files exist
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please check the file path and try again.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    print("Starting Universal Data Preprocessor...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    preprocessor, processed_data = process_single_csv(input_file, output_file)
    
    if processed_data is not None:
        print("\nFirst 5 rows of processed data:")
        print(processed_data.head())
        print(f"\n✓ Processing completed! Check your output file: {output_file}")
    else:
        print("✗ Processing failed. Please check the error messages above.")

# Example usage patterns
def show_usage_examples():
    """Show different ways to use the preprocessor"""
    
    print("=== USAGE EXAMPLES ===\n")
    
    print("1. Quick preprocessing (simplest way):")
    print("   processed_data = quick_preprocess('your_data.csv')")
    print("   # This saves output as 'your_data_processed.csv'\n")
    
    print("2. Process without saving:")
    print("   processed_data = quick_preprocess('your_data.csv', save_output=False)\n")
    
    print("3. Custom preprocessing options:")
    print("   processed_data = preprocess_with_options('your_data.csv',")
    print("                                            missing_strategy='median',")
    print("                                            outlier_method='remove',")
    print("                                            scaling_type='minmax')\n")
    
    print("4. Full control:")
    print("   preprocessor = UniversalPreprocessor()")
    print("   processed_data = preprocessor.process_csv('input.csv', 'output.csv')\n")
    
    print("5. From command line:")
    print("   python script.py your_data.csv processed_data.csv\n")

if __name__ == "__main__":
    # Show usage examples and run demo if no arguments
    show_usage_examples()
    main()