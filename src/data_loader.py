import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.numerical_cols = []
        self.nominal_cols = []

    # function for loading input data
    def load_data(self, limit=None):
        try:
            self.data = pd.read_csv(self.file_path)

            # applying limit if it exists
            if limit is not None and limit < len(self.data):
                print(f'Limiting dataset to first {limit} rows')
                self.data = self.data.head(limit)
            print(f'Loaded {len(self.data)} rows with {len(self.data.columns)} features.')
            
            # identifying numerical and nominal features
            self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.nominal_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()
            return self.data
        
        except FileNotFoundError:
            print(f'Error: File {self.file_path} not found.')
            return None
        
        except Exception as e:
            print(f'Error loading data: {e}')
            return None
        
    # function for preprocessing data
    def preprocess_data(self):
        if self.data is None:
            print(f'Error: Data not loaded. Call load_data() first.')
            return None, None, None
        
        processed_data = self.data.copy()

        # if there is missing values in numerical feature we replace them with mean value for that feature
        for col in self.numerical_cols:
            if processed_data[col].isnull().any():
                mean_value = processed_data[col].mean()
                processed_data[col].fillna(mean_value, inplace=True)
        
        # if there is missing values in nominal feature we replace them with mode value (highest frequency) for that feature
        for col in self.nominal_cols:
            if processed_data[col].isnull().any():
                mode_value = processed_data[col].mode()[0]
                processed_data[col].fillna(mode_value, inplace=True)

        # performing normalization on numerical features
        if self.numerical_cols:
            scaler = MinMaxScaler()
            processed_data[self.numerical_cols] = scaler.fit_transform(processed_data[self.numerical_cols])
        
        return processed_data, self.numerical_cols, self.nominal_cols