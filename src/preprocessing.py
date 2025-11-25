import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class PreprocessData:
    def __init__(self, config):
        self.config = config
        self.seed = config['seed']
        self.test_size = config['test_size']
        self.detection_threshold = config['detection_threshold']
        self.scaler = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        X = pd.read_csv(self.config['data_path'])
        y = pd.read_csv(self.config['labels_path'])
        
        if 'Unnamed: 0' in X.columns:
            X = X.drop(columns=['Unnamed: 0'])
        if 'Unnamed: 0' in y.columns:
            y = y.drop(columns=['Unnamed: 0'])
            
        if y.shape[1] == 1:
            y.columns = ['type']
        elif 'type' not in y.columns and 'label' in y.columns:
            y.columns = ['type']
            
        if self.config.get('gene_mapping_path') is not None:
            gene_names = pd.read_csv(self.config['gene_mapping_path'])
            if 'Label' in gene_names.columns:
                gene_names = gene_names.drop(columns=['Label'])
            X.columns = gene_names.columns.tolist()
            
        return X, y
    
    def preprocess(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y['type'].values,
            test_size=self.test_size,
            stratify=y['type'].values,
            random_state=self.seed
        )
        
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        
        constant_cols = X_train_df.columns[X_train_df.max() == X_train_df.min()]
        X_train_df = X_train_df.drop(columns=constant_cols)
        X_test_df = X_test_df.drop(columns=constant_cols)
        
        genes_names = np.array(X_train_df.columns)
        
        X_train_log = np.log1p(X_train_df.values)
        X_test_log = np.log1p(X_test_df.values)
        
        detection_rate = (X_train_df.values > 0).mean(axis=0)
        sparse_mask = detection_rate <= self.detection_threshold
        sparse_idx = np.where(sparse_mask)[0]
        dense_idx = np.where(~sparse_mask)[0]
        
        self.scaler = ColumnTransformer([
            ('sparse', 'passthrough', sparse_idx),
            ('dense', RobustScaler(), dense_idx)
        ], remainder='drop')
        
        X_train_scaled = self.scaler.fit_transform(X_train_log)
        X_test_scaled = self.scaler.transform(X_test_log)
        
        selected_cols = (list(X_train_df.columns[sparse_idx]) + 
                        list(X_train_df.columns[dense_idx]))
        
        X_train_df_final = pd.DataFrame(X_train_scaled, columns=selected_cols)
        X_test_df_final = pd.DataFrame(X_test_scaled, columns=selected_cols)
        
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_test_enc = self.label_encoder.transform(y_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_enc,
            'y_test': y_test_enc,
            'X_train_df': X_train_df_final,
            'X_test_df': X_test_df_final,
            'genes_names': genes_names,
            'label_encoder': self.label_encoder
        }