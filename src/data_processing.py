import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOETransformer
import os

# --- Custom Transformers ---
class AggregateFeatures(BaseEstimator, TransformerMixin):
    """Aggregate transaction features per customer."""
    def __init__(self, customer_id_col='customer_id', amount_col='transaction_amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Assumes X is a DataFrame
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std'),
        ]).reset_index()
        # Merge back with original X (drop duplicates)
        X = X.drop_duplicates(subset=[self.customer_id_col]).merge(agg, on=self.customer_id_col, how='left')
        return X

class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Extract hour, day, month, year from transaction datetime."""
    def __init__(self, datetime_col='transaction_datetime'):
        self.datetime_col = datetime_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X

class WOEIVFeatures(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence encoding and calculate IV."""
    def __init__(self, target_col='target'):
        self.target_col = target_col
        self.woe_transformer = None
        self.iv_ = None
    def fit(self, X, y=None):
        # Only fit on training data with target
        if self.target_col in X.columns:
            self.woe_transformer = WOETransformer(features='auto', target=self.target_col)
            self.woe_transformer.fit(X, X[self.target_col])
            self.iv_ = self.woe_transformer.iv_dict_
        return self
    def transform(self, X):
        if self.woe_transformer:
            X_woe = self.woe_transformer.transform(X)
            X_woe = X_woe.drop(columns=[self.target_col], errors='ignore')
            return X_woe
        else:
            return X

# --- Main Processing Function ---
def process_data(raw_data_path, processed_data_path,
                 customer_id_col='customer_id',
                 amount_col='transaction_amount',
                 datetime_col='transaction_datetime',
                 target_col='target'):
    """
    Process raw data and save processed/model-ready data.
    """
    # Load raw data
    df = pd.read_csv(raw_data_path)

    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if datetime_col in cat_cols:
        cat_cols.remove(datetime_col)
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # --- Pipeline Construction ---
    pipeline = Pipeline([
        ('agg_features', AggregateFeatures(customer_id_col=customer_id_col, amount_col=amount_col)),
        ('temporal_features', TemporalFeatures(datetime_col=datetime_col)),
        # Handle missing values
        ('impute', SimpleImputer(strategy='median')),
        # Feature engineering: WOE/IV
        ('woe_iv', WOEIVFeatures(target_col=target_col)),
        # Scaling (standardization)
        ('scaler', StandardScaler()),
    ])

    # Apply pipeline (fit_transform if training, transform if inference)
    processed = pipeline.fit_transform(df)
    if isinstance(processed, np.ndarray):
        processed = pd.DataFrame(processed)

    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed.to_csv(processed_data_path, index=False)

# Example usage (uncomment to run as script)
# if __name__ == "__main__":
#     process_data('data/raw/transactions.csv', 'data/processed/model_ready.csv')
