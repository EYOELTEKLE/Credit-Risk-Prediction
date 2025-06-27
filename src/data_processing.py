import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from xverse.transformer import WOETransformer

from sklearn.cluster import KMeans
from datetime import datetime
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
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg([
            ('total_amount', 'sum'),
            ('avg_amount', 'mean'),
            ('transaction_count', 'count'),
            ('std_amount', 'std'),
        ]).reset_index()
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
def process_data(
    raw_data_path,
    processed_data_path,
    customer_id_col='customer_id',
    amount_col='transaction_amount',
    datetime_col='transaction_datetime',
    target_col='target',
    missing_strategy='median', # 'mean', 'median', 'most_frequent', 'knn', 'remove'
    scaling_strategy='standard', # 'standard' or 'minmax'
    encode_nominal=True, # True: OneHot, False: Label
    remove_missing_thresh=None # If set, remove rows/cols with missing > thresh (fraction)
):
    """
    Process raw data and save processed/model-ready data.
    """
    # Load raw data
    df = pd.read_csv(raw_data_path)

    # Aggregate and temporal features
    df = AggregateFeatures(customer_id_col=customer_id_col, amount_col=amount_col).fit_transform(df)
    df = TemporalFeatures(datetime_col=datetime_col).fit_transform(df)

    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if datetime_col in cat_cols:
        cat_cols.remove(datetime_col)
    if target_col in num_cols:
        num_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # Optionally remove rows/columns with too much missing
    if remove_missing_thresh is not None:
        df = df.loc[df.isnull().mean(axis=1) < remove_missing_thresh, :]
        df = df.loc[:, df.isnull().mean(axis=0) < remove_missing_thresh]

    # Choose imputer
    if missing_strategy == 'knn':
        num_imputer = KNNImputer()
    elif missing_strategy in ['mean', 'median', 'most_frequent']:
        num_imputer = SimpleImputer(strategy=missing_strategy)
    elif missing_strategy == 'remove':
        df = df.dropna()
        num_imputer = 'passthrough'
    else:
        num_imputer = SimpleImputer(strategy='median')

    # Choose scaler
    if scaling_strategy == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Categorical encoding
    if encode_nominal:
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    else:
        # Use LabelEncoder for each categorical column
        class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                self.encoders_ = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.columns}
                return self
            def transform(self, X):
                X = X.copy()
                for col, le in self.encoders_.items():
                    X[col] = le.transform(X[col].astype(str))
                return X
        cat_encoder = MultiColumnLabelEncoder()

    # ColumnTransformer for robust column-wise transforms
    preprocessor = ColumnTransformer([
        ('num', num_imputer, num_cols),
        ('cat', cat_encoder, cat_cols),
    ], remainder='passthrough')

    # Full pipeline
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('woe_iv', WOEIVFeatures(target_col=target_col)),
        ('scaler', scaler),
    ])

    # Fit/transform
    processed = pipeline.fit_transform(df)
    if isinstance(processed, np.ndarray):
        processed = pd.DataFrame(processed)

    # Save processed data
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed.to_csv(processed_data_path, index=False)

def assign_high_risk_label(
    transactions_df,
    processed_df,
    customer_id_col='customer_id',
    amount_col='transaction_amount',
    datetime_col='transaction_datetime',
    n_clusters=3,
    random_state=42,
    snapshot_date=None
):
    """
    Assign high-risk label to customers using RFM and KMeans clustering.
    transactions_df: Raw or minimally processed transactions DataFrame (must include customer_id_col, amount_col, datetime_col)
    processed_df: The processed/model-ready DataFrame to merge the label into
    Returns: processed_df with new column 'is_high_risk'
    """
    # If snapshot_date not provided, use max date + 1 day
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(transactions_df[datetime_col]).max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    # Calculate RFM
    rfm = transactions_df.groupby(customer_id_col).agg({
        datetime_col: lambda x: (snapshot_date - pd.to_datetime(x).max()).days,
        amount_col: ['count', 'sum']
    })
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm.reset_index()

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm['cluster'] = clusters

    # Identify high-risk cluster: high recency, low freq, low monetary
    centers = kmeans.cluster_centers_
    high_risk_idx = np.argmax(centers[:,0] - centers[:,1] - centers[:,2])
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_idx).astype(int)

    # Merge label into processed_df
    result = processed_df.merge(rfm[[customer_id_col, 'is_high_risk']], on=customer_id_col, how='left')
    result['is_high_risk'] = result['is_high_risk'].fillna(0).astype(int)
    return result

# Example usage:
# raw_df = pd.read_csv('data/raw/transactions.csv')
# processed_df = pd.read_csv('data/processed/model_ready.csv')
# labeled_df = assign_high_risk_label(raw_df, processed_df)
# labeled_df.to_csv('data/processed/model_ready_labeled.csv', index=False)
