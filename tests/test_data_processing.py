import pytest
import pandas as pd
from src.data_processing import AggregateFeatures, assign_high_risk_label

def test_aggregate_features():
    df = pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 2],
        'transaction_amount': [100, 200, 50, 60, 70]
    })
    agg = AggregateFeatures(customer_id_col='customer_id', amount_col='transaction_amount')
    result = agg.fit_transform(df)
    # Check correct columns
    assert 'total_amount' in result.columns
    assert 'avg_amount' in result.columns
    assert 'transaction_count' in result.columns
    assert 'std_amount' in result.columns
    # Check correct values
    cust1 = result[result['customer_id'] == 1].iloc[0]
    assert cust1['total_amount'] == 300
    assert cust1['avg_amount'] == 150
    assert cust1['transaction_count'] == 2
    cust2 = result[result['customer_id'] == 2].iloc[0]
    assert cust2['total_amount'] == 180
    assert cust2['avg_amount'] == 60
    assert cust2['transaction_count'] == 3

def test_rfm_high_risk_label():
    # Create mock transactions
    tx = pd.DataFrame({
        'customer_id': [1, 1, 2, 3],
        'transaction_amount': [100, 200, 50, 10],
        'transaction_datetime': [
            '2025-06-01', '2025-06-10', '2025-06-05', '2025-01-01'
        ]
    })
    # Processed data (just customer_id for merge)
    processed = pd.DataFrame({'customer_id': [1, 2, 3]})
    labeled = assign_high_risk_label(tx, processed, customer_id_col='customer_id', amount_col='transaction_amount', datetime_col='transaction_datetime', n_clusters=2, random_state=0, snapshot_date='2025-06-15')
    assert 'is_high_risk' in labeled.columns
    # There should be at least one high risk and one not high risk
    assert set(labeled['is_high_risk']).issubset({0, 1})
    assert labeled['is_high_risk'].sum() >= 1
