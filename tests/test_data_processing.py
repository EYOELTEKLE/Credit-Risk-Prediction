import pytest
from src.data_processing import process_data

def test_process_data():
    # Dummy test
    assert process_data("dummy_raw.csv", "dummy_processed.csv") is None
