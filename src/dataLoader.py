import pandas as pd

def loadDataset(csv_path):
    """Loads dataset from a CSV file and returns a DataFrame."""
    print(f"Loading dataset from {csv_path}...")
    return pd.read_csv(csv_path)