import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def get_mitbih_data():
    print("Loading MIT-BIH Train Data...", flush=True)
    train_df = pd.read_csv('mitbih/mitbih_train.csv', header=None)
    print("Loading MIT-BIH Test Data...", flush=True)
    test_df = pd.read_csv('mitbih/mitbih_test.csv', header=None)
    
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(np.int64)
    
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(np.int64)
    
    print(f"  > Dataset loaded: {len(X_train)} train samples, {len(X_test)} test samples.", flush=True)
    
    # ECG data is usually already normalized in these CSVs (0 to 1),
    # but let's double check or apply standard scaler if needed.
    # The head showed values like 9.77e-01, so it seems normalized.
    
    num_classes = len(np.unique(y_train))
    return X_train, X_test, y_train, y_test, num_classes
