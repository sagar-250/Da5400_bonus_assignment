import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

def get_helena_data():
    """
    Loads the Helena dataset (ID 41169).
    Uses the full dataset as established in baseline verification.
    """
    print("Fetching Helena dataset (ID 41169)...", flush=True)
    helena = fetch_openml(data_id=41169, as_frame=False, parser='auto')
    X = helena.data.astype(np.float64)
    y = LabelEncoder().fit_transform(helena.target)
    
    print(f"  > Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes.", flush=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    
    num_classes = len(np.unique(y))
    return X_train, X_test, y_train, y_test, num_classes
