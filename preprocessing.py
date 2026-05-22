"""
preprocessing.py
----------------
Data cleaning, imputation, and encoding pipeline
for the Loan Eligibility dataset.
"""

import pandas as pd
import numpy as np


CATEGORICAL_COLS = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
NUMERICAL_COLS   = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

ENCODINGS = {
    'Gender':        {'Male': 1, 'Female': 0},
    'Married':       {'Yes': 1, 'No': 0},
    'Education':     {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
    'Dependents':    {'0': 0, '1': 1, '2': 2, '3+': 3},
}


def load_data(train_path: str = "data/raw/train.csv",
              test_path: str  = "data/raw/test.csv"):
    """Load raw train and test CSVs."""
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def clean(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Drop Loan_ID
    2. Encode target (train only)
    3. Impute missing values
    4. Encode categorical features
    """
    df = df.copy()

    # Drop ID
    if 'Loan_ID' in df.columns:
        df.drop(columns=['Loan_ID'], inplace=True)

    # Encode target
    if is_train and 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    # Impute
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Encode
    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df
