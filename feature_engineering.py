"""
feature_engineering.py
-----------------------
Creates derived features to better capture creditworthiness
beyond raw financial figures.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the dataframe."""
    df = df.copy()

    # Log-transform skewed distributions
    df['Log_ApplicantIncome']   = np.log1p(df['ApplicantIncome'])
    df['Log_CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])
    df['Log_LoanAmount']        = np.log1p(df['LoanAmount'])

    # Combined household income
    df['Total_Income']     = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Total_Income'] = np.log1p(df['Total_Income'])

    # Income per dependent — purchasing power per person
    df['Income_Per_Dependent'] = df['Total_Income'] / (df['Dependents'] + 1)

    # Debt burden ratio
    df['Loan_to_Income'] = df['LoanAmount'] / (df['Total_Income'] / 1000 + 1)

    # Location × income interaction
    df['Area_Income_Interaction'] = df['Property_Area'] * df['Log_Total_Income']

    # Total household size
    df['Family_Size'] = df['Dependents'] + df['Married'] + 1

    return df


# Features dropped after statistical testing (Chi-sq / T-test not significant
# or redundant with engineered versions)
DROP_AFTER_ENGINEERING = [
    'Gender', 'Dependents', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome',
    'LoanAmount', 'Total_Income', 'Loan_Amount_Term',
]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop raw/redundant columns after engineering."""
    cols_to_drop = [c for c in DROP_AFTER_ENGINEERING if c in df.columns]
    return df.drop(columns=cols_to_drop)
