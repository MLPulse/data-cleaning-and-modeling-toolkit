import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class SafeLabelEncoder:
    """Custom LabelEncoder to handle unseen labels."""
    def __init__(self):
        self.le = LabelEncoder()
        self.classes_ = None

    def fit(self, y):
        """Fit the encoder to the unique values in y."""
        self.le.fit(y)
        self.classes_ = set(self.le.classes_)
        return self

    def transform(self, y):
        """Transform y, assigning -1 to unseen labels."""
        return np.array([self.le.transform([label])[0] if label in self.classes_ else -1 for label in y])

    def fit_transform(self, y):
        """Fit the encoder and transform y."""
        self.fit(y)
        return self.transform(y)

def filter_dataframe_by_missing_data(df, threshold=25):
    """
    Filters a DataFrame to only include columns with a percentage of missing data less than or equal to the threshold.
    """
    missing_data = df.isnull().sum()
    missing_data_percentage = (missing_data / len(df)) * 100
    features_with_less_missing_data = missing_data_percentage[missing_data_percentage <= threshold].index.tolist()
    filtered_df = df[features_with_less_missing_data]
    return filtered_df

def create_imputation_models(filtered_train, num_rows=None):
    """Create imputation models for categorical and numeric columns."""
    if num_rows is None:
        num_rows = len(filtered_train)

    df = filtered_train.copy().iloc[:num_rows]
    numeric_cols = [col for col in df.columns if df[col].dtype == 'float64' and df[col].nunique() > 5]
    categorical_cols = [col for col in df.columns if col not in numeric_cols]

    # Impute categorical columns using the most frequent value
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    label_encoders = {col: SafeLabelEncoder().fit(df[col]) for col in categorical_cols}
    df[categorical_cols] = df[categorical_cols].apply(lambda col: label_encoders[col.name].transform(col))

    # Impute numeric columns using RandomForestRegressor
    regressors = {}
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            non_missing_data = df[~df[col].isnull()]
            X_train = non_missing_data.drop(columns=[col])
            y_train = non_missing_data[col]
            regressor = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
            regressor.fit(X_train, y_train)
            regressors[col] = regressor

    return categorical_imputer, label_encoders, regressors, categorical_cols, numeric_cols

def impute_data(df, categorical_imputer, label_encoders, regressors, categorical_cols, numeric_cols, num_rows=None):
    """Impute missing values in the dataframe using the provided imputation models."""
    if num_rows is None:
        num_rows = len(df)

    df = df.iloc[:num_rows].copy(deep=True)
    df[categorical_cols] = categorical_imputer.transform(df[categorical_cols])
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            missing_data = df[df[col].isnull()]
            X_missing = missing_data.drop(columns=[col])
            df.loc[df[col].isnull(), col] = regressors[col].predict(X_missing)

    return df
