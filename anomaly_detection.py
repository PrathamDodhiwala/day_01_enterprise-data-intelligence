import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def detect_iqr_outliers(df):
    numeric = df.select_dtypes(include=np.number)

    result = pd.DataFrame(index=df.index)

    for col in numeric.columns:
        series = numeric[col]

        if series.dropna().empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        result[f"{col}_iqr_outlier"] = (
            (series < lower) | (series > upper)
        ).fillna(False)

    return result


def detect_zscore_outliers(df, threshold=3.0):
    numeric = df.select_dtypes(include=np.number)

    result = pd.DataFrame(index=df.index)

    for col in numeric.columns:
        series = numeric[col]

        mean = series.mean()
        std = series.std()

        if pd.isna(std) or std == 0:
            continue

        z_scores = (series - mean) / std

        result[f"{col}_zscore_outlier"] = (
            z_scores.abs() > threshold
        ).fillna(False)

    return result


def detect_isolation_forest(df, contamination=0.05):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty or len(numeric) < 10:
        return None

    # Remove constant and entirely missing columns.
    numeric = numeric.loc[
        :, numeric.nunique(dropna=True) > 1
    ]

    if numeric.empty:
        return None

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
    )

    predictions = model.fit_predict(numeric)

    scores = model.decision_function(numeric)

    return pd.DataFrame({
        "Isolation Forest Prediction": predictions,
        "Anomaly Score": scores,
        "Is Anomaly": predictions == -1
    }, index=df.index)


def generate_anomaly_summary(df, contamination=0.05):
    iqr = detect_iqr_outliers(df)
    zscore = detect_zscore_outliers(df)

    isolation = detect_isolation_forest(
        df, contamination
    )

    summary = {
        "iqr": iqr,
        "zscore": zscore,
        "isolation_forest": isolation
    }

    return summary