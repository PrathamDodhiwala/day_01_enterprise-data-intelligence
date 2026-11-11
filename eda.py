import numpy as np
import pandas as pd

from data_processing import get_column_types


def generate_descriptive_statistics(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    return numeric.describe().T.round(3)


def generate_categorical_statistics(df):
    types = get_column_types(df)
    results = []

    for col in types["categorical"]:
        results.append({
            "Column": col,
            "Unique Values": df[col].nunique(dropna=True),
            "Most Frequent": (
                str(df[col].mode(dropna=True).iloc[0])
                if not df[col].mode(dropna=True).empty
                else "N/A"
            ),
            "Most Frequent Count": int(
                df[col].value_counts().iloc[0]
            ) if not df[col].value_counts().empty else 0
        })

    return pd.DataFrame(results)


def calculate_statistical_insights(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    results = []

    for col in numeric.columns:
        series = numeric[col].dropna()

        if series.empty:
            continue

        results.append({
            "Column": col,
            "Mean": series.mean(),
            "Median": series.median(),
            "Std Dev": series.std(),
            "Variance": series.var(),
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis()
        })

    return pd.DataFrame(results).round(3)


def generate_automatic_insights(df):
    insights = []

    if df.empty:
        return ["Dataset contains no records."]

    rows, cols = df.shape

    insights.append(
        f"The dataset contains {rows:,} records and {cols} columns."
    )

    missing = int(df.isna().sum().sum())

    if missing:
        insights.append(
            f"There are {missing:,} missing cells requiring attention."
        )
    else:
        insights.append("No missing values were detected.")

    duplicates = int(df.duplicated().sum())

    insights.append(
        f"{duplicates:,} duplicate rows were identified."
    )

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        correlations = numeric.corr()

        pairs = []

        for i in range(len(correlations.columns)):
            for j in range(i + 1, len(correlations.columns)):
                value = correlations.iloc[i, j]

                if pd.notna(value):
                    pairs.append((
                        correlations.columns[i],
                        correlations.columns[j],
                        value
                    ))

        if pairs:
            strongest = max(pairs, key=lambda x: abs(x[2]))

            insights.append(
                f"Strongest numerical correlation: "
                f"{strongest[0]} and {strongest[1]} "
                f"({strongest[2]:.2f})."
            )

        for col in numeric.columns:
            series = numeric[col].dropna()

            if len(series) > 2:
                skewness = series.skew()

                if abs(skewness) > 1:
                    direction = (
                        "right-skewed" if skewness > 0
                        else "left-skewed"
                    )

                    insights.append(
                        f"{col} has a {direction} distribution."
                    )

    return insights


def get_correlation_matrix(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return pd.DataFrame()

    return numeric.corr(method="pearson")