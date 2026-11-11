import io
import re
import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def validate_dataframe(df):
    if df is None or df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("The dataset contains no columns.")

    if len(df.columns) > 500:
        raise ValueError("The dataset has too many columns (maximum 500).")

    if len(df) > 1_000_000:
        raise ValueError("Dataset exceeds the 1-million-row limit.")

    return df


def load_dataset(uploaded_file, sheet_name=0):
    filename = uploaded_file.name.lower()

    if not any(filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise ValueError("Unsupported format. Upload CSV or Excel.")

    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    buffer = io.BytesIO(file_bytes)

    if filename.endswith(".csv"):
        errors = []

        for encoding in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError as exc:
                errors.append(str(exc))
        else:
            raise ValueError("Unable to decode CSV file.")

    else:
        try:
            engine = (
                "xlrd" if filename.endswith(".xls")
                else "openpyxl"
            )

            df = pd.read_excel(
                buffer,
                sheet_name=sheet_name,
                engine=engine
            )

        except Exception as exc:
            raise ValueError(f"Excel loading failed: {exc}")

    return validate_dataframe(df)


def prepare_dataframe(df):
    df = df.copy()

    # Standardise column names.
    df.columns = [str(col).strip() for col in df.columns]

    # Replace blank column names.
    new_columns = []

    for i, col in enumerate(df.columns):
        new_columns.append(col if col else f"column_{i + 1}")

    df.columns = new_columns

    # Make duplicate column names unique.
    seen = {}
    unique_columns = []

    for col in df.columns:
        count = seen.get(col, 0)
        seen[col] = count + 1

        if count == 0:
            unique_columns.append(col)
        else:
            unique_columns.append(f"{col}_{count}")

    df.columns = unique_columns

    # Replace infinite numerical values.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def get_column_types(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()

    datetime = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    categorical = [
        col for col in df.columns
        if col not in numeric and col not in datetime
    ]

    return {
        "numeric": numeric,
        "categorical": categorical,
        "datetime": datetime
    }


def get_data_quality(df):
    missing = df.isna().sum()
    missing_percent = (missing / max(len(df), 1)) * 100

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Missing Values": missing.values,
        "Missing Percentage": missing_percent.values,
        "Unique Values": [
            df[col].nunique(dropna=True) for col in df.columns
        ]
    })

    return quality.sort_values(
        "Missing Percentage",
        ascending=False
    ).reset_index(drop=True)


def get_dataset_profile(df):
    types = get_column_types(df)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "memory_mb": round(
            df.memory_usage(deep=True).sum() / (1024 ** 2), 2
        ),
        "numeric_columns": len(types["numeric"]),
        "categorical_columns": len(types["categorical"]),
        "datetime_columns": len(types["datetime"])
    }


def clean_dataset(df, remove_duplicates=False, fill_missing=False):
    cleaned = prepare_dataframe(df)

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if fill_missing:
        for col in cleaned.columns:
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                median = cleaned[col].median()
                if pd.notna(median):
                    cleaned[col] = cleaned[col].fillna(median)
            else:
                mode = cleaned[col].mode(dropna=True)
                if not mode.empty:
                    cleaned[col] = cleaned[col].fillna(mode.iloc[0])

    return cleaned


def search_dataframe(df, search_text):
    if not search_text:
        return df

    search_text = str(search_text).lower()

    mask = df.astype(str).apply(
        lambda col: col.str.lower().str.contains(
            re.escape(search_text),
            na=False,
            regex=True
        )
    ).any(axis=1)

    return df[mask]