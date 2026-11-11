import io
import json
import pandas as pd

from data_processing import (
    get_dataset_profile,
    get_data_quality
)

from eda import (
    generate_descriptive_statistics,
    calculate_statistical_insights,
    generate_automatic_insights
)


def export_csv(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def export_excel(df):
    buffer = io.BytesIO()

    with pd.ExcelWriter(
        buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Cleaned Data",
            index=False
        )

        get_data_quality(df).to_excel(
            writer,
            sheet_name="Data Quality",
            index=False
        )

        generate_descriptive_statistics(df).to_excel(
            writer,
            sheet_name="Statistics"
        )

        calculate_statistical_insights(df).to_excel(
            writer,
            sheet_name="Statistical Insights",
            index=False
        )

    buffer.seek(0)

    return buffer.getvalue()


def generate_json_report(df):
    report = {
        "dataset_profile": get_dataset_profile(df),
        "data_quality": get_data_quality(df).to_dict(
            orient="records"
        ),
        "descriptive_statistics":
            generate_descriptive_statistics(df).to_dict(),
        "statistical_insights":
            calculate_statistical_insights(df).to_dict(
                orient="records"
            ),
        "automated_insights":
            generate_automatic_insights(df)
    }

    return json.dumps(
        report,
        indent=4,
        default=str
    ).encode("utf-8")