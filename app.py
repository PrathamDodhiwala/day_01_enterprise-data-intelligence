import os
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.io as pio

# Enforce a consistent light chart theme, including charts created directly in app.py.
pio.templates.default = "plotly_white"

from data_processing import (
    load_dataset,
    prepare_dataframe,
    get_column_types,
    get_data_quality,
    get_dataset_profile,
    clean_dataset,
    search_dataframe
)

from eda import (
    generate_descriptive_statistics,
    generate_categorical_statistics,
    calculate_statistical_insights,
    generate_automatic_insights,
    get_correlation_matrix
)

from anomaly_detection import (
    generate_anomaly_summary
)

from database import (
    init_database,
    save_dataset_to_database,
    get_registered_datasets,
    run_sql_query
)

from visualization import (
    create_histogram,
    create_boxplot,
    create_bar_chart,
    create_scatter_plot,
    create_line_chart,
    create_correlation_heatmap,
    create_auto_visualizations
)

from reporting import (
    export_csv,
    export_excel,
    generate_json_report
)


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Data Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_database()


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
:root {
    --navy: #14243A;
    --navy-2: #1D3553;
    --blue: #356AE6;
    --ink: #172B4D;
    --muted: #64748B;
    --line: #E5EAF2;
    --canvas: #F5F7FB;
    --surface: #FFFFFF;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--canvas);
    color: var(--ink);
}
[data-testid="stHeader"] {
    background: rgba(245,247,251,.92);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12233A 0%, #1B3554 100%);
    border-right: 1px solid #263F5E;
}
[data-testid="stSidebar"] * {
    color: #EAF0F8 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: rgba(255,255,255,.08);
    border: 1px dashed rgba(255,255,255,.35);
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #B8C7D9 !important;
}
[data-testid="stSidebar"] button {
    border-radius: 9px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 10px;
    border-radius: 8px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.09);
}
.main .block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] {
    color: var(--ink);
}
h1, h2, h3 {
    letter-spacing: -0.025em;
}
.main-title {
    font-size: clamp(1.65rem, 2.5vw, 2.15rem);
    line-height: 1.15;
    font-weight: 780;
    color: var(--ink) !important;
    letter-spacing: -0.04em;
}
.subtitle {
    color: var(--muted) !important;
    font-size: .93rem;
    margin-top: .45rem;
}
.eyebrow {
    color: var(--blue) !important;
    text-transform: uppercase;
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .12em;
    margin-bottom: .45rem;
}
.page-intro {
    color: var(--muted) !important;
    margin-top: -.35rem;
    margin-bottom: 1.1rem;
}
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 17px 18px;
    box-shadow: 0 3px 12px rgba(20,36,58,.035);
    min-height: 112px;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #66758B !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-size: 1.75rem !important;
    font-weight: 760 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: .75rem;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 8px 8px 2px 8px;
    box-shadow: 0 3px 12px rgba(20,36,58,.025);
}
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] hr {
    border-color: var(--line);
}
.stButton button, .stDownloadButton button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 2.55rem;
}
.stButton button[kind="primary"] {
    background: var(--blue);
    border-color: var(--blue);
}
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    border-radius: 9px;
}
[data-testid="stAlert"] {
    border-radius: 11px;
}
[data-testid="stExpander"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 11px;
}
small, .stCaption {
    color: var(--muted) !important;
}
@media (max-width: 768px) {
    .main .block-container { padding: 1rem 1rem 2rem; }
    div[data-testid="stMetric"] { padding: 13px; }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""

if "file_fingerprint" not in st.session_state:
    st.session_state.file_fingerprint = None


# --------------------------------------------------
# LOAD DEFAULT SAMPLE DATASET
# --------------------------------------------------

if st.session_state.df is None:

    sample_path = "sample_dataset.csv"

    if os.path.exists(sample_path):
        try:
            sample_df = pd.read_csv(sample_path)

            st.session_state.df = prepare_dataframe(sample_df)
            st.session_state.dataset_name = "Sample Dataset"

        except Exception as exc:
            st.error(f"Sample dataset loading failed: {exc}")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 8px 2px 14px 2px;">
      <div style="font-size:11px; letter-spacing:2px; color:#91A9C5; font-weight:800;">
        DATA INTELLIGENCE
      </div>
      <div style="font-size:29px; line-height:1.2; font-weight:850; color:#FFFFFF;">
        EDID<span style="color:#78A5FF;">.</span>
      </div>
      <div style="font-size:12px; color:#B8C7D9; margin-top:5px;">
        Enterprise analytics workspace
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:

    fingerprint = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if fingerprint != st.session_state.file_fingerprint:

        try:
            with st.spinner("Processing dataset..."):

                loaded_df = load_dataset(uploaded_file)

                st.session_state.df = prepare_dataframe(loaded_df)

                st.session_state.dataset_name = uploaded_file.name

                st.session_state.file_fingerprint = fingerprint

            st.sidebar.success("Dataset loaded successfully.")

        except Exception as exc:
            st.sidebar.error(f"Upload failed: {exc}")

if st.sidebar.button("Load Sample Dataset", use_container_width=True):

    try:
        sample_df = pd.read_csv("sample_dataset.csv")

        st.session_state.df = prepare_dataframe(sample_df)
        st.session_state.dataset_name = "Sample Dataset"
        st.session_state.file_fingerprint = None

        st.rerun()

    except Exception as exc:
        st.error(f"Could not load sample: {exc}")


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

pages = [
    "Executive Overview",
    "Dataset Explorer",
    "Data Quality Monitor",
    "Exploratory Data Analysis",
    "Statistical Insights",
    "Correlation Analysis",
    "Anomaly Detection",
    "Visualization Studio",
    "Export & Reporting"
]

selected_page = st.sidebar.radio(
    "NAVIGATION",
    pages
)

st.sidebar.divider()

st.sidebar.caption(
    "DAY 01 | 60-DAY PROJECT MARATHON"
)


# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------

if st.session_state.df is None:

    st.warning(
        "Upload a CSV/Excel dataset or load the sample dataset."
    )

    st.stop()


df = st.session_state.df

types = get_column_types(df)

profile = get_dataset_profile(df)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">DAY 01 · PROJECT MARATHON</div>'
    '<div class="main-title">Enterprise Data Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="subtitle">Your analytical workspace &nbsp;·&nbsp; '
    f'<strong>{st.session_state.dataset_name}</strong></div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==================================================

if selected_page == "Executive Overview":

    st.subheader("Executive Overview")

    st.write(
        "A consolidated view of dataset structure, "
        "quality, and analytical opportunities."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{profile['rows']:,}")

    c2.metric("Total Columns", profile["columns"])

    c3.metric("Missing Cells", f"{profile['missing_cells']:,}")

    c4.metric("Duplicate Rows", f"{profile['duplicates']:,}")

    c5, c6, c7 = st.columns(3)

    c5.metric("Numerical Features", profile["numeric_columns"])

    c6.metric("Categorical Features", profile["categorical_columns"])

    c7.metric("Memory Usage", f"{profile['memory_mb']} MB")

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    with right:

        st.subheader("Column Type Distribution")

        type_data = pd.DataFrame({
            "Type": ["Numerical", "Categorical", "Datetime"],
            "Count": [
                profile["numeric_columns"],
                profile["categorical_columns"],
                profile["datetime_columns"]
            ]
        })

        st.plotly_chart(
            __import__("plotly.express", fromlist=["px"]).pie(
                type_data,
                names="Type",
                values="Count",
                hole=0.55,
                title="Feature Composition"
            ),
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    st.divider()

    st.subheader("Automatically Generated Insights")

    for insight in generate_automatic_insights(df):
        st.info(insight)

    st.subheader("Automatic Visualizations")

    for title, fig in create_auto_visualizations(df):
        st.markdown(f"**{title}**")

        st.plotly_chart(
            fig,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )


# ==================================================
# PAGE 2: DATASET EXPLORER
# ==================================================

elif selected_page == "Dataset Explorer":

    st.subheader("Dataset Explorer")

    st.write(
        "Search, filter, inspect and query your dataset."
    )

    search_text = st.text_input(
        "Search across all columns"
    )

    filtered_df = search_dataframe(df, search_text)

    st.markdown("### Interactive Filters")

    filter_columns = st.multiselect(
        "Select categorical columns to filter",
        types["categorical"]
    )

    for col in filter_columns:

        options = sorted(
            filtered_df[col].dropna().astype(str).unique().tolist()
        )

        selected = st.multiselect(
            f"Filter {col}",
            options,
            key=f"filter_{col}"
        )

        if selected:
            filtered_df = filtered_df[
                filtered_df[col].astype(str).isin(selected)
            ]

    numeric_filter = st.selectbox(
        "Optional numerical filter",
        ["None"] + types["numeric"]
    )

    if numeric_filter != "None":

        series = pd.to_numeric(
            filtered_df[numeric_filter],
            errors="coerce"
        ).dropna()

        if not series.empty:

            minimum = float(series.min())
            maximum = float(series.max())

            if minimum < maximum:

                selected_range = st.slider(
                    f"Range: {numeric_filter}",
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum)
                )

                filtered_df = filtered_df[
                    filtered_df[numeric_filter].between(
                        selected_range[0],
                        selected_range[1]
                    )
                ]

    st.metric(
        "Filtered Records",
        f"{len(filtered_df):,}"
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=400
    )

    st.divider()

    st.subheader("SQL Database Integration")

    st.write(
        "Save your dataset to SQLite and execute read-only SQL queries."
    )

    table_name = st.text_input(
        "SQL Table Name",
        value="uploaded_dataset"
    )

    if st.button("Save Dataset to SQLite"):

        try:

            save_dataset_to_database(
                df,
                table_name
            )

            st.success(
                f"Dataset saved as '{table_name}'."
            )

        except Exception as exc:
            st.error(str(exc))

    st.markdown("**SQL Query Console**")

    query = st.text_area(
        "Enter SQL query",
        value="SELECT * FROM uploaded_dataset LIMIT 10"
    )

    if st.button("Execute SQL Query"):

        try:

            result = run_sql_query(query)

            st.success(
                f"Query returned {len(result)} rows."
            )

            st.dataframe(
                result,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

        except Exception as exc:
            st.error(f"SQL error: {exc}")

    st.markdown("### Registered Datasets")

    try:

        registered = get_registered_datasets()

        st.dataframe(
            registered,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    except Exception as exc:
        st.warning(str(exc))


# ==================================================
# PAGE 3: DATA QUALITY MONITOR
# ==================================================

elif selected_page == "Data Quality Monitor":

    st.subheader("Data Quality Monitor")

    st.write(
        "Identify missing values, duplicate records, "
        "and column-level quality issues."
    )

    quality = get_data_quality(df)

    missing = int(df.isna().sum().sum())

    duplicates = int(df.duplicated().sum())

    a, b, c = st.columns(3)

    a.metric("Missing Cells", f"{missing:,}")

    b.metric("Duplicate Records", f"{duplicates:,}")

    c.metric(
        "Completeness",
        f"{100 * (1 - missing / max(df.size, 1)):.2f}%"
    )

    st.divider()

    st.subheader("Column-Level Quality Report")

    st.dataframe(
        quality,
        use_container_width=True
    )

    missing_data = quality[
        quality["Missing Values"] > 0
    ]

    if not missing_data.empty:

        st.subheader("Missing Value Distribution")

        import plotly.express as px

        fig = px.bar(
            missing_data,
            x="Column",
            y="Missing Percentage",
            title="Missing Values by Column",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    else:

        st.success("No missing values detected.")

    st.subheader("Data Cleaning Options")

    remove_duplicates = st.checkbox(
        "Remove duplicate records"
    )

    fill_missing = st.checkbox(
        "Fill missing numerical values with median "
        "and categorical values with mode"
    )

    if st.button("Generate Cleaned Dataset"):

        cleaned = clean_dataset(
            df,
            remove_duplicates=remove_duplicates,
            fill_missing=fill_missing
        )

        st.session_state.cleaned_df = cleaned

        st.success("Cleaning operation completed.")

        st.write(
            f"Original rows: {len(df):,} | "
            f"Cleaned rows: {len(cleaned):,}"
        )

        st.dataframe(
            cleaned.head(20),
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

        st.download_button(
            "Download Cleaned CSV",
            data=export_csv(cleaned),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )


# ==================================================
# PAGE 4: EXPLORATORY DATA ANALYSIS
# ==================================================

elif selected_page == "Exploratory Data Analysis":

    st.subheader("Exploratory Data Analysis")

    st.write(
        "Automatically explore distributions, "
        "frequencies and relationships."
    )

    st.markdown("### Numerical Distributions")

    if types["numeric"]:

        selected_num = st.selectbox(
            "Select numerical feature",
            types["numeric"]
        )

        from visualization import (
            create_histogram,
            create_boxplot
        )

        col1, col2 = st.columns(2)

        with col1:

            st.plotly_chart(
                create_histogram(df, selected_num),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

        with col2:

            st.plotly_chart(
                create_boxplot(df, selected_num),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

    else:

        st.info("No numerical features available.")

    st.divider()

    st.markdown("### Categorical Analysis")

    if types["categorical"]:

        selected_cat = st.selectbox(
            "Select categorical feature",
            types["categorical"]
        )

        st.plotly_chart(
            create_bar_chart(df, selected_cat),
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    else:

        st.info("No categorical features available.")

    st.divider()

    st.markdown("### Automatic EDA Report")

    for title, fig in create_auto_visualizations(df):

        st.write(title)

        st.plotly_chart(
            fig,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )


# ==================================================
# PAGE 5: STATISTICAL INSIGHTS
# ==================================================

elif selected_page == "Statistical Insights":

    st.subheader("Statistical Insights")

    st.write(
        "Explore central tendency, dispersion, "
        "distribution shape, and categorical frequencies."
    )

    st.markdown("### Descriptive Statistics")

    numeric_stats = generate_descriptive_statistics(df)

    if not numeric_stats.empty:

        st.dataframe(
            numeric_stats,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    else:

        st.info("No numerical columns available.")

    st.divider()

    st.markdown("### Detailed Statistical Analysis")

    detailed = calculate_statistical_insights(df)

    if not detailed.empty:

        st.dataframe(
            detailed,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    st.divider()

    st.markdown("### Categorical Statistics")

    categorical_stats = generate_categorical_statistics(df)

    if not categorical_stats.empty:

        st.dataframe(
            categorical_stats,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

    else:

        st.info("No categorical columns available.")


# ==================================================
# PAGE 6: CORRELATION ANALYSIS
# ==================================================

elif selected_page == "Correlation Analysis":

    st.subheader("Correlation Analysis")

    st.write(
        "Study linear relationships between numerical features."
    )

    correlation = get_correlation_matrix(df)

    if correlation.empty:

        st.warning(
            "At least two numerical columns are required."
        )

    else:

        fig = create_correlation_heatmap(correlation)

        st.plotly_chart(
            fig,
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

        st.subheader("Correlation Matrix")

        st.dataframe(
            correlation.round(3),
            use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
        )

        st.caption(
            "Pearson correlation ranges from -1 to +1. "
            "Correlation does not establish causation."
        )


# ==================================================
# PAGE 7: ANOMALY DETECTION
# ==================================================

elif selected_page == "Anomaly Detection":

    st.subheader("Anomaly Detection Engine")

    st.write(
        "Identify unusual numerical observations "
        "using statistical and machine learning techniques."
    )

    if not types["numeric"]:

        st.warning("Numerical columns are required.")

    else:

        method = st.selectbox(
            "Choose Detection Algorithm",
            [
                "IQR Method",
                "Z-Score Method",
                "Isolation Forest"
            ]
        )

        contamination = st.slider(
            "Expected anomaly proportion (Isolation Forest)",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01
        )

        if st.button("Run Anomaly Detection"):

            with st.spinner("Analysing observations..."):

                results = generate_anomaly_summary(
                    df,
                    contamination
                )

                if method == "IQR Method":

                    flags = results["iqr"]

                    if flags.empty:

                        st.info("No usable numerical columns.")

                    else:

                        anomaly_mask = flags.any(axis=1)

                        result = df.copy()

                        result["Is Anomaly"] = anomaly_mask

                        st.session_state.anomaly_result = result

                        st.session_state.anomaly_method = method

                elif method == "Z-Score Method":

                    flags = results["zscore"]

                    if flags.empty:

                        st.info("No usable numerical columns.")

                    else:

                        anomaly_mask = flags.any(axis=1)

                        result = df.copy()

                        result["Is Anomaly"] = anomaly_mask

                        st.session_state.anomaly_result = result

                        st.session_state.anomaly_method = method

                else:

                    isolation = results["isolation_forest"]

                    if isolation is None:

                        st.warning(
                            "Isolation Forest requires at least "
                            "10 rows and usable numerical features."
                        )

                    else:

                        result = df.copy()

                        result["Is Anomaly"] = (
                            isolation["Is Anomaly"]
                        )

                        result["Anomaly Score"] = (
                            isolation["Anomaly Score"]
                        )

                        st.session_state.anomaly_result = result

                        st.session_state.anomaly_method = method

        if "anomaly_result" in st.session_state:

            result = st.session_state.anomaly_result

            st.success(
                f"Completed using {st.session_state.anomaly_method}."
            )

            anomaly_count = int(
                result["Is Anomaly"].sum()
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Detected Anomalies",
                f"{anomaly_count:,}"
            )

            c2.metric(
                "Normal Observations",
                f"{len(result) - anomaly_count:,}"
            )

            st.markdown("### Detection Results")

            st.dataframe(
                result[result["Is Anomaly"]],
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

            st.download_button(
                "Download Anomaly Results",
                data=export_csv(result),
                file_name="anomaly_detection_results.csv",
                mime="text/csv"
            )


# ==================================================
# PAGE 8: VISUALIZATION STUDIO
# ==================================================

elif selected_page == "Visualization Studio":

    st.subheader("Interactive Visualization Studio")

    st.write(
        "Build custom charts using your dataset columns."
    )

    chart_type = st.selectbox(
        "Choose chart type",
        [
            "Histogram",
            "Box Plot",
            "Bar Chart",
            "Scatter Plot",
            "Line Chart"
        ]
    )

    if chart_type in ["Histogram", "Box Plot"]:

        if not types["numeric"]:

            st.warning("Numerical columns required.")

        else:

            col = st.selectbox(
                "Select numerical column",
                types["numeric"]
            )

            if chart_type == "Histogram":

                fig = create_histogram(df, col)

            else:

                fig = create_boxplot(df, col)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

    elif chart_type == "Bar Chart":

        if not types["categorical"]:

            st.warning("Categorical columns required.")

        else:

            col = st.selectbox(
                "Select categorical column",
                types["categorical"]
            )

            st.plotly_chart(
                create_bar_chart(df, col),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

    elif chart_type == "Scatter Plot":

        if len(types["numeric"]) < 2:

            st.warning(
                "At least two numerical columns required."
            )

        else:

            x_col = st.selectbox(
                "X Axis",
                types["numeric"]
            )

            y_options = [
                col for col in types["numeric"]
                if col != x_col
            ]

            y_col = st.selectbox(
                "Y Axis",
                y_options
            )

            color_col = st.selectbox(
                "Optional colour grouping",
                ["None"] + types["categorical"]
            )

            fig = create_scatter_plot(
                df,
                x_col,
                y_col,
                None if color_col == "None" else color_col
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )

    elif chart_type == "Line Chart":

        if len(types["numeric"]) < 1:

            st.warning("Numerical column required.")

        else:

            x_options = types["datetime"] + types["numeric"]

            x_col = st.selectbox(
                "X Axis",
                x_options
            )

            y_col = st.selectbox(
                "Y Axis",
                types["numeric"]
            )

            fig = create_line_chart(
                df,
                x_col,
                y_col
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True}
            )


# ==================================================
# PAGE 9: EXPORT AND REPORTING
# ==================================================

elif selected_page == "Export & Reporting":

    st.subheader("Data Export & Reporting")

    st.write(
        "Export your dataset and generate an analytical report."
    )

    st.markdown("### Dataset Downloads")

    st.download_button(
        "Download Original Dataset as CSV",
        data=export_csv(df),
        file_name="dataset.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Dataset as Excel",
        data=export_excel(df),
        file_name="data_intelligence_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    st.markdown("### Analytical Report")

    st.download_button(
        "Download Analytical JSON Report",
        data=generate_json_report(df),
        file_name="analytical_report.json",
        mime="application/json"
    )

    st.divider()

    st.markdown("### Dataset Summary")

    st.json(profile)

    st.success(
        "Your dataset analysis and export tools are ready."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Enterprise Data Intelligence Dashboard | "
    "Day 01 of 60 | Built with Python, Streamlit, "
    "Pandas, Plotly, Scikit-learn and SQLite."
)