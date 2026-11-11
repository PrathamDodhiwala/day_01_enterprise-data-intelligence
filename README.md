# Enterprise Data Intelligence Dashboard

An interactive, enterprise-style data intelligence platform built using Python, Streamlit, Pandas, Plotly, Scikit-learn, and SQLite.

The application enables users to upload structured datasets, perform automated exploratory data analysis, monitor data quality, identify statistical anomalies, explore feature relationships, and generate downloadable analytical reports.

## Project Overview

The Enterprise Data Intelligence Dashboard aims to simplify exploratory data analysis by automating repetitive analytical tasks and presenting insights through an interactive interface.

Users can upload CSV and Excel datasets and explore their structure, statistical properties, quality issues, and unusual observations without manually writing individual analysis scripts.

## Key Features

- CSV and Excel dataset uploads.
- Automatic dataset profiling and schema inspection.
- Missing-value and duplicate detection.
- Automated data cleaning.
- Descriptive and statistical analysis.
- Automatic exploratory data analysis.
- Interactive Plotly visualizations.
- Correlation heatmaps.
- IQR-based outlier detection.
- Z-score anomaly detection.
- Isolation Forest machine learning.
- SQLite database integration.
- Read-only SQL query console.
- CSV and Excel exports.
- Downloadable JSON analytical reports.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Database | SQLite |
| Query Language | SQL |
| Application | Streamlit |

## Project Architecture

- `app.py` — Main Streamlit application.
- `data_processing.py` — Data loading, cleaning and profiling.
- `database.py` — SQLite integration and SQL queries.
- `eda.py` — Statistical analysis and automated insights.
- `anomaly_detection.py` — Statistical and ML anomaly detection.
- `visualization.py` — Interactive charts.
- `reporting.py` — Analytical report generation and exports.
- `sample_dataset.csv` — Sample dataset.
- `requirements.txt` — Project dependencies.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-data-intelligence-dashboard.git
cd enterprise-data-intelligence-dashboard
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open:

`http://localhost:8501`

## Anomaly Detection Methodology

The application implements three approaches:

1. Interquartile Range (IQR): identifies observations outside statistical quartile boundaries.
2. Z-score: identifies observations based on standard deviation from the mean.
3. Isolation Forest: uses an ensemble-based unsupervised learning algorithm to identify unusual observations.

These methods provide complementary ways of examining potentially anomalous records.

## Database Integration

SQLite is used for local dataset storage.

Users can save datasets as database tables and execute read-only SQL queries through the application.

## Testing

The project includes a sample dataset for testing numerical analysis, categorical analysis, visualizations, filtering, and anomaly detection.

Additional tests cover missing values, duplicate records, invalid files, and data exports.

## Future Enhancements

- Automated date and time feature recognition.
- Advanced data drift detection.
- Custom SQL query builder.
- Automated feature engineering.
- Additional anomaly detection algorithms.
- Large dataset processing using Apache Spark.
- AI-generated natural-language analytical summaries.
- Role-based user authentication.
- Cloud database integration.
- Advanced automated report generation.

## Screenshots

Add screenshots of the following application modules:

1. Executive Overview.
2. Dataset Explorer.
3. Data Quality Monitor.
4. Exploratory Data Analysis.
5. Correlation Analysis.
6. Anomaly Detection.
7. Visualization Studio.
8. Export and Reporting.

## Author

Pratham Dodhiwala

Data Science Student | AI & Data Science Developer

## Disclaimer

Anomaly detection results are statistical indicators and should not be interpreted as definitive evidence of errors or unusual behaviour without further investigation.
