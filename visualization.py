import plotly.express as px
import plotly.graph_objects as go

from data_processing import get_column_types

CHART_TEMPLATE = "plotly_white"

CHART_LAYOUT = {
    "template": CHART_TEMPLATE,
    "paper_bgcolor": "#FFFFFF",
    "plot_bgcolor": "#FFFFFF",
    "font": {
        "family": "Inter, Segoe UI, Arial, sans-serif",
        "color": "#26364B",
        "size": 12,
    },
    "title": {
        "font": {"size": 16, "color": "#172B4D"},
        "x": 0.02,
        "xanchor": "left",
    },
    "margin": {"l": 42, "r": 24, "t": 64, "b": 42},
    "legend": {
        "bgcolor": "rgba(255,255,255,0)",
        "font": {"color": "#526277"},
    },
    "hoverlabel": {
        "bgcolor": "#172B4D",
        "font": {"color": "#FFFFFF"},
    },
}


def _polish(fig):
    """Apply the shared visual system to every Plotly figure."""
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#EDF1F6",
        zeroline=False,
        linecolor="#DCE3ED",
        tickfont={"color": "#64748B"},
        title_font={"color": "#526277"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EDF1F6",
        zeroline=False,
        linecolor="#DCE3ED",
        tickfont={"color": "#64748B"},
        title_font={"color": "#526277"},
    )
    return fig


def create_histogram(df, column):
    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        title=f"Distribution of {column}",
        color_discrete_sequence=["#4778E8"],
        opacity=0.88,
    )
    return _polish(fig)


def create_boxplot(df, column):
    fig = px.box(
        df,
        y=column,
        points="outliers",
        title=f"Outlier Analysis: {column}",
        color_discrete_sequence=["#16A085"],
    )
    return _polish(fig)


def create_bar_chart(df, column, top_n=15):
    counts = (
        df[column]
        .astype("string")
        .fillna("(Missing)")
        .value_counts()
        .head(top_n)
        .rename_axis(column)
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Top Categories: {column}",
        color_discrete_sequence=["#4778E8"],
    )
    fig.update_traces(marker_line_width=0, marker_cornerradius=5)
    return _polish(fig)


def create_scatter_plot(df, x_col, y_col, color_col=None):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{y_col} vs {x_col}",
        color_discrete_sequence=["#4778E8", "#16A085", "#E6A23C", "#8B72D8"],
        opacity=0.78,
    )
    fig.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "#FFFFFF"}})
    return _polish(fig)


def create_line_chart(df, x_col, y_col):
    ordered = df.sort_values(x_col)
    fig = px.line(
        ordered,
        x=x_col,
        y=y_col,
        title=f"{y_col} over {x_col}",
        color_discrete_sequence=["#4778E8"],
        markers=True,
    )
    fig.update_traces(line={"width": 2.5}, marker={"size": 6})
    return _polish(fig)


def create_correlation_heatmap(correlation):
    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[
            [0.0, "#B64B55"],
            [0.5, "#F7F8FA"],
            [1.0, "#356AE6"],
        ],
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(
        coloraxis_colorbar={
            "title": "Correlation",
            "tickfont": {"color": "#64748B"},
        }
    )
    return _polish(fig)


def create_auto_visualizations(df):
    types = get_column_types(df)
    figures = []

    for col in types["numeric"][:4]:
        figures.append((f"Distribution of {col}", create_histogram(df, col)))

    for col in types["categorical"][:3]:
        if df[col].nunique(dropna=True) <= 50:
            figures.append((f"Category Distribution: {col}", create_bar_chart(df, col)))

    if len(types["numeric"]) >= 2:
        figures.append((
            "Numerical Relationship",
            create_scatter_plot(df, types["numeric"][0], types["numeric"][1])
        ))

    return figures
