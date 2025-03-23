import numpy as np
import pandas as pd
import altair as alt
from sklearn.metrics import mean_squared_error, r2_score


def plot_prediction_analysis(
    df: pd.DataFrame, name: str, y_col="y", y_hat_col="y_hat", sample_size=5000
):
    """
    Generate a set of plots for analyzing predictions using Altair.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing actual values and predictions
    name : str
        Name to insert into plot titles
    y_col : str, default='y'
        Column name for actual values
    y_hat_col : str, default='y_hat'
        Column name for predicted values
    sample_size : int, default=5000
        Number of points to sample for visualization (to avoid over plotting)

    Returns:
    --------
    chart : altair.Chart
        The combined Altair chart containing all visualizations
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df[y_col], df[y_hat_col]))
    mse = mean_squared_error(df[y_col], df[y_hat_col])
    r2 = r2_score(df[y_col], df[y_hat_col])

    # Prepare data for plotting
    plot_df = df[[y_col, y_hat_col]].copy()
    plot_df.columns = ["Actual", "Predicted"]

    # Sample data if necessary to avoid overplotting
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(sample_size, random_state=42)

    # Calculate residuals
    plot_df["Residual"] = plot_df["Actual"] - plot_df["Predicted"]

    # Find min and max values for diagonal line
    min_val = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_val = max(plot_df["Actual"].max(), plot_df["Predicted"].max())
    line_df = pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]})

    # Create diagonal line for actual vs predicted charts
    diagonal_line = (
        alt.Chart(line_df)
        .mark_line(color="red", strokeDash=[4, 4])
        .encode(x="x", y="y")
    )

    # Metrics text
    metrics_text = (
        f"{name} Performance Metrics:\nRMSE: {rmse:.4f} | MSE: {mse:.4f} | R²: {r2:.4f}"
    )

    # Chart 1: Actual vs Predicted
    chart1 = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("Actual", title="Actual"),
            y=alt.Y("Predicted", title="Predicted"),
            tooltip=["Actual", "Predicted"],
        )
        .properties(
            width=400, height=300, title=f"Actual vs Predicted Values for {name}"
        )
    )

    # Add diagonal line to chart1
    chart1 = chart1 + diagonal_line

    # Chart 2: Residuals vs Predicted
    zero_line_df = pd.DataFrame(
        {"x": [plot_df["Predicted"].min(), plot_df["Predicted"].max()], "y": [0, 0]}
    )
    zero_line = (
        alt.Chart(zero_line_df)
        .mark_line(color="red", strokeDash=[4, 4])
        .encode(x="x", y="y")
    )

    chart2 = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("Predicted", title="Predicted"),
            y=alt.Y("Residual", title="Residuals"),
            tooltip=["Predicted", "Residual"],
        )
        .properties(
            width=400, height=300, title=f"Residuals vs Predicted Values for {name}"
        )
    )

    # Add zero line to chart2
    chart2 = chart2 + zero_line

    # Chart 3: Residual Distribution
    chart3 = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Residual", bin=alt.Bin(maxbins=30), title="Residual"),
            y=alt.Y("count()", title="Frequency"),
        )
        .properties(width=400, height=300, title=f"Residual Distribution for {name}")
    )

    # Chart 4: Predicted vs Actual (inverse of chart1)
    chart4 = (
        alt.Chart(plot_df)
        .mark_circle(opacity=0.5)
        .encode(
            x=alt.X("Predicted", title="Predicted"),
            y=alt.Y("Actual", title="Actual"),
            tooltip=["Predicted", "Actual"],
        )
        .properties(
            width=400, height=300, title=f"Predicted vs Actual Values for {name}"
        )
    )

    # Add diagonal line to chart4
    chart4 = chart4 + diagonal_line

    # Add metrics as a text annotation
    metrics_df = pd.DataFrame([{"text": metrics_text}])
    metrics_chart = (
        alt.Chart(metrics_df)
        .mark_text(align="center", baseline="middle", fontSize=14)
        .encode(text="text:N")
        .properties(width=800, height=50)
    )

    # Combine all charts
    top_row = alt.hconcat(chart1, chart2)
    bottom_row = alt.hconcat(chart3, chart4)
    main_chart = alt.vconcat(top_row, bottom_row)

    # Add metrics at the bottom
    final_chart = alt.vconcat(main_chart, metrics_chart)

    return final_chart
