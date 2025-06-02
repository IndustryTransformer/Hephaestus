import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_prediction_analysis(df, name, y_col="y", y_hat_col="y_hat", figsize=(15, 12)):
    """
    Generate a set of plots for analyzing predictions.

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
    figsize : tuple, default=(15, 12)
        Figure size for the plot grid

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing all subplots
    """
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df[y_col], df[y_hat_col]))
    mae = mean_absolute_error(df[y_col], df[y_hat_col])
    r2 = r2_score(df[y_col], df[y_hat_col])

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: Actual vs Predicted
    axs[0, 0].scatter(df[y_col], df[y_hat_col], alpha=0.5)
    axs[0, 0].plot(
        [df[y_col].min(), df[y_col].max()], [df[y_col].min(), df[y_col].max()], "r--"
    )
    axs[0, 0].set_xlabel("Actual")
    axs[0, 0].set_ylabel("Predicted")
    axs[0, 0].set_title(f"Actual vs Predicted Values for {name}")

    # Plot 2: Residuals
    residuals = df[y_col] - df[y_hat_col]
    axs[0, 1].scatter(df[y_hat_col], residuals, alpha=0.5)
    axs[0, 1].axhline(y=0, color="r", linestyle="--")
    axs[0, 1].set_xlabel("Predicted")
    axs[0, 1].set_ylabel("Residuals")
    axs[0, 1].set_title(f"Residuals vs Predicted Values for {name}")

    # Plot 3: Residual Distribution
    sns.histplot(residuals, kde=True, ax=axs[1, 0])
    axs[1, 0].set_xlabel("Residual")
    axs[1, 0].set_title(f"Residual Distribution for {name}")

    # Plot 4: Predicted vs Actual (with perfect fit line)
    axs[1, 1].scatter(df[y_hat_col], df[y_col], alpha=0.5)
    axs[1, 1].plot(
        [df[y_hat_col].min(), df[y_hat_col].max()],
        [df[y_hat_col].min(), df[y_hat_col].max()],
        "r--",
    )
    axs[1, 1].set_xlabel("Predicted")
    axs[1, 1].set_ylabel("Actual")
    axs[1, 1].set_title(f"Predicted vs Actual Values for {name}")

    # Add performance metrics as text
    plt.figtext(
        0.5,
        0.01,
        f"{name} Performance Metrics:\nRMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}",
        ha="center",
        bbox={"facecolor": "lightgray", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
