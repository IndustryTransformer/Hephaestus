import matplotlib.pyplot as plt

from hephaestus.analysis.analysis import DFComparison


def plot_col_error(dfs: DFComparison, col: str):
    plt.figure(figsize=(12, 6))

    # Plotting wind direction comparison
    plt.scatter(dfs.input_df[col], dfs.output_df[col], alpha=0.7)
    plt.xlabel(f"Actual {col.title()}")
    plt.ylabel("Predicted {col.title()}")
    plt.title("Actual vs Predicted")
    plt.grid(True, alpha=0.3)

    # Add a diagonal line representing perfect prediction
    min_val = min(dfs.input_df[col].min(), dfs.output_df[col].min())
    max_val = max(dfs.input_df[col].max(), dfs.output_df[col].max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_col_comparison(dfs: DFComparison, col: str):
    plt.figure(figsize=(12, 6))

    # Plotting wind direction comparison
    plt.plot(dfs.input_df[col], label=f"Actual {col.title()}")
    plt.plot(dfs.output_df[col], label=f"Predicted {col.title()}")
    plt.xlabel("Time")
    plt.ylabel(col.title())
    plt.title(f"Actual vs Predicted {col.title()}")
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.tight_layout()
    plt.show()
