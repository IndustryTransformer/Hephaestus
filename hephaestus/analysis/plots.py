import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from hephaestus.analysis.analysis import (
    AutoRegressiveResults,
    auto_regressive_predictions,
    create_test_inputs_df,
    plot_column_variants,
)


def plot_training_history(history, save_dir=None):
    """Plot training and validation loss history.

    Args:
        history: Dictionary containing loss history
        save_dir: Optional directory to save the plot
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_numeric_loss"], label="Train Numeric")
    plt.plot(history["train_categorical_loss"], label="Train Categorical")
    plt.plot(history["val_numeric_loss"], label="Val Numeric")
    plt.plot(history["val_categorical_loss"], label="Val Categorical")
    plt.title("Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "training_history.png"))

    plt.show()


def plot_planet_positions(pred_df, actual_df, cols_to_plot, save_dir=None):
    """Plot comparisons for planet positions.

    Args:
        pred_df: DataFrame containing predicted values
        actual_df: DataFrame containing actual values
        cols_to_plot: List of columns to plot
        save_dir: Optional directory to save the plots
    """
    for col in cols_to_plot:
        plot_column_variants(pred_df, actual_df, col)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"prediction_{col}.png"))


def plot_position_error_over_time(planet_errors, save_dir=None):
    """Plot position error over time for each planet.

    Args:
        planet_errors: Dictionary mapping planet IDs to error arrays
        save_dir: Optional directory to save the plot
    """
    plt.figure(figsize=(15, 8))
    for planet_id, errors in planet_errors.items():
        plt.plot(errors, label=f"{planet_id}")

    plt.title("Position Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Euclidean Distance Error")
    plt.axvline(x=10, color="r", linestyle="--", label="Prediction Start")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "position_error_over_time.png"))

    plt.show()


def plot_planet_trajectory(pred_df, actual_df, planet_id, save_dir=None):
    """Plot trajectory comparison for a single planet.

    Args:
        pred_df: DataFrame containing predicted values
        actual_df: DataFrame containing actual values
        planet_id: ID of the planet to plot
        save_dir: Optional directory to save the plot
    """
    x_col = f"{planet_id}_x"
    y_col = f"{planet_id}_y"

    plt.figure(figsize=(10, 10))

    # Plot predicted trajectory
    plt.plot(
        pred_df[x_col],
        pred_df[y_col],
        "b-",
        label="Predicted Trajectory",
    )

    # Mark prediction start point
    plt.plot(
        pred_df[x_col][10],
        pred_df[y_col][10],
        "bo",
        markersize=8,
        label="Prediction Start",
    )

    # Plot actual trajectory
    plt.plot(
        actual_df[x_col],
        actual_df[y_col],
        "r--",
        label="Actual Trajectory",
    )

    plt.title(f"{planet_id} Trajectory Comparison")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")  # Equal scaling for x and y
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{planet_id}_trajectory.png"))

    plt.show()


def plot_final_system_state(pred_df, actual_df, planet_ids, save_dir=None):
    """Plot overall system state comparison at final step.

    Args:
        pred_df: DataFrame containing predicted values
        actual_df: DataFrame containing actual values
        planet_ids: List of planet IDs to plot
        save_dir: Optional directory to save the plot
    """
    if len(pred_df) > 0 and len(actual_df) > 0:
        plt.figure(figsize=(12, 10))

        # Get the final position of each planet
        final_idx = min(len(pred_df) - 1, len(actual_df) - 1)

        for planet_id in planet_ids:
            x_col = f"{planet_id}_x"
            y_col = f"{planet_id}_y"

            if x_col in pred_df.columns and y_col in pred_df.columns:
                # Plot predicted final position
                plt.plot(
                    pred_df[x_col][final_idx],
                    pred_df[y_col][final_idx],
                    "bo",
                    markersize=8,
                )

                # Plot actual final position
                plt.plot(
                    actual_df[x_col][final_idx],
                    actual_df[y_col][final_idx],
                    "ro",
                    markersize=8,
                )

                # Connect predicted and actual with a line
                plt.plot(
                    [
                        pred_df[x_col][final_idx],
                        actual_df[x_col][final_idx],
                    ],
                    [
                        pred_df[y_col][final_idx],
                        actual_df[y_col][final_idx],
                    ],
                    "k--",
                    alpha=0.5,
                )

                # Add planet label
                plt.text(
                    pred_df[x_col][final_idx],
                    pred_df[y_col][final_idx],
                    planet_id,
                    fontsize=10,
                )

        plt.title("Final System State Comparison")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)

        # Add legend for the colors
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="b",
                markersize=8,
                label="Predicted Position",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="r",
                markersize=8,
                label="Actual Position",
            ),
        ]
        plt.legend(handles=legend_elements)

        plt.axis("equal")  # Equal scaling for x and y
        plt.tight_layout()

        if save_dir:
            plt.savefig(os.path.join(save_dir, "final_system_state.png"))

        plt.show()


def evaluate_planetary_predictions(
    model,
    test_ds,
    time_series_config,
    test_sample_idx=0,
    n_steps=20,
    save_dir=None,
    device="cpu",
):
    """Evaluate model predictions on planetary data and generate plots.

    Args:
        model: Trained model
        test_ds: Test dataset
        time_series_config: Time series configuration
        test_sample_idx: Index of test sample to use
        n_steps: Number of prediction steps to generate
        save_dir: Directory to save plots
        device: Computation device (cuda/mps/cpu)

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("\n=== Evaluating Model Predictions on Sample Data ===")
    model.eval()

    results = {}

    try:
        # Create auto-regressive inputs from the first 10 time steps
        print("\nCreating auto-regressive predictions...")
        test_inputs = AutoRegressiveResults.from_ds(
            test_ds, test_sample_idx, stop_idx=10
        )

        # Ensure the inputs are on the correct device
        test_inputs.numeric_inputs = test_inputs.numeric_inputs.to(device)
        test_inputs.categorical_inputs = test_inputs.categorical_inputs.to(device)

        # Generate predictions for next n_steps steps
        print(f"Generating predictions for next {n_steps} steps...")
        model_predictions = test_inputs
        for i in range(n_steps):
            model_predictions = auto_regressive_predictions(model, model_predictions)
            if i % 5 == 0:
                print(f"  Step {i + 1} completed")

        # Move predictions back to CPU for analysis
        model_predictions.numeric_inputs = model_predictions.numeric_inputs.cpu()
        model_predictions.categorical_inputs = (
            model_predictions.categorical_inputs.cpu()
        )

        # Create DataFrames for predicted and actual data
        print("\nProcessing results...")
        pred_df = create_test_inputs_df(model_predictions, time_series_config)

        # Get actual data for comparison
        actuals = test_ds[test_sample_idx]
        actual_inputs = AutoRegressiveResults(
            torch.tensor(actuals.numeric), torch.tensor(actuals.categorical)
        )
        actual_df = create_test_inputs_df(actual_inputs, time_series_config)

        # Compare predictions with actual data for key numeric columns
        print("\n=== Comparison of Predicted vs Actual Values ===")

        # Find planets position columns (numeric columns with 'planet' and 'x' or 'y')
        planet_cols = [
            col
            for col in time_series_config.numeric_col_tokens
            if "planet" in col and ("_x" in col or "_y" in col)
        ]

        # Calculate Mean Absolute Error for each planet position
        mae_results = {}
        for col in planet_cols:
            # Calculate MAE for the predicted steps
            overlap_len = min(len(pred_df), len(actual_df))
            if overlap_len > 10:  # Skip the first 10 steps used as input
                mae = np.abs(
                    pred_df[col][10:overlap_len].values
                    - actual_df[col][10:overlap_len].values
                ).mean()
                mae_results[col] = mae

        # Print MAE results for planet positions
        print("\nMean Absolute Error for Planet Positions:")
        for col, mae in mae_results.items():
            print(f"{col}: {mae:.6f}")

        results["mae_results"] = mae_results

        # Plot comparisons for the first 4 planet positions
        cols_to_plot = planet_cols[: min(4, len(planet_cols))]
        print("\nGenerating plots for planet positions...")
        plot_planet_positions(pred_df, actual_df, cols_to_plot, save_dir)

        # Calculate position error over time
        if len(planet_cols) >= 2:
            print("\nCalculating position error over time...")
            planet_ids = set()

            # Extract planet IDs from column names
            for col in planet_cols:
                if "_x" in col:
                    planet_id = col.split("_")[0]  # Extract "planet0", "planet1", etc.
                    planet_ids.add(planet_id)

            # For each planet, calculate Euclidean distance error
            planet_errors = {}
            for planet_id in planet_ids:
                x_col = f"{planet_id}_x"
                y_col = f"{planet_id}_y"

                if x_col in pred_df.columns and y_col in pred_df.columns:
                    # Calculate Euclidean distance error for each time step
                    errors = np.sqrt(
                        (pred_df[x_col] - actual_df[x_col]) ** 2
                        + (pred_df[y_col] - actual_df[y_col]) ** 2
                    )

                    planet_errors[planet_id] = errors

            # Plot position error over time for each planet
            plot_position_error_over_time(planet_errors, save_dir)

            # Print average position error for each planet (excluding input period)
            print("\nAverage Position Error (excluding input period):")
            avg_errors = {}
            for planet_id, errors in planet_errors.items():
                if len(errors) > 10:
                    avg_error = errors[10:].mean()
                    avg_errors[planet_id] = avg_error
                    print(f"{planet_id}: {avg_error:.6f}")

            results["avg_position_errors"] = avg_errors

            # Generate planetary trajectory comparison plots
            print("\nGenerating planetary trajectory comparison plots...")
            for planet_id in list(planet_ids)[
                :2
            ]:  # Limit to first 2 planets for clarity
                plot_planet_trajectory(pred_df, actual_df, planet_id, save_dir)

            # Plot final system state
            plot_final_system_state(pred_df, actual_df, planet_ids, save_dir)

        return results, pred_df, actual_df

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        print("\nSkipping detailed evaluation due to error.")
        return {"error": str(e)}, None, None
