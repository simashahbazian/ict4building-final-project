
Energy Signature & Prediction Analysis
======================================

This script contains the energy-analysis contribution developed for the
ICT4Building project.

Analyses included:
1. Energy Signature analysis using ΔT
2. Hourly, daily, and weekly aggregation
3. Before vs. after optimization comparison
4. Linear regression and R² evaluation
5. Multivariate cooling analysis using ΔT and GHI
6. Multivariate LSTM prediction of heating consumption

Input datasets:
- data/energy_signature_old_with_ghi.csv
- data/energy_signature_opt_with_ghi.csv

Expected columns:
- time
- delta_T
- outdoor_temp
- GHI
- heating_kWh
- cooling_kWh

# =============================================================================
# 1. Imports
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# 2. Configuration
# =============================================================================

DATA_DIR = Path("data")

OLD_FILE = DATA_DIR / "energy_signature_old_with_ghi.csv"
OPT_FILE = DATA_DIR / "energy_signature_opt_with_ghi.csv"

LSTM_WINDOW_SIZE = 24
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 16


# =============================================================================
# 3. Data Loading
# =============================================================================

def load_energy_data(file_path):
    """
    Load an energy dataset and prepare the time column.
    """

    df = pd.read_csv(
        file_path,
        parse_dates=["time"]
    )

    df = df.sort_values("time").reset_index(drop=True)

    return df


# =============================================================================
# 4. Basic Energy Signature Plot
# =============================================================================

def plot_raw_energy_signature(df, title):
    """
    Plot heating and cooling consumption against ΔT
    using the original hourly observations.
    """

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["delta_T"],
        df["heating_kWh"],
        label="Heating",
        alpha=0.6
    )

    plt.scatter(
        df["delta_T"],
        df["cooling_kWh"],
        label="Cooling",
        alpha=0.6
    )

    plt.xlabel(
        "Delta T (Indoor - Outdoor) [°C]"
    )

    plt.ylabel(
        "Energy Consumption (kWh)"
    )

    plt.title(title)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 5. Season Classification
# =============================================================================

def get_season(month):
    """
    Classify observations into Winter, Summer, or Other.
    """

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [6, 7, 8]:
        return "Summer"

    return "Other"


# =============================================================================
# 6. Time Aggregation
# =============================================================================

def aggregate_energy_data(df, frequency):
    """
    Aggregate energy data by hourly, daily, or weekly periods.

    Parameters
    ----------
    df : pandas.DataFrame
        Input energy dataset.

    frequency : str
        One of:
        - "hourly"
        - "daily"
        - "weekly"
    """

    data = df.copy()

    data["time"] = pd.to_datetime(
        data["time"]
    )

    if frequency == "hourly":

        data["period"] = (
            data["time"].dt.floor("h")
        )

    elif frequency == "daily":

        data["period"] = (
            data["time"].dt.floor("D")
        )

    elif frequency == "weekly":

        data["period"] = (
            data["time"]
            .dt.to_period("W")
            .apply(lambda r: r.start_time)
        )

    else:

        raise ValueError(
            "frequency must be 'hourly', 'daily', or 'weekly'"
        )

    data["month"] = (
        data["time"].dt.month
    )

    aggregated = (
        data
        .groupby("period")
        .agg(
            {
                "delta_T": "mean",
                "heating_kWh": "mean",
                "cooling_kWh": "mean",
                "month": "first",
            }
        )
        .reset_index()
    )

    aggregated["season"] = (
        aggregated["month"].apply(get_season)
    )

    return aggregated


# =============================================================================
# 7. Energy Signature Regression
# =============================================================================

def calculate_energy_signature(
    df,
    frequency="daily",
    filter_cooling_outliers=True,
):
    """
    Calculate heating and cooling energy signatures.

    Heating:
        Winter energy consumption vs. ΔT

    Cooling:
        Summer energy consumption vs. ΔT

    Returns regression models, predictions, R² values,
    and processed seasonal datasets.
    """

    aggregated = aggregate_energy_data(
        df,
        frequency
    )

    winter = aggregated[
        aggregated["season"] == "Winter"
    ].copy()

    summer = aggregated[
        aggregated["season"] == "Summer"
    ].copy()

    # -------------------------------------------------------------------------
    # Cooling filtering used in the original daily/hourly analysis
    # -------------------------------------------------------------------------

    if filter_cooling_outliers:

        summer = summer[
            ~(
                (summer["delta_T"] > -2)
                &
                (summer["cooling_kWh"] > 7)
            )
        ].copy()

    # -------------------------------------------------------------------------
    # Remove missing values before regression
    # -------------------------------------------------------------------------

    winter = winter.dropna(
        subset=[
            "delta_T",
            "heating_kWh"
        ]
    )

    summer = summer.dropna(
        subset=[
            "delta_T",
            "cooling_kWh"
        ]
    )

    # -------------------------------------------------------------------------
    # Heating regression
    # -------------------------------------------------------------------------

    X_w = winter[
        ["delta_T"]
    ].values

    y_w = winter[
        "heating_kWh"
    ].values

    heating_model = LinearRegression()

    heating_model.fit(
        X_w,
        y_w
    )

    heating_prediction = (
        heating_model.predict(X_w)
    )

    heating_r2 = r2_score(
        y_w,
        heating_prediction
    )

    # -------------------------------------------------------------------------
    # Cooling regression
    # -------------------------------------------------------------------------

    X_s = summer[
        ["delta_T"]
    ].values

    y_s = summer[
        "cooling_kWh"
    ].values

    cooling_model = LinearRegression()

    cooling_model.fit(
        X_s,
        y_s
    )

    cooling_prediction = (
        cooling_model.predict(X_s)
    )

    cooling_r2 = r2_score(
        y_s,
        cooling_prediction
    )

    return {
        "aggregated": aggregated,
        "winter": winter,
        "summer": summer,
        "heating_model": heating_model,
        "cooling_model": cooling_model,
        "heating_prediction": heating_prediction,
        "cooling_prediction": cooling_prediction,
        "heating_r2": heating_r2,
        "cooling_r2": cooling_r2,
    }


# =============================================================================
# 8. Plot Energy Signature
# =============================================================================

def plot_energy_signature(
    results,
    frequency,
    optimization_label,
):
    """
    Plot heating and cooling energy signatures together.
    """

    winter = results["winter"]
    summer = results["summer"]

    heating_prediction = (
        results["heating_prediction"]
    )

    cooling_prediction = (
        results["cooling_prediction"]
    )

    # Sort points for a cleaner regression line
    winter_order = np.argsort(
        winter["delta_T"].values
    )

    summer_order = np.argsort(
        summer["delta_T"].values
    )

    plt.figure(figsize=(10, 6))

    # Winter / Heating
    plt.scatter(
        winter["delta_T"],
        winter["heating_kWh"],
        alpha=0.5,
        label="Winter Data (Heating)"
    )

    plt.plot(
        winter["delta_T"].values[winter_order],
        heating_prediction[winter_order],
        label="Winter Regression"
    )

    # Summer / Cooling
    plt.scatter(
        summer["delta_T"],
        summer["cooling_kWh"],
        alpha=0.5,
        label="Summer Data (Cooling)"
    )

    plt.plot(
        summer["delta_T"].values[summer_order],
        cooling_prediction[summer_order],
        label="Summer Regression"
    )

    plt.xlabel(
        "Delta T (Indoor - Outdoor) [°C]"
    )

    plt.ylabel(
        "Energy Consumption (kWh)"
    )

    plt.title(
        f"{frequency.capitalize()} Energy Signature - "
        f"Heating & Cooling ({optimization_label})"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 9. Run Energy Signature Analysis
# =============================================================================

def run_energy_signature_analysis(
    df,
    frequency,
    optimization_label,
    filter_cooling_outliers=True,
):
    """
    Run the complete energy signature analysis for one dataset.
    """

    results = calculate_energy_signature(
        df,
        frequency=frequency,
        filter_cooling_outliers=filter_cooling_outliers,
    )

    plot_energy_signature(
        results,
        frequency,
        optimization_label,
    )

    print(
        f"\n{frequency.capitalize()} Energy Signature - "
        f"{optimization_label}"
    )

    print(
        f"Heating R²: "
        f"{results['heating_r2']:.3f}"
    )

    print(
        f"Cooling R²: "
        f"{results['cooling_r2']:.3f}"
    )

    return results


# =============================================================================
# 10. Before vs. After Optimization
# =============================================================================

def compare_before_after(
    old_df,
    optimized_df,
    frequency,
):
    """
    Compare energy signature results before and after optimization.
    """

    print("\n" + "=" * 70)

    print(
        f"{frequency.upper()} ENERGY SIGNATURE "
        "COMPARISON"
    )

    print("=" * 70)

    # -------------------------------------------------------------------------
    # Before optimization
    # -------------------------------------------------------------------------

    print(
        "\nBefore Optimization"
    )

    old_results = run_energy_signature_analysis(
        old_df,
        frequency=frequency,
        optimization_label="Before Optimization",
        filter_cooling_outliers=(
            frequency != "weekly"
        ),
    )

    # -------------------------------------------------------------------------
    # After optimization
    # -------------------------------------------------------------------------

    print(
        "\nAfter Optimization"
    )

    optimized_results = run_energy_signature_analysis(
        optimized_df,
        frequency=frequency,
        optimization_label="After Optimization",
        filter_cooling_outliers=(
            frequency != "weekly"
        ),
    )

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    print(
        "\nR² Comparison"
    )

    print("-" * 70)

    print(
        f"Heating: "
        f"{old_results['heating_r2']:.3f} "
        f"→ "
        f"{optimized_results['heating_r2']:.3f}"
    )

    print(
        f"Cooling: "
        f"{old_results['cooling_r2']:.3f} "
        f"→ "
        f"{optimized_results['cooling_r2']:.3f}"
    )

    return (
        old_results,
        optimized_results
    )


# =============================================================================
# 11. Multivariate Cooling Analysis
# =============================================================================

def prepare_cooling_data(df):
    """
    Select positive cooling observations during warm months.

    Months:
        June, July, August, September
    """

    data = df.copy()

    data["time"] = pd.to_datetime(
        data["time"]
    )

    cooling_df = data[
        (
            data["time"].dt.month.isin(
                [6, 7, 8, 9]
            )
        )
        &
        (
            data["cooling_kWh"] > 0
        )
    ].copy()

    cooling_df = cooling_df.dropna(
        subset=[
            "delta_T",
            "GHI",
            "cooling_kWh",
        ]
    )

    return cooling_df


def run_multivariate_cooling_analysis(
    df,
):
    """
    Analyze cooling consumption using:
        - ΔT
        - GHI

    Includes:
        - 3D scatter plot
        - multiple linear regression
        - regression surface
    """

    cooling_df = (
        prepare_cooling_data(df)
    )

    X = cooling_df[
        [
            "delta_T",
            "GHI",
        ]
    ].values

    y = cooling_df[
        "cooling_kWh"
    ].values

    # -------------------------------------------------------------------------
    # Multiple linear regression
    # -------------------------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X,
        y
    )

    predictions = model.predict(
        X
    )

    r2 = r2_score(
        y,
        predictions
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTIVARIATE COOLING ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"Intercept: "
        f"{model.intercept_:.4f}"
    )

    print(
        f"Delta T coefficient: "
        f"{model.coef_[0]:.4f}"
    )

    print(
        f"GHI coefficient: "
        f"{model.coef_[1]:.4f}"
    )

    print(
        f"R²: "
        f"{r2:.3f}"
    )

    # -------------------------------------------------------------------------
    # 3D scatter
    # -------------------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    scatter = ax.scatter(
        cooling_df["delta_T"],
        cooling_df["GHI"],
        cooling_df["cooling_kWh"],
        c=cooling_df["cooling_kWh"],
        alpha=0.7,
    )

    ax.set_xlabel(
        "Delta T (Indoor - Outdoor) [°C]"
    )

    ax.set_ylabel(
        "GHI (W/m²)"
    )

    ax.set_zlabel(
        "Cooling Energy Consumption (kWh)"
    )

    ax.set_title(
        "3D Plot - Delta T vs GHI vs Cooling Consumption"
    )

    fig.colorbar(
        scatter,
        label="Cooling (kWh)"
    )

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Regression surface
    # -------------------------------------------------------------------------

    x_surface, y_surface = np.meshgrid(
        np.linspace(
            X[:, 0].min(),
            X[:, 0].max(),
            20,
        ),
        np.linspace(
            X[:, 1].min(),
            X[:, 1].max(),
            20,
        ),
    )

    z_surface = model.predict(
        np.c_[
            x_surface.ravel(),
            y_surface.ravel(),
        ]
    ).reshape(
        x_surface.shape
    )

    def plot_3d(
        elevation,
        azimuth,
        title,
    ):
        fig = plt.figure(
            figsize=(8, 6)
        )

        ax = fig.add_subplot(
            111,
            projection="3d"
        )

        ax.scatter(
            X[:, 0],
            X[:, 1],
            y,
            alpha=0.5,
            label="Cooling Data",
        )

        ax.plot_surface(
            x_surface,
            y_surface,
            z_surface,
            alpha=0.3,
        )

        ax.set_xlabel(
            "Delta T (Indoor - Outdoor) [°C]"
        )

        ax.set_ylabel(
            "GHI (W/m²)"
        )

        ax.set_zlabel(
            "Cooling Energy (kWh)"
        )

        ax.set_title(title)

        ax.view_init(
            elev=elevation,
            azim=azimuth,
        )

        plt.tight_layout()
        plt.show()

    # Same views used in the original analysis
    plot_3d(
        30,
        45,
        "3D View - Cooling After Optimization",
    )

    plot_3d(
        10,
        0,
        "Front View - GHI vs Cooling",
    )

    return {
        "data": cooling_df,
        "model": model,
        "predictions": predictions,
        "r2": r2,
    }


# =============================================================================
# 12. LSTM Data Preparation
# =============================================================================

def prepare_lstm_data(
    df,
):
    """
    Prepare winter data for multivariate LSTM.

    Months:
        November, December, January, February

    Features:
        - outdoor_temp
        - GHI
        - hour

    Target:
        - heating_kWh
    """

    data = df.copy()

    data["time"] = pd.to_datetime(
        data["time"]
    )

    data = data[
        data["time"].dt.month.isin(
            [11, 12, 1, 2]
        )
    ].copy()

    # Clock feature
    data["hour"] = (
        data["time"].dt.hour
    )

    # Required columns
    data = data[
        [
            "time",
            "outdoor_temp",
            "GHI",
            "hour",
            "heating_kWh",
        ]
    ]

    data = data.dropna()

    data = data.set_index(
        "time"
    )

    return data


# =============================================================================
# 13. Create LSTM Sequences
# =============================================================================

def create_multivariate_sequences(
    data,
    target_col,
    window_size=24,
):
    """
    Create rolling sequences for the LSTM.

    The target is the observation immediately
    following each 24-hour input window.
    """

    X = []
    y = []

    feature_columns = [
        column
        for column in data.columns
        if column != target_col
    ]

    for i in range(
        len(data) - window_size
    ):

        X.append(
            data.iloc[
                i:i + window_size
            ][feature_columns].values
        )

        y.append(
            data.iloc[
                i + window_size
            ][target_col]
        )

    return (
        np.array(X),
        np.array(y),
    )


# =============================================================================
# 14. LSTM Analysis
# =============================================================================

def run_lstm_analysis(
    df,
    optimization_label,
    window_size=24,
    epochs=20,
    batch_size=16,
):
    """
    Train and evaluate the multivariate LSTM.

    Features:
        outdoor_temp
        GHI
        hour

    Target:
        heating_kWh

    Important:
    This follows the original project workflow, where the model is
    fitted and predictions are generated on the same sequences.
    Therefore the reported MAE, MSE, and R² are in-sample metrics,
    not test-set performance on unseen data.
    """

    # TensorFlow is imported only when LSTM analysis is requested.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    data = prepare_lstm_data(
        df
    )

    # -------------------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------------------

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        data
    )

    scaled_df = pd.DataFrame(
        scaled,
        index=data.index,
        columns=data.columns,
    )

    # -------------------------------------------------------------------------
    # Create sequences
    # -------------------------------------------------------------------------

    X, y = create_multivariate_sequences(
        scaled_df,
        target_col="heating_kWh",
        window_size=window_size,
    )

    if len(X) == 0:

        raise ValueError(
            "Not enough observations to create LSTM sequences."
        )

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    model = Sequential()

    model.add(
        LSTM(
            64,
            activation="relu",
            input_shape=(
                X.shape[1],
                X.shape[2],
            ),
        )
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mse",
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------------------

    y_pred = model.predict(
        X,
        verbose=0,
    )

    # -------------------------------------------------------------------------
    # Restore original scale
    # -------------------------------------------------------------------------

    heating_index = list(
        data.columns
    ).index(
        "heating_kWh"
    )

    dummy_prediction = np.zeros(
        (
            len(y_pred),
            scaled.shape[1],
        )
    )

    dummy_prediction[
        :,
        heating_index
    ] = y_pred[:, 0]

    y_pred_rescaled = (
        scaler.inverse_transform(
            dummy_prediction
        )[:, heating_index]
    )

    dummy_y = np.zeros(
        (
            len(y),
            scaled.shape[1],
        )
    )

    dummy_y[
        :,
        heating_index
    ] = y

    y_true_rescaled = (
        scaler.inverse_transform(
            dummy_y
        )[:, heating_index]
    )

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    mae = mean_absolute_error(
        y_true_rescaled,
        y_pred_rescaled,
    )

    mse = mean_squared_error(
        y_true_rescaled,
        y_pred_rescaled,
    )

    r2 = r2_score(
        y_true_rescaled,
        y_pred_rescaled,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"MULTIVARIATE LSTM - "
        f"{optimization_label.upper()}"
    )

    print(
        "=" * 70
    )

    print(
        f"MAE: {mae:.4f}"
    )

    print(
        f"MSE: {mse:.4f}"
    )

    print(
        f"R²: {r2:.4f}"
    )

    # -------------------------------------------------------------------------
    # Prediction visualization
    # -------------------------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        y_true_rescaled[:100],
        label="Actual",
    )

    plt.plot(
        y_pred_rescaled[:100],
        label="Predicted",
        linestyle="--",
    )

    plt.title(
        "Multivariate LSTM Prediction "
        f"of Heating_kWh - {optimization_label}"
    )

    plt.xlabel(
        "Time Step"
    )

    plt.ylabel(
        "Heating (kWh)"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "model": model,
        "scaler": scaler,
        "X": X,
        "y": y,
        "y_true": y_true_rescaled,
        "y_pred": y_pred_rescaled,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }


# =============================================================================
# 15. Main Execution
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------------

    old_df = load_energy_data(
        OLD_FILE
    )

    optimized_df = load_energy_data(
        OPT_FILE
    )

    # -------------------------------------------------------------------------
    # Basic raw energy signature
    # -------------------------------------------------------------------------

    plot_raw_energy_signature(
        old_df,
        "Energy Signature - Heating & Cooling Real Data"
    )

    # -------------------------------------------------------------------------
    # Hourly Energy Signature
    # -------------------------------------------------------------------------

    old_hourly, optimized_hourly = (
        compare_before_after(
            old_df,
            optimized_df,
            frequency="hourly",
        )
    )

    # -------------------------------------------------------------------------
    # Daily Energy Signature
    # -------------------------------------------------------------------------

    old_daily, optimized_daily = (
        compare_before_after(
            old_df,
            optimized_df,
            frequency="daily",
        )
    )

    # -------------------------------------------------------------------------
    # Weekly Energy Signature
    # -------------------------------------------------------------------------

    old_weekly, optimized_weekly = (
        compare_before_after(
            old_df,
            optimized_df,
            frequency="weekly",
        )
    )

    # -------------------------------------------------------------------------
    # Multivariate Cooling Analysis
    # -------------------------------------------------------------------------

    cooling_results = (
        run_multivariate_cooling_analysis(
            optimized_df
        )
    )

    # -------------------------------------------------------------------------
    # Multivariate LSTM - Before Optimization
    # -------------------------------------------------------------------------

    lstm_before = run_lstm_analysis(
        old_df,
        optimization_label="Before Optimization",
        window_size=LSTM_WINDOW_SIZE,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
    )

    # -------------------------------------------------------------------------
    # Multivariate LSTM - After Optimization
    # -------------------------------------------------------------------------

    lstm_after = run_lstm_analysis(
        optimized_df,
        optimization_label="After Optimization",
        window_size=LSTM_WINDOW_SIZE,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
    )
