"""
forecast_model.py
------------------
Trains a per-route demand (bookings) forecasting model using engineered
time features + price + promo signals, evaluates forecast accuracy on a
held-out test period (last 90 days per route), and saves an
actual-vs-predicted chart.

Model choice: Gradient Boosting Regressor. Chosen over a naive/linear
baseline because bookings respond non-linearly to price (elasticity) and
interact with seasonality - tree ensembles capture this without manual
interaction terms, while still being fast to train and easy to explain
to non-technical stakeholders (feature importances map directly to
business drivers: price, season, promo, trend).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

DATA_PATH = "/home/claude/revenue_project/data/airline_bookings.csv"
TEST_DAYS = 90


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["t_index"] = (df["date"] - df["date"].min()).dt.days
    df = pd.get_dummies(df, columns=["route"], prefix="route")
    return df


FEATURE_COLS_BASE = [
    "price", "promo_flag", "t_index",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
]


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    raw = pd.read_csv(DATA_PATH)
    df = build_features(raw)
    route_cols = [c for c in df.columns if c.startswith("route_")]
    feature_cols = FEATURE_COLS_BASE + route_cols

    df = df.sort_values("date")
    cutoff_date = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["date"] <= cutoff_date]
    test = df[df["date"] > cutoff_date]

    X_train, y_train = train[feature_cols], train["bookings"]
    X_test, y_test = test[feature_cols], test["bookings"]

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape_val = mape(y_test.values, preds)

    print("=== Forecast accuracy on last 90 days (all routes) ===")
    print(f"MAE:  {mae:.1f} bookings/day")
    print(f"RMSE: {rmse:.1f} bookings/day")
    print(f"MAPE: {mape_val:.2f}%")

    # naive baseline for comparison: same day last week
    baseline = df.set_index("date")
    naive_preds = []
    naive_actual = []
    for route_col in route_cols:
        route_mask = baseline[route_col] == 1
        route_df = baseline[route_mask].sort_index()
        shifted = route_df["bookings"].shift(7)
        test_mask = route_df.index > cutoff_date
        naive_preds.extend(shifted[test_mask].dropna().values)
        naive_actual.extend(route_df["bookings"][test_mask].values[-len(shifted[test_mask].dropna()):])
    naive_preds = np.array(naive_preds)
    naive_actual = np.array(naive_actual)
    naive_mae = mean_absolute_error(naive_actual, naive_preds)
    naive_mape = mape(naive_actual, naive_preds)

    print("\n=== Naive baseline (same day, previous week) ===")
    print(f"MAE:  {naive_mae:.1f} bookings/day")
    print(f"MAPE: {naive_mape:.2f}%")
    improvement = (1 - mae / naive_mae) * 100
    print(f"\nModel improves MAE over naive baseline by {improvement:.1f}%")

    # feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(6).round(3))

    # --- Chart: actual vs predicted for one representative route over test window ---
    focus_route_col = "route_DXB-LHR"
    mask = (test[focus_route_col] == 1)
    plot_dates = test.loc[mask, "date"]
    plot_actual = y_test[mask]
    plot_pred = preds[mask.values]

    plt.figure(figsize=(11, 5))
    plt.plot(plot_dates, plot_actual, label="Actual bookings", linewidth=2)
    plt.plot(plot_dates, plot_pred, label="Forecast (Gradient Boosting)", linewidth=2, linestyle="--")
    plt.title("DXB–LHR: Actual vs Forecast Daily Bookings (90-day holdout)")
    plt.xlabel("Date")
    plt.ylabel("Bookings")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/revenue_project/outputs/forecast_accuracy.png", dpi=150)
    print("\nSaved chart: outputs/forecast_accuracy.png")

    # save trained model artifacts needed by scenario script
    import joblib
    joblib.dump({"model": model, "feature_cols": feature_cols, "route_cols": route_cols},
                "/home/claude/revenue_project/outputs/model.pkl")


if __name__ == "__main__":
    main()
