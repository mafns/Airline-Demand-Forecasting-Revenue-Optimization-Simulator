"""
generate_data.py
-----------------
Generates a synthetic but realistic daily booking/revenue dataset for a set
of long-haul routes, with:
  - an underlying growth trend
  - weekly seasonality (weekend vs weekday travel patterns)
  - annual seasonality (summer / holiday peaks)
  - route-specific price elasticity of demand
  - random promotional fare events
  - random noise

The goal is a dataset realistic enough to support genuine forecasting and
price/revenue scenario analysis, in the absence of access to a proprietary
airline booking system.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---- Route definitions ----------------------------------------------------
# elasticity: % change in demand for a 1% change in price (higher = more
# price-sensitive, e.g. leisure-heavy routes; lower = more inelastic,
# e.g. business/VFR-heavy routes)
ROUTES = {
    "DXB-LHR": {"base_demand": 320, "base_fare": 650, "elasticity": 1.1, "growth": 0.00035},
    "DXB-JFK": {"base_demand": 260, "base_fare": 980, "elasticity": 0.8,  "growth": 0.00030},
    "DXB-BOM": {"base_demand": 410, "base_fare": 210, "elasticity": 1.8,  "growth": 0.00045},
    "DXB-SYD": {"base_demand": 180, "base_fare": 1150, "elasticity": 1.3, "growth": 0.00020},
    "DXB-BKK": {"base_demand": 300, "base_fare": 340, "elasticity": 1.6,  "growth": 0.00040},
}

N_DAYS = 730  # 2 years of daily data
START_DATE = pd.Timestamp("2024-01-01")


def weekly_seasonality(day_of_week: np.ndarray) -> np.ndarray:
    """Leisure-skewed travel: demand lifts Thu-Sun."""
    pattern = {0: 0.92, 1: 0.90, 2: 0.94, 3: 1.05, 4: 1.15, 5: 1.20, 6: 1.05}
    return np.array([pattern[d] for d in day_of_week])


def annual_seasonality(day_of_year: np.ndarray) -> np.ndarray:
    """Summer peak (~day 180) and a winter holiday bump (~day 355)."""
    summer = 0.22 * np.exp(-((day_of_year - 190) ** 2) / (2 * 35 ** 2))
    winter = 0.18 * np.exp(-((day_of_year - 355) ** 2) / (2 * 15 ** 2))
    return 1.0 + summer + winter


def generate_route(route_name: str, params: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(dates)
    t = np.arange(n)
    dow = dates.dayofweek.values
    doy = dates.dayofyear.values

    trend = 1.0 + params["growth"] * t
    weekly = weekly_seasonality(dow)
    annual = annual_seasonality(doy)

    # --- Pricing process: fares drift, with periodic promotions ---
    fare_noise = np.cumsum(np.random.normal(0, 1.5, n))
    fare_drift = params["base_fare"] * (1 + 0.10 * np.sin(t / 120))
    promo_flag = (np.random.rand(n) < 0.08).astype(int)  # ~8% of days on promo
    promo_discount = np.where(promo_flag == 1, np.random.uniform(0.12, 0.30, n), 0.0)
    price = np.clip(fare_drift + fare_noise, params["base_fare"] * 0.6, params["base_fare"] * 1.6)
    price = price * (1 - promo_discount)

    # --- Demand process: base * seasonality * trend * price elasticity * noise ---
    price_ratio = price / params["base_fare"]
    elasticity_effect = price_ratio ** (-params["elasticity"])
    noise = np.random.normal(1.0, 0.06, n)

    demand = params["base_demand"] * trend * weekly * annual * elasticity_effect * noise
    demand = np.clip(demand, 20, None)
    bookings = np.round(demand).astype(int)
    revenue = bookings * price

    return pd.DataFrame({
        "date": dates,
        "route": route_name,
        "day_of_week": dow,
        "month": dates.month.values,
        "price": np.round(price, 2),
        "promo_flag": promo_flag,
        "bookings": bookings,
        "revenue": np.round(revenue, 2),
    })


def main():
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    frames = [generate_route(name, p, dates) for name, p in ROUTES.items()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["route", "date"]).reset_index(drop=True)
    out_path = "/home/claude/revenue_project/data/airline_bookings.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(df.groupby("route")[["bookings", "revenue"]].mean().round(1))


if __name__ == "__main__":
    main()
