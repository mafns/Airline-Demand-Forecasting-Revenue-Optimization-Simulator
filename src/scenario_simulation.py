"""
scenario_simulation.py
-----------------------
Uses the trained forecasting model to run business "what-if" scenarios:

  1. PRICE OPTIMIZATION CURVE
     For each route, sweep fare changes from -25% to +25% and predict the
     resulting demand and revenue, holding all other conditions (season,
     day-of-week, promo) at a representative future date. Demand response to
     price is modelled as: demand(price) = baseline_demand * (price/ref)^(-e),
     where baseline_demand comes from the GBM forecaster (season/trend/promo)
     and e (elasticity) is estimated per route with a log-log regression on
     historical price/booking data. This separates "how many people fly
     regardless of price" (ML forecast) from "how price-sensitive are they"
     (econometric elasticity) -- a standard revenue-management decomposition,
     and it keeps the price-response curve smooth and economically sane
     (tree ensembles alone are noisy/non-monotonic near the training price
     range, which isn't appropriate for pricing decisions).

  2. DEMAND SHOCK CONTINGENCY
     Simulates an unplanned demand shock (e.g. -25% demand from a disruption
     event) on a route, then evaluates a proposed mitigating fare action
     (a temporary fare cut) and quantifies the modelled revenue recovery,
     supporting "react to unanticipated business conditions" style decisions.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ARTIFACTS = joblib.load("/home/claude/revenue_project/outputs/model.pkl")
MODEL = ARTIFACTS["model"]
FEATURE_COLS = ARTIFACTS["feature_cols"]
ROUTE_COLS = ARTIFACTS["route_cols"]

RAW = pd.read_csv("/home/claude/revenue_project/data/airline_bookings.csv")
RAW["date"] = pd.to_datetime(RAW["date"])


def make_feature_row(route: str, date: pd.Timestamp, price: float, promo_flag: int, t_index: int) -> pd.DataFrame:
    doy = date.dayofyear
    dow = date.dayofweek
    row = {
        "price": price,
        "promo_flag": promo_flag,
        "t_index": t_index,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }
    for rc in ROUTE_COLS:
        row[rc] = 1 if rc == f"route_{route}" else 0
    return pd.DataFrame([row])[FEATURE_COLS]


def current_reference_price(route: str) -> float:
    sub = RAW[RAW["route"] == route].sort_values("date")
    return sub["price"].tail(30).mean()


def t_index_for(date: pd.Timestamp) -> int:
    min_date = RAW["date"].min()
    return (date - min_date).days


def estimate_price_elasticity(route: str) -> float:
    """
    Log-log regression of bookings on price, controlling for weekly and
    monthly seasonal dummies and a linear trend, isolates the price
    coefficient from confounding seasonal demand swings. The negative of
    the price coefficient is the price elasticity of demand.
    """
    sub = RAW[RAW["route"] == route].copy().sort_values("date").reset_index(drop=True)
    sub["log_price"] = np.log(sub["price"])
    sub["log_bookings"] = np.log(sub["bookings"])
    sub["t"] = np.arange(len(sub))
    dow_dummies = pd.get_dummies(sub["date"].dt.dayofweek, prefix="dow", drop_first=True)
    month_dummies = pd.get_dummies(sub["date"].dt.month, prefix="month", drop_first=True)

    X = pd.concat([sub[["log_price", "t"]], dow_dummies, month_dummies], axis=1)
    y = sub["log_bookings"]

    reg = LinearRegression().fit(X, y)
    price_coef = reg.coef_[list(X.columns).index("log_price")]
    elasticity = -price_coef
    return max(elasticity, 0.05)  # guard against pathological negative/zero estimates


def price_optimization_curve(route: str, target_date: pd.Timestamp):
    ref_price = current_reference_price(route)
    elasticity = estimate_price_elasticity(route)

    # baseline demand at the reference price, from the ML forecaster
    baseline_feat = make_feature_row(route, target_date, ref_price, 0, t_index_for(target_date))
    baseline_demand = MODEL.predict(baseline_feat)[0]

    pct_changes = np.arange(-0.25, 0.26, 0.02)
    rows = []
    for pct in pct_changes:
        price = ref_price * (1 + pct)
        demand = baseline_demand * (price / ref_price) ** (-elasticity)
        revenue = demand * price
        rows.append({"pct_change": pct, "price": price, "predicted_demand": demand, "predicted_revenue": revenue})
    df = pd.DataFrame(rows)
    best = df.loc[df["predicted_revenue"].idxmax()]
    return df, ref_price, best, elasticity


# True elasticities used in the synthetic data generator, kept here only to
# validate the estimation approach against a known ground truth -- in a real
# deployment this would instead be validated against held-out A/B fare tests.
TRUE_ELASTICITY = {
    "DXB-LHR": 1.1, "DXB-JFK": 0.8, "DXB-BOM": 1.8, "DXB-SYD": 1.3, "DXB-BKK": 1.6,
}


def plot_price_curves(target_date: pd.Timestamp, routes: list):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = []
    for route in routes:
        df, ref_price, best, elasticity = price_optimization_curve(route, target_date)
        axes[0].plot(df["pct_change"] * 100, df["predicted_demand"], label=route)
        axes[1].plot(df["pct_change"] * 100, df["predicted_revenue"], label=route)
        axes[1].scatter([best["pct_change"] * 100], [best["predicted_revenue"]], zorder=5)
        summary.append({
            "route": route,
            "reference_fare": round(ref_price, 0),
            "est_elasticity": round(elasticity, 2),
            "true_elasticity": TRUE_ELASTICITY[route],
            "optimal_fare_change_%": round(best["pct_change"] * 100, 1),
            "optimal_fare": round(best["price"], 0),
            "predicted_demand_at_optimum": round(best["predicted_demand"], 0),
            "predicted_daily_revenue_at_optimum": round(best["predicted_revenue"], 0),
        })

    axes[0].set_title("Predicted Demand vs Fare Change")
    axes[0].set_xlabel("Fare change (%)")
    axes[0].set_ylabel("Predicted daily bookings")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Predicted Revenue vs Fare Change")
    axes[1].set_xlabel("Fare change (%)")
    axes[1].set_ylabel("Predicted daily revenue (AED)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/home/claude/revenue_project/outputs/price_optimization_curve.png", dpi=150)
    print("Saved chart: outputs/price_optimization_curve.png\n")

    summary_df = pd.DataFrame(summary)
    print("=== Optimal fare adjustment per route (revenue-maximizing) ===")
    print(summary_df.to_string(index=False))
    return summary_df


def demand_shock_scenario(route: str, target_date: pd.Timestamp, shock_pct: float = -0.25,
                           mitigation_fare_cut_pct: float = -0.15, horizon_days: int = 14):
    ref_price = current_reference_price(route)
    elasticity = estimate_price_elasticity(route)
    dates = pd.date_range(target_date, periods=horizon_days, freq="D")

    rows = []
    for d in dates:
        t = t_index_for(d)

        # baseline (no shock) - ML forecast at reference fare
        feat_base = make_feature_row(route, d, ref_price, 0, t)
        demand_base = MODEL.predict(feat_base)[0]

        # shocked, no action taken: an exogenous demand shock (e.g. disruption,
        # competitor capacity change) hits regardless of price
        demand_shock_noaction = demand_base * (1 + shock_pct)
        revenue_noaction = demand_shock_noaction * ref_price

        # shocked, mitigating fare cut applied: price-driven demand recovery
        # follows the estimated elasticity curve, on top of the same shock
        mitigated_price = ref_price * (1 + mitigation_fare_cut_pct)
        demand_mitigated = demand_base * (1 + shock_pct) * (mitigated_price / ref_price) ** (-elasticity)
        revenue_mitigated = demand_mitigated * mitigated_price

        revenue_baseline = demand_base * ref_price

        rows.append({
            "date": d, "revenue_baseline": revenue_baseline,
            "revenue_shock_no_action": revenue_noaction,
            "revenue_shock_with_fare_cut": revenue_mitigated,
        })

    df = pd.DataFrame(rows)
    total_baseline = df["revenue_baseline"].sum()
    total_noaction = df["revenue_shock_no_action"].sum()
    total_mitigated = df["revenue_shock_with_fare_cut"].sum()

    loss_noaction = total_baseline - total_noaction
    loss_mitigated = total_baseline - total_mitigated
    recovered = loss_noaction - loss_mitigated

    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["revenue_baseline"], label="Baseline (no disruption)", linewidth=2)
    plt.plot(df["date"], df["revenue_shock_no_action"], label=f"Disruption, no action ({shock_pct:+.0%} demand)", linewidth=2, linestyle="--")
    plt.plot(df["date"], df["revenue_shock_with_fare_cut"], label=f"Disruption + {mitigation_fare_cut_pct:+.0%} fare response", linewidth=2, linestyle=":")
    plt.title(f"{route}: Demand Shock Contingency – 14-Day Revenue Impact")
    plt.xlabel("Date")
    plt.ylabel("Predicted daily revenue (AED)")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/revenue_project/outputs/demand_shock_scenario.png", dpi=150)
    print("Saved chart: outputs/demand_shock_scenario.png\n")

    print(f"=== Demand shock contingency: {route} ({horizon_days}-day window) ===")
    print(f"Revenue loss with no action:        AED {loss_noaction:,.0f}")
    print(f"Revenue loss with fare-cut response: AED {loss_mitigated:,.0f}")
    print(f"Revenue recovered by taking action:  AED {recovered:,.0f} "
          f"({recovered / loss_noaction * 100:.1f}% of the loss offset)")

    return df, {"loss_noaction": loss_noaction, "loss_mitigated": loss_mitigated, "recovered": recovered}


if __name__ == "__main__":
    target_date = pd.Timestamp("2025-12-15")
    routes = list(pd.unique(RAW["route"]))

    plot_price_curves(target_date, routes)
    print()
    demand_shock_scenario("DXB-BKK", pd.Timestamp("2025-11-01"))
