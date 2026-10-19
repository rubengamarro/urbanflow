from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from urbanflow.io import write_json


def analyze(panel_path: Path, lookup_path: Path, reports: Path) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(panel_path)
    lookup = pd.read_csv(lookup_path).rename(columns={"LocationID": "zone_id"})
    local = frame.hour.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    frame["local_hour"] = local.dt.hour
    frame["local_dow"] = local.dt.dayofweek
    daily = frame.groupby(local.dt.tz_localize(None).dt.floor("D")).demand.sum()
    zone = frame.groupby("zone_id").demand.sum().sort_values(ascending=False)
    top = zone.head(15).reset_index().merge(lookup, on="zone_id")
    top.to_csv(reports / "top_zones.csv", index=False)
    summary = {
        "rows": len(frame),
        "zones": frame.zone_id.nunique(),
        "total_valid_pickups_in_scope": int(frame.demand.sum()),
        "zero_fraction": float(frame.demand.eq(0).mean()),
        "start_utc": str(frame.hour.min()),
        "end_utc": str(frame.hour.max()),
        "hourly_count_quantiles": frame.demand.quantile([0, 0.5, 0.9, 0.99, 1]).to_dict(),
    }
    write_json(reports / "eda.json", summary)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    daily.plot(ax=axes[0, 0], title="Daily observed pickups (NYC)", color="#0e7490")
    axes[0, 0].set_ylabel("Pickups")
    top.set_index("Zone").demand.sort_values().plot.barh(
        ax=axes[0, 1], title="Top 15 zones", color="#0e7490"
    )
    heat = frame.pivot_table(index="local_dow", columns="local_hour", values="demand")
    im = axes[1, 0].imshow(heat, aspect="auto", cmap="Blues")
    axes[1, 0].set(title="Mean zone demand, NYC calendar", xlabel="Hour", ylabel="Weekday (Mon=0)")
    fig.colorbar(im, ax=axes[1, 0], label="Pickups per zone-hour")
    frame.demand.clip(upper=frame.demand.quantile(0.99)).hist(
        bins=50, ax=axes[1, 1], color="#0e7490"
    )
    axes[1, 1].set(title="Hourly zone counts (clipped at p99)", yscale="log", xlabel="Pickups")
    fig.savefig(reports / "eda.png", dpi=140)
    plt.close(fig)
    return summary
