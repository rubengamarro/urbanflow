import json

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from urbanflow.config import Settings

settings = Settings()
st.set_page_config(page_title="UrbanFlow AI", page_icon="🚕", layout="wide")
st.title("🚕 UrbanFlow AI")
st.caption(
    "Observed NYC yellow taxi pickups · one-hour forecasting · reproducible temporal benchmark"
)


def fetch(path: str):
    response = httpx.get(f"{settings.api_url}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


try:
    metadata = fetch("/model")
    evaluation = fetch("/evaluation")
except httpx.HTTPError as exc:
    st.error(f"API not ready: {exc}. Run the pipeline and start the API.")
    st.stop()

st.sidebar.subheader("Model snapshot")
st.sidebar.write(metadata["model"])
st.sidebar.code(metadata["version"])
st.sidebar.caption(f"Observations through {metadata['history_end_utc']} UTC")
st.sidebar.caption("Historical observations are replayed. This is not a live dispatch feed.")
forecast_tab, benchmark_tab, explorer_tab, operations_tab = st.tabs(
    ["Demand forecast", "Model benchmark", "Data explorer", "Operations"]
)

with forecast_tab:
    lookup_path = settings.raw / "taxi_zone_lookup.csv"
    names = {}
    if lookup_path.exists():
        lookup = pd.read_csv(lookup_path)
        names = dict(zip(lookup.LocationID, lookup.Zone, strict=True))
    selected = st.multiselect(
        "Taxi zones",
        metadata["zones"],
        default=[161] if 161 in metadata["zones"] else metadata["zones"][:3],
        format_func=lambda z: f"{z} · {names.get(z, 'Zone')}",
    )
    replay = st.checkbox("Replay a held-out historical hour")
    target = None
    if replay:
        start = pd.Timestamp(evaluation["partitions"]["test"]["start"])
        end = pd.Timestamp(metadata["history_end_utc"])
        hours = pd.date_range(start, end, freq="h")
        target = st.select_slider("Target hour (UTC)", options=list(hours), value=hours[-1])
    if st.button("Forecast demand", type="primary", disabled=not selected):
        try:
            response = httpx.post(
                f"{settings.api_url}/predict",
                timeout=30,
                json={
                    "zone_ids": selected,
                    "target_hour": target.tz_localize("UTC").isoformat()
                    if target is not None
                    else None,
                },
            )
            response.raise_for_status()
            result = response.json()
            st.caption(f"Target: {result['target_hour']} · mode: {result['mode']}")
            predictions = pd.DataFrame(result["predictions"])
            predictions["zone"] = predictions.zone_id.map(lambda z: names.get(z, str(z)))
            predictions["upper_error"] = predictions.upper_90 - predictions.predicted_pickups
            predictions["lower_error"] = predictions.predicted_pickups - predictions.lower_90
            st.plotly_chart(
                px.bar(
                    predictions,
                    x="zone",
                    y="predicted_pickups",
                    error_y="upper_error",
                    error_y_minus="lower_error",
                    title="Next-hour pickups with calibrated 90% intervals",
                ),
                width="stretch",
            )
            st.dataframe(
                predictions[["zone_id", "zone", "predicted_pickups", "lower_90", "upper_90"]],
                hide_index=True,
            )
            st.caption(
                "Intervals use held-out absolute residuals per zone. "
                "Temporal dependence can reduce coverage."
            )
        except httpx.HTTPError as exc:
            st.error(str(exc))

with benchmark_tab:
    a, b, c, d = st.columns(4)
    a.metric("Held-out MAE", f"{evaluation['test']['mae']:.3f}")
    b.metric("WAPE", f"{evaluation['test']['wape']:.1%}")
    c.metric("Weekly baseline MAE", f"{evaluation['test_baseline']['mae']:.3f}")
    d.metric("90% interval coverage", f"{evaluation['test_coverage_90']:.1%}")
    st.subheader("Validation comparison (model selection)")
    scores = pd.DataFrame(evaluation["validation"]).T.reset_index(names="model")
    st.plotly_chart(px.bar(scores, x="model", y="mae"), width="stretch")
    st.dataframe(scores, hide_index=True)
    st.subheader("Temporal partitions (UTC)")
    st.dataframe(pd.DataFrame(evaluation["partitions"]).T)
    st.json(evaluation["bootstrap"])
    prediction_path = settings.reports / "test_predictions.parquet"
    if prediction_path.exists():
        test = pd.read_parquet(prediction_path)
        zone = st.selectbox("Inspect a zone", metadata["zones"])
        view = test[test.zone_id.eq(zone)]
        st.plotly_chart(
            px.line(
                view,
                x="hour",
                y=["demand", "prediction", "baseline"],
                title="Rolling one-hour forecasts vs observed counts",
            ),
            width="stretch",
        )

with explorer_tab:
    panel_path = settings.processed / "hourly.parquet"
    if panel_path.exists():

        @st.cache_data
        def load_panel(path: str, modified: float):
            return pd.read_parquet(path)

        panel = load_panel(str(panel_path), panel_path.stat().st_mtime)
        local = panel.hour.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        daily = panel.groupby(local.dt.tz_localize(None).dt.floor("D")).demand.sum().reset_index()
        st.plotly_chart(
            px.line(daily, x="hour", y="demand", title="Daily pickups (NYC)"), width="stretch"
        )
        heat = panel.assign(local_hour=local.dt.hour, day=local.dt.dayofweek).pivot_table(
            index="day", columns="local_hour", values="demand"
        )
        st.plotly_chart(
            px.imshow(
                heat, labels={"x": "NYC hour", "y": "Weekday (Mon=0)", "color": "Mean pickups"}
            ),
            width="stretch",
        )
    else:
        st.info("Local processed data is needed for exploratory charts.")

with operations_tab:
    st.json(fetch("/health/ready"))
    for name in ("monitoring.json", "eda.json"):
        path = settings.reports / name
        if path.exists():
            st.subheader(name.removesuffix(".json").title())
            st.json(json.loads(path.read_text(encoding="utf-8")))
