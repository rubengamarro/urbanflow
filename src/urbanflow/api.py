import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from urbanflow.config import Settings
from urbanflow.serving import Predictor

logger = logging.getLogger("urbanflow.api")


class ForecastRequest(BaseModel):
    zone_ids: list[int] = Field(min_length=1, max_length=263)
    target_hour: datetime | None = None


class ZonePrediction(BaseModel):
    zone_id: int
    predicted_pickups: float
    lower_90: float
    upper_90: float


class ForecastResponse(BaseModel):
    target_hour: str
    observed_through: str
    model_version: str
    mode: str
    predictions: list[ZonePrediction]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        app.state.requests = 0
        app.state.failures = 0
        app.state.latency_seconds = 0.0
        try:
            app.state.predictor = Predictor(settings.artifacts)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            logger.error("Model readiness failed: %s", exc)
        yield

    app = FastAPI(
        title="UrbanFlow AI",
        version="1.0.0",
        lifespan=lifespan,
        description="UTC one-hour-ahead yellow taxi pickup forecasts. Historical replay available.",
    )

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        app.state.requests += 1
        app.state.failures += int(response.status_code >= 400)
        app.state.latency_seconds += elapsed
        logger.info(
            "method=%s path=%s status=%s duration=%.4f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    def predictor() -> Predictor:
        if app.state.predictor is None:
            raise HTTPException(
                503, "No verified model bundle. Run the training pipeline and restart API."
            )
        return app.state.predictor

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        model = predictor()
        return {
            "status": "ready",
            "model_version": model.metadata["version"],
            "history_end_utc": model.latest.tz_localize("UTC").isoformat(),
        }

    @app.get("/model")
    def metadata():
        return predictor().metadata

    @app.get("/evaluation")
    def evaluation():
        return predictor().evaluation

    @app.get("/metrics", response_class=PlainTextResponse)
    def operational_metrics():
        return (
            f"urbanflow_requests_total {app.state.requests}\n"
            f"urbanflow_failed_requests_total {app.state.failures}\n"
            f"urbanflow_request_duration_seconds_sum {app.state.latency_seconds}\n"
            f"urbanflow_model_ready {int(app.state.predictor is not None)}\n"
        )

    @app.post("/predict", response_model=ForecastResponse)
    def forecast(body: ForecastRequest):
        import pandas as pd

        try:
            target = pd.Timestamp(body.target_hour) if body.target_hour else None
            return predictor().forecast(body.zone_ids, target)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
