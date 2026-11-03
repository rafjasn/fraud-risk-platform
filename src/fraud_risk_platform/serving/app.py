"""The prediction API: one pinned release, pooled database connections, metrics."""

import logging
import os
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraud_risk_platform.contracts.api import PredictionRequest, PredictionResponse
from fraud_risk_platform.models.deployment import deployment
from fraud_risk_platform.persistence.files import state_root
from fraud_risk_platform.serving.metrics import NO_RELEASE, ApiMetrics
from fraud_risk_platform.serving.predictor import FraudPredictor
from fraud_risk_platform.serving.storage import (
    card_history,
    pool,
    record_prediction,
    unconfigured,
)

logger = logging.getLogger("uvicorn.error.fraud")


def create_app(
    *,
    root: Path | None = None,
    database_url: str | None = None,
    source_database_url: str | None = None,
    release_id: str | None = None,
) -> FastAPI:
    metrics = ApiMetrics()
    # Explicit URLs win over the environment; the source defaults to the platform
    # database.
    platform_url = database_url or os.environ.get("DATABASE_URL")
    source_url = (
        source_database_url
        or database_url
        or os.environ.get("SOURCE_DATABASE_URL")
        or platform_url
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        storage = root or state_root()
        with ExitStack() as pools:

            def connections(url, name):
                if not url:
                    return unconfigured
                return pools.enter_context(pool(url, name=name)).connection

            # Requests check connections out of these pools for the app's lifetime.
            app.state.platform = connections(platform_url, "platform")
            app.state.source = source = connections(source_url, "source")
            try:
                selected = (
                    release_id
                    or os.environ.get("MODEL_RELEASE_ID")
                    or deployment(storage)["current"]
                )
                if selected:
                    app.state.predictor = FraudPredictor(
                        root=storage,
                        release_id=selected,
                        history=lambda **query: card_history(source, **query),
                        step=lambda name: metrics.step(name, selected),
                    )
                    logger.info("release_loaded release_id=%s", selected)
                else:
                    logger.warning("no_deployed_release; promote and restart the API")
            except Exception:
                logger.exception("release_load_failed")
            metrics.loaded(app.state.predictor and app.state.predictor.release_id)
            yield
            app.state.predictor = None

    app = FastAPI(title="Fraud Risk Platform", version="0.1.0", lifespan=lifespan)

    def serving(request: Request) -> str:
        predictor = request.app.state.predictor
        return predictor.release_id if predictor else NO_RELEASE

    @app.middleware("http")
    async def measure(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        started, status = perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # The route template, never the raw path, keeps label values bounded.
            route = request.scope.get("route")
            metrics.request(
                endpoint=getattr(route, "path", "unmatched"),
                status=status,
                release=serving(request),
                seconds=perf_counter() - started,
                cause=getattr(request.state, "error_cause", None),
            )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(
            generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(
                503, "No compatible model loaded; promote a release and restart"
            )
        try:
            with request.app.state.platform() as conn:
                conn.execute(
                    "SELECT prediction_id, feature_inputs FROM predictions LIMIT 0"
                )
            with request.app.state.source() as conn:
                conn.execute("SELECT source_row FROM ibm_complete_transactions LIMIT 0")
        except psycopg.Error:
            raise HTTPException(503, "Prediction storage unavailable") from None
        return {"status": "ready", "release_id": predictor.release_id}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        predictor = request.app.state.predictor
        if predictor is None:
            request.state.error_cause = "no_model"
            raise HTTPException(503, "No compatible model loaded")
        inputs = payload.model_dump(exclude_unset=True, exclude={"transaction_id"})
        try:
            prediction = predictor.predict(**inputs)
        except ValueError as error:
            request.state.error_cause = "validation"
            raise HTTPException(422, str(error)) from error
        except psycopg.Error:
            # Card-history releases read the source database before scoring.
            logger.exception("card_history_failed")
            request.state.error_cause = "storage"
            raise HTTPException(
                503, "Card history is unavailable; retry when storage is healthy"
            ) from None
        result = {
            "prediction_id": str(uuid4()),
            "transaction_id": payload.transaction_id,
            **prediction,
        }
        try:
            with metrics.step("persistence", predictor.release_id):
                record_prediction(
                    payload,
                    result,
                    platform=request.app.state.platform,
                    source=request.app.state.source,
                    feature_inputs={
                        name: inputs[name]
                        for name in predictor.feature_spec.inputs
                        if name not in {"amount", "tx_datetime"}
                    },
                )
        except ValueError as error:
            request.state.error_cause = "conflict"
            raise HTTPException(409, str(error)) from error
        except psycopg.Error:
            logger.exception(
                "prediction_persistence_failed prediction_id=%s",
                result["prediction_id"],
            )
            request.state.error_cause = "storage"
            raise HTTPException(
                503, "Prediction could not be recorded; retry when storage is healthy"
            ) from None
        metrics.predictions.labels(
            predictor.release_id, str(result["flagged_for_review"]).lower()
        ).inc()
        logger.info(
            "prediction_recorded prediction_id=%s transaction_id=%s release_id=%s flagged=%s",
            result["prediction_id"],
            payload.transaction_id,
            predictor.release_id,
            result["flagged_for_review"],
        )
        return result

    return app


app = create_app()
