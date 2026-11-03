"""Prometheus metrics for the prediction API: request latency, time per step, errors.

Each app owns its registry, so tests can create apps independently. The API runs one
Uvicorn worker, so in-process metrics cover all of its traffic. Prometheus scrapes
`/metrics`; the API never depends on Prometheus being up.
"""

from contextlib import contextmanager
from time import perf_counter

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
)

disable_created_metrics()

# Seconds: scoring takes milliseconds, while a card-history query can take seconds.
BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
# Causes the handlers name; other error statuses fall back to these or `http_<code>`.
DEFAULT_CAUSES = {422: "validation", 500: "unhandled"}
NO_RELEASE = "none"


class ApiMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Histogram(
            "fraud_api_request_seconds",
            "Request duration by endpoint, HTTP status, and loaded release",
            ["endpoint", "status", "release"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.steps = Histogram(
            "fraud_api_step_seconds",
            "Prediction time per step: history, scoring, persistence",
            ["step", "release"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.errors = Counter(
            "fraud_api_errors",
            "Error responses by endpoint, cause, and loaded release",
            ["endpoint", "cause", "release"],
            registry=self.registry,
        )
        self.predictions = Counter(
            "fraud_api_predictions",
            "Recorded predictions by release and review decision",
            ["release", "flagged"],
            registry=self.registry,
        )
        self.release = Gauge(
            "fraud_api_release_info",
            "1 for the release this process serves, or for 'none'",
            ["release"],
            registry=self.registry,
        )

    def loaded(self, release: str | None):
        self.release.labels(release or NO_RELEASE).set(1)

    @contextmanager
    def step(self, name: str, release: str):
        """Time one step of a prediction, including a step that fails."""
        started = perf_counter()
        try:
            yield
        finally:
            self.steps.labels(name, release).observe(perf_counter() - started)

    def request(
        self,
        *,
        endpoint: str,
        status: int,
        release: str,
        seconds: float,
        cause: str | None,
    ):
        self.requests.labels(endpoint, str(status), release).observe(seconds)
        if status >= 400:
            cause = cause or DEFAULT_CAUSES.get(status, f"http_{status}")
            self.errors.labels(endpoint, cause, release).inc()
