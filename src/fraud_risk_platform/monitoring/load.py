"""Load test: the coming month's transactions sent to the running API over HTTP.

Replay scores a month in one batch and never calls the API, so it says nothing about
serving. A load test sends a deterministic sample of the month after the simulation
clock to `/predict` at a fixed rate and records each response's status and latency.
It reads the API's own metrics before and after, so the report also shows where the
server spent its time. The API builds card history as of each transaction's time, so
the requests reveal no outcomes; their predictions are stored like any other API
prediction, without a replay ID.

Requests are scheduled at fixed times rather than sent as fast as responses return,
so a slow API delays later requests instead of hiding its latency; the report shows
how far sending fell behind. Measurements from a laptop compare releases and steps;
they are not production latency.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from prometheus_client.parser import text_string_to_metric_families

from fraud_risk_platform.dataops.workflow import simulation_clock
from fraud_risk_platform.evaluation.charts import write_load
from fraud_risk_platform.persistence.database import connect_source
from fraud_risk_platform.persistence.files import write_json
from fraud_risk_platform.reporting.page import (
    badge,
    e,
    evidence,
    figure,
    page,
    section,
    table,
    tiles,
)

logger = logging.getLogger(__name__)
# Optional API request fields available in the source; nulls are sent as null.
API_FIELDS = (
    "use_chip",
    "mcc",
    "merchant_city",
    "merchant_state",
    "postal_code",
    "user_id",
    "card_id",
    "merchant_id",
)
STEPS = ("history", "scoring", "persistence")


class HttpApi:
    """The prediction API over HTTP; a POST without any response has status None."""

    def __init__(self, url: str, timeout: float = 10.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict) -> int | None:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response.read()
                return response.status
        except urllib.error.HTTPError as error:
            error.read()
            return error.code
        except OSError:
            return None

    def get(self, path: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(
                self.url + path, timeout=self.timeout
            ) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()


def _sample(settings, requests, *, source_url, platform_url):
    """The simulation clock and a deterministic sample of the month after it."""
    clock = simulation_clock(settings, source_url=source_url, platform_url=platform_url)
    start, end = clock["clock"], clock["next_cutoff"]
    if start >= end:
        raise ValueError("The simulation has reached the end of the source data")
    with connect_source(source_url) as conn:
        rows = conn.execute(
            """SELECT source_row, tx_datetime, amount::float8 AS amount, use_chip, mcc,
            merchant_city, merchant_state, postal_code, user_id, card_id, merchant_id
            FROM ibm_complete_transactions
            WHERE import_id=%s AND tx_datetime >= %s AND tx_datetime < %s
            ORDER BY md5(source_row::text) LIMIT %s""",
            (clock["import_id"], start, end, requests),
        ).fetchall()
    rows.sort(key=lambda row: (row["tx_datetime"], row["source_row"]))
    return clock, rows


def payload(row) -> dict:
    return {
        "transaction_id": row["source_row"],
        "amount": row["amount"],
        "timestamp": row["tx_datetime"].isoformat(),
        **{name: row[name] for name in API_FIELDS},
    }


def _ready_release(api) -> str:
    status, body = api.get("/ready")
    if status != 200:
        raise ValueError(
            f"The API is not ready ({status}): promote a release and run make up"
        )
    return json.loads(body)["release_id"]


def _send(api, bodies, *, rate, concurrency):
    """Send each body at its scheduled time; returns per-request results and seconds."""
    started = perf_counter()

    def send(index, body):
        scheduled = started + index / rate
        if (delay := scheduled - perf_counter()) > 0:
            sleep(delay)
        sent = perf_counter()
        status = api.post("/predict", body)
        return {
            "offset": sent - started,
            "lag": sent - scheduled,
            "seconds": perf_counter() - sent,
            "status": status,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(send, range(len(bodies)), bodies))
    return results, perf_counter() - started


def _percentiles(seconds) -> dict | None:
    if not seconds:
        return None
    p50, p95, p99 = np.percentile(seconds, [50, 95, 99]) * 1000
    return {
        "count": len(seconds),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "max": float(max(seconds) * 1000),
    }


def client_summary(results, seconds) -> dict:
    """What the load generator saw: statuses, error rates, and latency."""
    statuses = Counter(
        "none" if r["status"] is None else str(r["status"]) for r in results
    )
    total = len(results)
    failed = sum(n for s, n in statuses.items() if s == "none" or s.startswith("5"))
    rejected = sum(n for s, n in statuses.items() if s.startswith("4"))
    return {
        "requests": total,
        "statuses": dict(sorted(statuses.items())),
        "server_error_rate": failed / total,
        "client_error_rate": rejected / total,
        "achieved_rate": total / seconds,
        "max_lag_ms": 1000 * max(r["lag"] for r in results),
        "latency_ms": _percentiles(
            [r["seconds"] for r in results if r["status"] == 200]
        ),
    }


def _totals(text: str) -> dict:
    """Counter, histogram count, and histogram sum samples of a metrics scrape."""
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(text)
        for sample in family.samples
        if sample.name.endswith(("_count", "_sum", "_total"))
    }


def server_summary(before: str, after: str) -> dict:
    """The API's own measurements between two scrapes of /metrics."""
    start = _totals(before)
    delta = {key: value - start.get(key, 0.0) for key, value in _totals(after).items()}

    def grouped(name, by, **match):
        result = {}
        for (sample, labels), value in delta.items():
            labels = dict(labels)
            if (
                sample == name
                and value
                and all(labels[k] == v for k, v in match.items())
            ):
                result[labels[by]] = result.get(labels[by], 0.0) + value
        return result

    counts = grouped("fraud_api_request_seconds_count", "status", endpoint="/predict")
    total_seconds = sum(
        grouped("fraud_api_request_seconds_sum", "status", endpoint="/predict").values()
    )
    step_counts = grouped("fraud_api_step_seconds_count", "step")
    step_seconds = grouped("fraud_api_step_seconds_sum", "step")
    requests = sum(counts.values())
    steps = {
        step: {
            "count": int(step_counts[step]),
            "mean_ms": 1000 * step_seconds[step] / step_counts[step],
            "ms_per_request": 1000 * step_seconds[step] / requests,
        }
        for step in STEPS
        if step_counts.get(step)
    }
    mean = 1000 * total_seconds / requests if requests else None
    return {
        "requests": int(requests),
        "statuses": {status: int(n) for status, n in sorted(counts.items())},
        "errors": {
            cause: int(n)
            for cause, n in sorted(
                grouped("fraud_api_errors_total", "cause", endpoint="/predict").items()
            )
        },
        "mean_ms": mean,
        "steps": steps,
        # Request time outside the measured steps: validation, routing, queueing.
        "other_ms_per_request": None
        if mean is None
        else max(mean - sum(s["ms_per_request"] for s in steps.values()), 0.0),
    }


def run_load(
    root: Path,
    settings,
    *,
    requests: int,
    rate: float,
    concurrency: int,
    api=None,
    api_url: str | None = None,
    source_url=None,
    platform_url=None,
) -> dict:
    if requests < 1 or rate <= 0 or concurrency < 1:
        raise ValueError("Requests, rate, and concurrency must be positive")
    api = api or HttpApi(api_url or os.environ.get("API_URL", "http://api:8000"))
    clock, rows = _sample(
        settings, requests, source_url=source_url, platform_url=platform_url
    )
    if not rows:
        raise ValueError("The month after the simulation clock has no transactions")
    release = _ready_release(api)
    before = api.get("/metrics")[1]
    began = datetime.now(timezone.utc)
    logger.info(
        "Sending %d transactions from %s at %g requests/s with %d connections to release %s",
        len(rows),
        clock["clock"].date(),
        rate,
        concurrency,
        release[:12],
    )
    results, seconds = _send(
        api, [payload(row) for row in rows], rate=rate, concurrency=concurrency
    )
    finished = datetime.now(timezone.utc)
    after = api.get("/metrics")[1]
    release_after = _ready_release(api)
    client = client_summary(results, seconds)
    server = server_summary(before, after)
    output = root / "load" / f"{began:%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True)
    grafana = os.environ.get("GRAFANA_URL", "http://127.0.0.1:3000")
    window = [began - timedelta(seconds=30), finished + timedelta(seconds=30)]
    report = {
        "schema_version": 1,
        "started_at": began.isoformat(),
        "finished_at": finished.isoformat(),
        "release_id": release,
        "release_changed_to": None if release_after == release else release_after,
        "simulation_clock": clock["clock"].isoformat(),
        "traffic_window": [
            clock["clock"].isoformat(),
            clock["next_cutoff"].isoformat(),
        ],
        "sample": f"{len(rows):,} transactions with the lowest MD5 of their source row, "
        "sent in transaction-time order",
        "target_rate": rate,
        "concurrency": concurrency,
        "client": client,
        "server": server,
        "dashboard": f"{grafana}/d/fraud-api/fraud-api"
        f"?from={int(window[0].timestamp() * 1000)}&to={int(window[1].timestamp() * 1000)}",
    }
    steps = {step: values["ms_per_request"] for step, values in server["steps"].items()}
    if server["other_ms_per_request"] is not None:
        steps["other"] = server["other_ms_per_request"]
    report["plots"] = [
        write_load(
            output,
            results,
            client["latency_ms"],
            steps,
            title=f"Load test of release {release[:12]}: {len(rows):,} requests "
            f"at {rate:g}/s, {_connections(concurrency)}, {began:%Y-%m-%d %H:%M} UTC",
        )
    ]
    write_json(output / "report.json", report)
    (output / "report.html").write_text(_html(output, report))
    logger.info(
        "Load test finished: p95 %s ms, server errors %.2f%%, client errors %.2f%%",
        "–" if client["latency_ms"] is None else f"{client['latency_ms']['p95']:.1f}",
        client["server_error_rate"] * 100,
        client["client_error_rate"] * 100,
    )
    return {"report": str(output / "report.html"), **report}


def _connections(count):
    return f"{count} connection{'' if count == 1 else 's'}"


def _ms(value):
    return "–" if value is None else f"{value:,.1f} ms"


def _html(directory: Path, report: dict) -> str:
    client, server = report["client"], report["server"]
    latency = client["latency_ms"] or {}
    failed = client["server_error_rate"]
    statuses = ", ".join(f"{s}: {n:,}" for s, n in client["statuses"].items())
    errors = ", ".join(f"{c}: {n:,}" for c, n in server["errors"].items()) or "none"
    changed = (
        f'<p class="notice"><strong>The API switched to release {e(report["release_changed_to"])} during the run;</strong> '
        "server figures mix both releases.</p>"
        if report["release_changed_to"]
        else ""
    )
    figures = "".join(
        figure(
            (directory / name).read_bytes(),
            alt="Client latency of each request over the run, with p50, p95, and p99 "
            "lines; and mean server time per request by step",
        )
        for name in report["plots"]
    )
    client_rows = [
        [
            "Requests (achieved rate)",
            f"{client['requests']:,} ({client['achieved_rate']:.1f}/s; sending fell behind by up to {_ms(client['max_lag_ms'])})",
        ],
        ["Statuses", statuses],
        ["Server errors (5xx or no response)", f"{failed:.2%}"],
        ["Client errors (4xx)", f"{client['client_error_rate']:.2%}"],
        [
            "Latency of successful requests: p50 / p95 / p99 / max",
            " / ".join(_ms(latency.get(name)) for name in ("p50", "p95", "p99", "max")),
        ],
    ]
    step_rows = [
        [
            step,
            f"{values['count']:,}",
            _ms(values["mean_ms"]),
            _ms(values["ms_per_request"]),
        ]
        for step, values in server["steps"].items()
    ] + [
        [
            "other (validation, routing, queueing)",
            "",
            "",
            _ms(server["other_ms_per_request"]),
        ],
        [
            "whole request",
            f"{server['requests']:,}",
            _ms(server["mean_ms"]),
            _ms(server["mean_ms"]),
        ],
    ]
    body = section(
        "client",
        "What the client saw",
        table(["Measure", "Value"], client_rows) + figures,
    ) + section(
        "server",
        "Where the server spent its time",
        table(
            ["Server step", "Times run", "Mean per run", "Mean per request"],
            step_rows,
            align="lrrr",
        )
        + f"<p>Server errors by cause: {e(errors)}. Server figures come from the API's metrics before and after the run, so they include any other requests the API served meanwhile. History runs only for releases with card-history features, and persistence only for requests that scored. Client latency includes HTTP and Docker networking; the server's request time does not.</p>"
        "<p>These figures come from one API worker on a local machine, sharing it with PostgreSQL. Use them to compare releases and steps, not as production latency.</p>",
    )
    return page(
        area="Serving",
        title=f"API load test — release {report['release_id'][:12]}",
        badge=badge(f"{failed:.2%} server errors", "critical")
        if failed
        else badge("No server errors", "good"),
        intro=f'<p>{e(report["sample"])} from the month starting {e(report["traffic_window"][0][:10])}, sent to <code>/predict</code> at {report["target_rate"]:g} requests/s over {_connections(report["concurrency"])}, from {e(report["started_at"][:19])} to {e(report["finished_at"][:19])} UTC. <a href="{e(report["dashboard"])}">Open this period in Grafana</a>.</p>'
        + changed,
        tiles=tiles(
            [
                (
                    "Requests",
                    f"{client['requests']:,}",
                    f"{client['achieved_rate']:.1f}/s achieved; target {report['target_rate']:g}/s",
                ),
                (
                    "Latency p95, successful requests",
                    _ms(latency.get("p95")),
                    f"p50 {_ms(latency.get('p50'))} · p99 {_ms(latency.get('p99'))}",
                ),
                ("Server errors", f"{failed:.2%}", "5xx or no response"),
                ("Client errors", f"{client['client_error_rate']:.2%}", "4xx"),
            ]
        ),
        body=body,
        evidence=evidence(report),
    )
