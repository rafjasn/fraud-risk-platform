FROM python:3.14-slim-trixie AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH" MLFLOW_DISABLE_AGENT_HINT=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

FROM runtime AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./

# Dependencies precede application code so code/config edits reuse their layers.
# uv's download cache stays in BuildKit, outside the final runtime images.
FROM builder AS ingestion-build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra data --python /usr/local/bin/python
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra data --python /usr/local/bin/python

FROM runtime AS ingestion
COPY --from=ingestion-build /app/.venv /app/.venv
COPY config ./config
ENTRYPOINT ["fraud-risk-platform"]

FROM builder AS serving-build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra serving --python /usr/local/bin/python
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra serving --python /usr/local/bin/python

FROM runtime AS serving
COPY --from=serving-build /app/.venv /app/.venv
COPY config ./config
EXPOSE 8000
ENTRYPOINT ["uvicorn", "fraud_risk_platform.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

FROM builder AS training-dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra training --extra data --python /usr/local/bin/python

FROM training-dependencies AS training-build
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra training --extra data --python /usr/local/bin/python

FROM runtime AS training
COPY --from=training-build /app/.venv /app/.venv
COPY config ./config
COPY pyproject.toml uv.lock ./
# The commit the image's code came from, recorded with every training run. The image
# has no Git checkout; scripts/compose.sh passes these, so only this layer rebuilds.
ARG GIT_COMMIT=unavailable
ARG GIT_DIRTY=unavailable
ENV TRAINING_GIT_COMMIT=$GIT_COMMIT TRAINING_GIT_DIRTY=$GIT_DIRTY
ENTRYPOINT ["fraud-risk-platform"]

FROM training AS training-sagemaker
ENTRYPOINT ["python", "-m", "fraud_risk_platform.training.entrypoint"]

FROM training-dependencies AS test-build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-install-project --python /usr/local/bin/python
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-extras --no-editable --python /usr/local/bin/python

FROM runtime AS test
COPY --from=test-build /app/.venv /app/.venv
COPY pyproject.toml ./
COPY config ./config
COPY tests ./tests
ENTRYPOINT ["pytest"]
