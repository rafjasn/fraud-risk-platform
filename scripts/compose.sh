#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
export PROJECT_STATE_DIR="${PROJECT_STATE_DIR:-$PWD/state}"
mkdir -p "$PROJECT_STATE_DIR/platform" "$PROJECT_STATE_DIR/mlflow" data/raw
# Provenance for training runs: the commit an image build copies, and the image a job
# runs. Any uncommitted change, including an untracked file, marks the build dirty.
if command -v git >/dev/null && git rev-parse --verify --quiet HEAD >/dev/null; then
    export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD)}"
    if [ -z "$(git status --porcelain)" ]; then
        export GIT_DIRTY="${GIT_DIRTY:-false}"
    else
        export GIT_DIRTY="${GIT_DIRTY:-true}"
    fi
fi
TRAINING_IMAGE_ID="${TRAINING_IMAGE_ID:-$(docker image inspect --format '{{.Id}}' fraud-risk-training:local 2>/dev/null || true)}"
export TRAINING_IMAGE_ID
exec docker compose "$@"
