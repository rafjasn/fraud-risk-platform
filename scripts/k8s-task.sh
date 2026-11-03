#!/usr/bin/env sh
# Run one platform command as a Kubernetes Job in the installed Helm release, follow its
# logs, and exit with its result. The arguments are the CLI's, for example:
#
#   scripts/k8s-task.sh promote --release <id> --reason "Manual operator promotion"
#
# The Job is rendered from the chart with the release's own values, so it runs with the
# same image, settings, and volumes as the pods the chart installed.
set -eu
cd "$(dirname "$0")/.."
release="${HELM_RELEASE:-fraud-risk-platform}"
namespace="${K8S_NAMESPACE:-fraud-risk}"
if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [arguments...]" >&2
    exit 2
fi

values=$(mktemp)
trap 'rm -f "$values"' EXIT
helm get values "$release" --namespace "$namespace" --all --output yaml >"$values"
args=$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")

job=$(helm template "$release" charts/fraud-risk-platform --namespace "$namespace" \
        --values "$values" --show-only templates/task-job.yaml --set-json "task.args=$args" \
    | kubectl create --namespace "$namespace" --filename - --output name)
echo "Started $job" >&2
kubectl logs --namespace "$namespace" --follow --pod-running-timeout=10m "$job"

# The pod has exited; give the Job controller a moment to record the outcome.
if kubectl wait --namespace "$namespace" --for=condition=Complete --timeout=60s "$job" >/dev/null 2>&1; then
    exit 0
fi
echo "$job failed" >&2
exit 1
