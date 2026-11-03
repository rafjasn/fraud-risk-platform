# Helm chart: fraud-risk-platform

Runs the platform on Kubernetes from the same images as Docker Compose, as an alternative local setup. It installs PostgreSQL, LocalStack S3, MLflow, and the prediction API. Every platform command runs as a Kubernetes Job: database setup is a Helm hook, the monthly workflow is a CronJob, and any other command runs on demand.

| Component | Kind | Notes |
| --- | --- | --- |
| PostgreSQL | StatefulSet with a volume claim | The source, platform, and MLflow databases; roles are created on first start from the Secret's passwords |
| LocalStack | Deployment | S3 only, in memory; buckets are created on start |
| MLflow | Deployment | Tracking server and registry, with artifacts in S3 |
| API | Deployment | Rolling updates with readiness checks on `/ready` |
| Database setup | Job, Helm post-install and post-upgrade hook | `init-db`; additive, safe to repeat |
| Month | CronJob, suspended by default | Replay, snapshot, monitoring, training |
| Shared state | PersistentVolumeClaim | Releases, `deployment.json`, and reports; kept on uninstall |

## Quick start on kind

You need Docker, [kind](https://kind.sigs.k8s.io/), kubectl, and Helm 3, with at least 8 GB of memory for Docker.

```sh
kind create cluster
make setup              # build the images
make k8s-images         # load them into the kind cluster
make k8s-install        # install the chart in namespace fraud-risk; waits for database setup

make k8s-task ARGS=seed-source          # once: download and load the IBM data
make k8s-task ARGS=initial-snapshot
make k8s-month                          # train the first cycle
make k8s-promote RELEASE=<release-id> INITIAL=--initial
make k8s-month                          # each further month; promote when recommended
make k8s-task ARGS=final-evaluation     # once a month reports holdout_reached
```

`make k8s-task ARGS="<command>"` runs any platform command as a Job and follows its logs, for example `ARGS="train --snapshot <id>"` or `ARGS=training-jobs`. Reach the API and MLflow with `kubectl port-forward`; the install notes print the commands. `helm test fraud-risk-platform --namespace fraud-risk` checks that the API answers.

## Promotion on Kubernetes

`make k8s-promote` runs the platform's own `promote` command, which refuses a release its training cycle did not recommend. It then restarts the API Deployment. The rolling update allows no unavailable pods, and a new pod receives traffic only once `/ready` reports its release loaded and verified. A release that fails to load therefore never serves, and the old pod keeps serving until `verify-deployment` restores the last confirmed release. `make k8s-rollback` returns to the previous confirmed release the same way.

To pin a release declaratively instead, set `api.releaseId`; changing it triggers the same gated rolling update.

## Values

| Value | Default | Purpose |
| --- | --- | --- |
| `images.training`, `images.serving` | `fraud-risk-training:local`, `fraud-risk-serving:local` | The project's images; `make setup` builds them |
| `credentials.existingSecret` | empty | A Secret with the keys in `templates/secret.yaml`, instead of the development defaults |
| `state.size`, `state.accessMode` | `20Gi`, `ReadWriteOnce` | Shared state; several nodes need `ReadWriteMany` |
| `postgres.storage.size` | `30Gi` | The source database alone takes about 15 GB |
| `api.replicas`, `api.releaseId` | `1`, empty | API scale, and an optional pinned release |
| `jobs.resources` | 2 Gi requested, 6 Gi limit | Every command Job; training needs the most |
| `month.schedule`, `month.suspend`, `month.trials` | first of the month, `true`, `20` | The monthly CronJob |

## Practices and limits

- **Security:** the project's containers run as uid 1000 with a read-only root filesystem, no privilege escalation, no Linux capabilities, and the default seccomp profile. No pod gets a Kubernetes API token. Passwords live in a Secret, and connection URLs are assembled from it at runtime, so no password appears in a ConfigMap or manifest. LocalStack, a local emulator, keeps its image defaults.
- **Reliability:** every container has resource requests and memory limits, and probes for readiness and liveness. Pods that need the database wait for it instead of crash-looping, and configuration changes roll the pods through checksum annotations.
- **Validation:** `make chart-check` lints the chart and validates every rendered manifest against the Kubernetes schemas; CI runs it on every push.
- **Limits:** the development credentials and in-memory S3 suit a local cluster only. On a cluster with several nodes, the shared state volume needs `ReadWriteMany` storage. Prometheus and Grafana are not included: the API pods carry the standard `prometheus.io` scrape annotations for an existing monitoring stack. [infra/terraform](../../infra/terraform/README.md) defines the AWS infrastructure on ECS.
