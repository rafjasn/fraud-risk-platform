.PHONY: infra-check setup setup-data check-ingestion-image up db-init seed-source initial-snapshot next-snapshot snapshots verify-snapshot snapshot-report refresh-snapshot-report bootstrap train training-jobs sagemaker-inputs import-release promote rollback rollout status month replay monitor load outcomes final-evaluation reset-simulation test unit lint down logs chart-check k8s-images k8s-install k8s-task k8s-month k8s-promote k8s-rollback k8s-rollout k8s-uninstall
COMPOSE = ./scripts/compose.sh
SEED_ARGS ?=
SNAPSHOT_ARGS ?=
RECIPE ?=
TRIALS ?= 20
REFIT_SNAPSHOT ?=
EXPERIMENT ?=
LOAD_REQUESTS ?= 2000
LOAD_RATE ?= 20
LOAD_CONCURRENCY ?= 4

setup:
	$(COMPOSE) build data-jobs jobs api

setup-data:
	$(COMPOSE) build data-jobs

check-ingestion-image:
	$(COMPOSE) run --rm --no-deps --entrypoint python -v "$(CURDIR)/scripts/check_ingestion_image.py:/checks/check_ingestion_image.py:ro" data-jobs /checks/check_ingestion_image.py

up:
	$(COMPOSE) up -d --wait postgres init localstack mlflow api prometheus grafana
	$(COMPOSE) exec -T localstack /etc/localstack/init/ready.d/init.sh

db-init:
	$(COMPOSE) run --rm init

seed-source:
	$(COMPOSE) run --rm source-loader seed-source $(SEED_ARGS)

initial-snapshot:
	$(COMPOSE) run --rm data-jobs initial-snapshot $(SNAPSHOT_ARGS)

next-snapshot:
	$(COMPOSE) run --rm data-jobs next-snapshot $(SNAPSHOT_ARGS)

snapshots:
	$(COMPOSE) run --rm data-jobs snapshots list

verify-snapshot:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) run --rm data-jobs verify-snapshot --snapshot $(SNAPSHOT)

snapshot-report:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) run --rm data-jobs snapshots report $(SNAPSHOT) --output /state/reports/$(SNAPSHOT).html

refresh-snapshot-report:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) run --rm data-jobs snapshots report $(SNAPSHOT) --refresh --output /state/reports/$(SNAPSHOT).refreshed.html

bootstrap:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) run --rm jobs $@ --snapshot $(SNAPSHOT) $(if $(EXPERIMENT),--experiment $(EXPERIMENT))

train:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) exec -T localstack /etc/localstack/init/ready.d/init.sh
	$(COMPOSE) run --rm jobs train --snapshot $(SNAPSHOT) $(if $(RECIPE),--recipe $(RECIPE)) --trials $(TRIALS) $(if $(REFIT_SNAPSHOT),--refit-snapshot $(REFIT_SNAPSHOT)) $(if $(EXPERIMENT),--experiment $(EXPERIMENT))

# Recent training jobs with every attempt: running, succeeded, failed, or abandoned.
training-jobs:
	$(COMPOSE) run --rm jobs training-jobs

# A SageMaker job's input channels in S3, printed as its InputDataConfig.
sagemaker-inputs:
	@test -n "$(SNAPSHOT)" || (echo 'Set SNAPSHOT to the snapshot ID'; exit 1)
	$(COMPOSE) run --rm jobs sagemaker-inputs --snapshot $(SNAPSHOT) $(if $(RECIPE),--recipe $(RECIPE)) --trials $(TRIALS) $(if $(REFIT_SNAPSHOT),--refit-snapshot $(REFIT_SNAPSHOT))

# Verify a training job's model.tar.gz (an s3:// URI, or a path under /state) and add its release.
import-release:
	@test -n "$(ARCHIVE)" || (echo 'Set ARCHIVE to the model.tar.gz URI or path'; exit 1)
	$(COMPOSE) run --rm jobs import-release --archive $(ARCHIVE)

promote:
	@test -n "$(RELEASE)" || (echo 'Set RELEASE to the release ID'; exit 1)
	$(COMPOSE) run --rm jobs promote --release $(RELEASE) $(INITIAL) --reason "Manual operator promotion"
	@$(MAKE) --no-print-directory rollout

rollback:
	$(COMPOSE) run --rm jobs rollback --reason "Manual rollback demonstration"
	@$(MAKE) --no-print-directory rollout

# Restart the API on the selected release and confirm it serves it. On failure the
# last confirmed release is selected again, restarted, and confirmed; the command fails.
rollout:
	$(COMPOSE) restart api
	@$(COMPOSE) run --rm jobs verify-deployment || { \
		$(COMPOSE) restart api; \
		$(COMPOSE) run --rm jobs verify-deployment; \
		exit 1; }

status:
	$(COMPOSE) run --rm jobs status

# One simulated month: replay, next snapshot, monitoring, training. Stops before promotion.
month:
	$(COMPOSE) exec -T localstack /etc/localstack/init/ready.d/init.sh
	$(COMPOSE) run --rm jobs month --trials $(TRIALS) $(if $(SKIP_PROMOTION),--skip-promotion)

replay:
	$(COMPOSE) run --rm jobs replay

monitor:
	$(COMPOSE) run --rm jobs monitor

# Send a sample of the coming month to the running API at a fixed rate, then report
# client latency, errors, and server time per step. Watch it live in Grafana.
load:
	$(COMPOSE) run --rm jobs load --requests $(LOAD_REQUESTS) --rate $(LOAD_RATE) --concurrency $(LOAD_CONCURRENCY)

outcomes:
	@test -n "$(AS_OF)" || (echo 'Set AS_OF to the outcome cutoff date'; exit 1)
	$(COMPOSE) run --rm jobs outcomes --as-of $(AS_OF)

# Score the confirmed champion, and every release confirmed before it, on the reserved
# holdout month. Runs once: a second run refuses. SNAPSHOT overrides the latest snapshot.
final-evaluation:
	$(COMPOSE) run --rm jobs final-evaluation $(if $(SNAPSHOT),--snapshot $(SNAPSHOT))

# Start a new simulation. Empties the platform database's clock, ledgers, replays, and
# predictions, renames the MLflow experiment and registered model, and moves
# state/platform into its archive/ folder. The source import and S3 objects stay.
reset-simulation:
	@test "$(CONFIRM)" = yes || (echo 'This ends the current simulation. Run: make reset-simulation CONFIRM=yes'; exit 1)
	$(COMPOSE) run --rm jobs reset-simulation
	$(COMPOSE) restart api

test:
	$(COMPOSE) build tests
	$(COMPOSE) run --rm tests

unit:
	uv run --locked --all-extras pytest -m 'not integration'

# Terraform checks that need no AWS account: format, validation, mocked-provider
# tests, tflint, and checkov (the last two run in pinned containers).
TERRAFORM = terraform -chdir=infra/terraform
TFLINT = docker run --rm -v "$(CURDIR)/infra/terraform:/data" -w /data -e TFLINT_PLUGIN_DIR=/data/.tflint.d ghcr.io/terraform-linters/tflint:v0.64.0
infra-check:
	$(TERRAFORM) fmt -check -recursive
	$(TERRAFORM) init -backend=false -input=false -lockfile=readonly
	$(TERRAFORM) validate
	$(TERRAFORM) test
	$(TFLINT) --init
	$(TFLINT) --recursive
	docker run --rm -v "$(CURDIR)/infra/terraform:/tf" -w /tf bridgecrew/checkov:3.3.21 -d /tf --config-file /tf/.checkov.yaml

# Kubernetes: the same images on a local cluster through the Helm chart; see
# charts/fraud-risk-platform/README.md. Each platform command runs as a Job.
CHART = charts/fraud-risk-platform
HELM_RELEASE ?= fraud-risk-platform
K8S_NAMESPACE ?= fraud-risk
KIND_CLUSTER ?= kind
K8S = --namespace $(K8S_NAMESPACE)
K8S_TASK = HELM_RELEASE=$(HELM_RELEASE) K8S_NAMESPACE=$(K8S_NAMESPACE) ./scripts/k8s-task.sh
API_SELECTOR = app.kubernetes.io/instance=$(HELM_RELEASE),app.kubernetes.io/component=api
KUBECONFORM = docker run --rm -i ghcr.io/yannh/kubeconform:v0.6.7

# Lint the chart, and validate everything it renders against the Kubernetes schemas.
chart-check:
	helm lint --strict $(CHART)
	helm template $(HELM_RELEASE) $(CHART) | $(KUBECONFORM) -strict -summary -
	helm template $(HELM_RELEASE) $(CHART) --show-only templates/task-job.yaml --set-json 'task.args=["status"]' | $(KUBECONFORM) -strict -summary -

# Copy the images `make setup` built into a local kind cluster.
k8s-images:
	kind load docker-image fraud-risk-training:local fraud-risk-serving:local --name $(KIND_CLUSTER)

k8s-install:
	helm upgrade --install $(HELM_RELEASE) $(CHART) $(K8S) --create-namespace --timeout 10m

# Any platform command, for example: make k8s-task ARGS="train --snapshot <id>"
k8s-task:
	@test -n "$(ARGS)" || (echo 'Set ARGS to a platform command, for example ARGS=status'; exit 1)
	$(K8S_TASK) $(ARGS)

# One simulated month, started from the chart's CronJob as a one-off Job.
k8s-month:
	@cronjob=$$(kubectl get cronjob $(K8S) --selector app.kubernetes.io/instance=$(HELM_RELEASE),app.kubernetes.io/component=month --output jsonpath='{.items[0].metadata.name}'); \
	job=month-$$(date -u +%Y%m%d%H%M%S); \
	kubectl create job $(K8S) --from=cronjob/$$cronjob $$job && \
	kubectl logs $(K8S) --follow --pod-running-timeout=10m job/$$job && \
	kubectl wait $(K8S) --for=condition=Complete --timeout=60s job/$$job >/dev/null

k8s-promote:
	@test -n "$(RELEASE)" || (echo 'Set RELEASE to the release ID'; exit 1)
	$(K8S_TASK) promote --release $(RELEASE) $(INITIAL) --reason "Manual operator promotion"
	@$(MAKE) --no-print-directory k8s-rollout

k8s-rollback:
	$(K8S_TASK) rollback --reason "Manual rollback demonstration"
	@$(MAKE) --no-print-directory k8s-rollout

# Roll the API onto the selected release. The rolling update keeps the old pod serving
# until a new one is ready; if the new release is not served in time, the last confirmed
# release is selected again, rolled out, and confirmed, and the command fails.
k8s-rollout:
	kubectl rollout restart deployment $(K8S) --selector $(API_SELECTOR)
	@$(K8S_TASK) verify-deployment || { \
		kubectl rollout restart deployment $(K8S) --selector $(API_SELECTOR); \
		$(K8S_TASK) verify-deployment; \
		exit 1; }

# Removes the release; the database and state volumes stay, so a reinstall continues.
k8s-uninstall:
	helm uninstall $(HELM_RELEASE) $(K8S)

lint:
	uv run --locked --all-extras ruff check src tests scripts
	uv run --locked --all-extras ruff format --check src tests scripts

down:
	$(COMPOSE) --profile test --profile jobs down

logs:
	$(COMPOSE) logs --tail=80 api mlflow
