{{/* Chart name, truncated to Kubernetes' 63-character limit. */}}
{{- define "frp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Release-qualified name; the release name alone when it already contains the chart name. */}}
{{- define "frp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "frp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Labels and selector labels for one component. Call with (dict "ctx" $ "component" "api"). */}}
{{- define "frp.labels" -}}
helm.sh/chart: {{ include "frp.chart" .ctx }}
{{ include "frp.selectorLabels" . }}
app.kubernetes.io/version: {{ .ctx.Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .ctx.Release.Service }}
app.kubernetes.io/part-of: fraud-risk-platform
{{- end }}

{{- define "frp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "frp.name" .ctx }}
app.kubernetes.io/instance: {{ .ctx.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "frp.image" -}}
{{- printf "%s:%s" .repository .tag }}
{{- end }}

{{- define "frp.secretName" -}}
{{- .Values.credentials.existingSecret | default (include "frp.fullname" .) }}
{{- end }}

{{- define "frp.postgresHost" -}}{{ include "frp.fullname" . }}-postgres{{- end }}
{{- define "frp.localstackHost" -}}{{ include "frp.fullname" . }}-localstack{{- end }}
{{- define "frp.mlflowHost" -}}{{ include "frp.fullname" . }}-mlflow{{- end }}
{{- define "frp.apiHost" -}}{{ include "frp.fullname" . }}-api{{- end }}

{{/* The project's images run as uid 1000 without privileges or a writable root filesystem. */}}
{{- define "frp.podSecurityContext" -}}
runAsNonRoot: true
runAsUser: 1000
runAsGroup: 1000
fsGroup: 1000
seccompProfile:
  type: RuntimeDefault
{{- end }}

{{- define "frp.containerSecurityContext" -}}
allowPrivilegeEscalation: false
readOnlyRootFilesystem: true
capabilities:
  drop:
    - ALL
{{- end }}

{{/*
Database connection URLs, built from the Secret's passwords by Kubernetes' $(VAR)
expansion, so no password appears in a ConfigMap or manifest.
Call with (dict "ctx" $ "sourceRole" "source_app") or "source_reader" for read-only access.
*/}}
{{- define "frp.databaseEnv" -}}
{{- $secret := include "frp.secretName" .ctx -}}
{{- $host := include "frp.postgresHost" .ctx -}}
- name: PLATFORM_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: platform-app-password
- name: SOURCE_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ $secret }}
      key: {{ ternary "source-reader-password" "source-app-password" (eq .sourceRole "source_reader") }}
- name: DATABASE_URL
  value: "postgresql://platform_app:$(PLATFORM_DB_PASSWORD)@{{ $host }}:5432/fraud_platform?connect_timeout=5"
- name: SOURCE_DATABASE_URL
  value: "postgresql://{{ .sourceRole }}:$(SOURCE_DB_PASSWORD)@{{ $host }}:5432/fraud_source?connect_timeout=5"
{{- end }}

{{- define "frp.awsEnv" -}}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "frp.secretName" . }}
      key: aws-access-key-id
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "frp.secretName" . }}
      key: aws-secret-access-key
{{- end }}

{{/*
Waits until PostgreSQL accepts connections, so dependants start cleanly instead of crash-looping.
The user name is required: uid 1000 has no account in the postgres image, and without one
pg_isready makes no attempt to connect.
*/}}
{{- define "frp.waitForPostgres" -}}
- name: wait-for-postgres
  image: {{ include "frp.image" .Values.images.postgres }}
  imagePullPolicy: {{ .Values.images.postgres.pullPolicy }}
  command:
    - sh
    - -c
    - until pg_isready --host={{ include "frp.postgresHost" . }} --port=5432 --username=fraud --dbname=fraud --timeout=2; do sleep 2; done
  resources:
    requests:
      cpu: 10m
      memory: 16Mi
    limits:
      memory: 64Mi
  securityContext:
    {{- include "frp.containerSecurityContext" . | nindent 4 }}
{{- end }}

{{/*
The pod that runs one platform command, shared by the setup hook, the monthly CronJob,
and on-demand tasks. Call with (dict "ctx" $ "args" (list "month")).
*/}}
{{- define "frp.commandPodSpec" -}}
{{- $ctx := .ctx -}}
serviceAccountName: {{ include "frp.fullname" $ctx }}
restartPolicy: Never
securityContext:
  {{- include "frp.podSecurityContext" $ctx | nindent 2 }}
initContainers:
  {{- include "frp.waitForPostgres" $ctx | nindent 2 }}
containers:
  - name: command
    image: {{ include "frp.image" $ctx.Values.images.training }}
    imagePullPolicy: {{ $ctx.Values.images.training.pullPolicy }}
    args:
      {{- range .args }}
      - {{ toString . | quote }}
      {{- end }}
    envFrom:
      - configMapRef:
          name: {{ include "frp.fullname" $ctx }}
    env:
      {{- include "frp.databaseEnv" (dict "ctx" $ctx "sourceRole" "source_app") | nindent 6 }}
      {{- include "frp.awsEnv" $ctx | nindent 6 }}
    resources:
      {{- toYaml $ctx.Values.jobs.resources | nindent 6 }}
    securityContext:
      {{- include "frp.containerSecurityContext" $ctx | nindent 6 }}
    volumeMounts:
      - name: state
        mountPath: /state
      - name: raw-data
        mountPath: /app/data/raw
      - name: tmp
        mountPath: /tmp
volumes:
  - name: state
    persistentVolumeClaim:
      claimName: {{ include "frp.fullname" $ctx }}-state
  # The downloaded IBM archive; seeding downloads it again if the pod is replaced.
  - name: raw-data
    emptyDir: {}
  - name: tmp
    emptyDir: {}
{{- end }}
