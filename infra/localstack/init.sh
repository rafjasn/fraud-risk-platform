#!/bin/sh
set -eu
for bucket in "${FRAUD_DATA_BUCKET:-fraud-data}" "${FRAUD_MLFLOW_BUCKET:-fraud-mlflow}"; do
    if ! awslocal s3api head-bucket --bucket "$bucket" >/dev/null 2>&1; then
        awslocal s3api create-bucket --bucket "$bucket"
    fi
done
