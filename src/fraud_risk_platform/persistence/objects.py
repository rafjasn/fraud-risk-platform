"""S3 objects with conditional publication and verified bounded downloads."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from fraud_risk_platform.persistence.files import canonical, sha256


class ObjectStore:
    def __init__(self, uri: str | None = None, *, client=None):
        location = urlsplit(uri or os.environ.get("FRAUD_DATA_URI", "s3://fraud-data"))
        if (
            location.scheme != "s3"
            or not location.netloc
            or location.query
            or location.fragment
        ):
            raise ValueError("Object storage must be an s3://bucket[/prefix] URI")
        self.bucket = location.netloc
        self.prefix = location.path.strip("/")
        self.client = client or boto3.client(
            "s3",
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )

    def key(self, key: str) -> str:
        if (
            not key
            or key.startswith("/")
            or "\\" in key
            or any(p in {".", ".."} for p in key.split("/"))
        ):
            raise ValueError("Unsafe object key")
        return f"{self.prefix}/{key}" if self.prefix else key

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self.key(key)}"

    def read_json(self, key: str) -> dict:
        response = self.client.get_object(Bucket=self.bucket, Key=self.key(key))
        with response["Body"] as body:
            if response["ContentLength"] > 32 * 1024 * 1024:
                raise ValueError("Manifest exceeds the supported size")
            return json.loads(body.read())

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self.key(key))
            return True
        except ClientError as error:
            if error.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def _put(self, key: str, body, digest: str, size: int, content_type: str):
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=self.key(key),
                Body=body,
                ContentLength=size,
                ContentType=content_type,
                Metadata={"sha256": digest},
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
                raise
            self.verify({"key": key, "sha256": digest, "bytes": size})
        return {"key": key, "sha256": digest, "bytes": size}

    def put_json(self, key: str, value: dict):
        data = canonical(value)
        return self._put(
            key, data, hashlib.sha256(data).hexdigest(), len(data), "application/json"
        )

    def put_file(self, key: str, path: Path, content_type="application/octet-stream"):
        with path.open("rb") as stream:
            return self._put(
                key, stream, sha256(path), path.stat().st_size, content_type
            )

    def verify(self, descriptor: dict, destination: Path | None = None):
        response = self.client.get_object(
            Bucket=self.bucket, Key=self.key(descriptor["key"])
        )
        digest = hashlib.sha256()
        count = 0
        output = destination.open("wb") if destination is not None else None
        try:
            with response["Body"] as body:
                while block := body.read(1024 * 1024):
                    count += len(block)
                    digest.update(block)
                    if output:
                        output.write(block)
            if (
                count != descriptor["bytes"]
                or digest.hexdigest() != descriptor["sha256"]
            ):
                raise ValueError(f"Object checksum/size mismatch: {descriptor['key']}")
        finally:
            if output:
                output.close()

    def list_manifests(self, prefix: str):
        for page in self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=self.key(prefix)
        ):
            for item in page.get("Contents", []):
                if PurePosixPath(item["Key"]).name == "manifest.json":
                    yield (
                        item["Key"][len(self.prefix) + 1 :]
                        if self.prefix
                        else item["Key"]
                    )
