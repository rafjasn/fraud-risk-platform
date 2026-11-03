"""The CLI returns each command's result, or its error, to a waiting workflow."""

import json
import sys

import boto3
import pytest

from fraud_risk_platform import cli


@pytest.fixture
def step_functions(monkeypatch):
    calls = []

    class Client:
        def send_task_success(self, **request):
            calls.append(("success", request))

        def send_task_failure(self, **request):
            calls.append(("failure", request))

    monkeypatch.setattr(boto3, "client", lambda service: Client())
    monkeypatch.setattr(sys, "argv", ["fraud-risk-platform", "status"])
    return calls


def test_without_a_task_token_nothing_is_reported(step_functions, monkeypatch):
    monkeypatch.delenv("STEP_FUNCTIONS_TASK_TOKEN", raising=False)
    monkeypatch.setattr(cli, "run", lambda args: {"status": "ok"})
    cli.main()
    assert step_functions == []


def test_a_task_token_receives_the_result_or_the_error(step_functions, monkeypatch):
    monkeypatch.setenv("STEP_FUNCTIONS_TASK_TOKEN", "token-1")
    result = {"status": "ready_to_train", "snapshot_id": "s" * 64}
    monkeypatch.setattr(cli, "run", lambda args: result)
    cli.main()
    assert step_functions == [
        ("success", {"taskToken": "token-1", "output": json.dumps(result)})
    ]

    def refused(args):
        raise ValueError("The last cycle recommends promoting abc")

    monkeypatch.setattr(cli, "run", refused)
    with pytest.raises(ValueError, match="recommends promoting"):
        cli.main()
    assert step_functions[1] == (
        "failure",
        {
            "taskToken": "token-1",
            "error": "ValueError",
            "cause": "The last cycle recommends promoting abc",
        },
    )
