"""Which release serves: deployment.json, promotion, rollout confirmation, rollback."""

import fcntl
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from uuid import uuid4

from fraud_risk_platform.models.releases import load_release
from fraud_risk_platform.persistence.files import read_json, safe_id, sha256, write_json


def deployment(root: Path) -> dict:
    path = root / "deployment.json"
    return (
        read_json(path)
        if path.exists()
        else {"schema_version": 1, "current": None, "deployed": None, "history": []}
    )


@contextmanager
def deployment_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".deployment.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _publish(root: Path, state: dict, target: str, action: str, reason: str) -> dict:
    state["history"].append(
        {
            "from": state["current"],
            "to": target,
            "action": action,
            "reason": reason,
            "at": _now(),
        }
    )
    state["current"] = target
    return _write_state(root, state)


def _write_state(root: Path, state: dict) -> dict:
    temporary = root / f".deployment-{uuid4().hex}.json"
    try:
        write_json(temporary, state)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, root / "deployment.json")
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return state


def promote(root: Path, release_id: str, *, initial: bool, reason: str) -> dict:
    with deployment_lock(root):
        state = deployment(root)
        if initial != (state["current"] is None):
            raise ValueError("Use --initial only for the first deployment")
        manifest, _ = load_release(root, release_id)
        cycle_dir = root / "cycles" / safe_id(manifest["source_run_id"])
        completed = read_json(cycle_dir / "complete.json")
        report = read_json(cycle_dir / "report.json")
        if (
            completed["report_sha256"] != sha256(cycle_dir / "report.json")
            or report["release_id"] != release_id
            or report["recommendation"] not in {"promote", "initial_promotion"}
        ):
            raise ValueError(
                "Release is not backed by a completed, successful training report"
            )
        if not manifest["promotion_eligible"]:
            raise ValueError(
                "Evaluation recommends keeping the incumbent; candidate cannot be promoted"
            )
        if manifest["incumbent_release_id"] != state["current"]:
            raise ValueError(
                "Stale recommendation: deployed incumbent changed; run a new cycle"
            )
        return _publish(
            root, state, release_id, "initial" if initial else "promote", reason
        )


def confirm_rollout(
    root: Path,
    probe: Callable[[], str | None],
    *,
    timeout: float = 120.0,
    interval: float = 2.0,
    clock: Callable[[], float] = monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Mark the selected release deployed once the API serves it, else restore.

    `probe()` returns the release the API reports as ready, or None. A ready API has
    verified the release's files and runtime versions and reproduced its stored
    sample scores at load. "current" is the release selected for serving; "deployed"
    is the last one confirmed serving. If the selected release is not served before
    the timeout, the last confirmed release is selected again, and the caller restarts
    the API on it and confirms it.
    """
    with deployment_lock(root):
        state = deployment(root)
        selected = state["current"]
        if selected is None:
            raise ValueError("No release is selected for serving")
        deadline = clock() + timeout
        served = probe()
        while served != selected and clock() < deadline:
            sleep(interval)
            served = probe()
        if served == selected:
            state["deployed"] = selected
            state["history"].append(
                {"to": selected, "action": "confirmed", "at": _now()}
            )
            _write_state(root, state)
            return {"status": "deployed", "release_id": selected}
        restored = (
            state["deployed"]
            if "deployed" in state
            # Records from before confirmation existed: restore the release that
            # was selected before this one.
            else next(
                (e["from"] for e in reversed(state["history"]) if e["to"] == selected),
                None,
            )
        )
        _publish(
            root,
            state,
            restored,
            "rollout_failed",
            f"API served {served or 'no release'} instead of {selected} "
            f"within {timeout:g} seconds",
        )
        return {
            "status": "rollout_failed",
            "selected": selected,
            "served": served,
            "restored": restored,
        }


def _rollback_targets(state: dict) -> list[str]:
    """Releases confirmed serving, oldest first. Histories that predate confirmation
    count every release they selected."""
    confirmed = [e["to"] for e in state["history"] if e["action"] == "confirmed"]
    if confirmed:
        return confirmed
    return [
        e["to"]
        for e in state["history"]
        if e["action"] in {"initial", "promote", "rollback"}
    ]


def rollback(root: Path, *, reason: str, target: str | None = None) -> dict:
    with deployment_lock(root):
        state = deployment(root)
        candidates = [r for r in _rollback_targets(state) if r != state["current"]]
        if target is None:
            if not candidates:
                raise ValueError("No previous deployment to roll back to")
            target = candidates[-1]
        if target not in candidates:
            raise ValueError(
                "Rollback target must be a different, previously deployed release"
            )
        load_release(root, target)
        return _publish(root, state, target, "rollback", reason)
