"""The base contract and the as-of date convention."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


def midnight(value: str | date) -> datetime:
    """An as-of DATE always means its start (00:00 UTC), never its end."""
    return datetime.combine(date.fromisoformat(str(value)), datetime.min.time())
