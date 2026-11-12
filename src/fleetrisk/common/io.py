import hashlib
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID


def json_default(value: Any) -> str:
    if isinstance(value, (datetime, date, Decimal, UUID, Path)):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def dumps(value: Any) -> str:
    return json.dumps(value, default=json_default, allow_nan=False, sort_keys=True)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
