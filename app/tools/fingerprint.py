import hashlib
import json
from typing import Any


def action_fingerprint(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    payload = {
        "tool_name": tool_name,
        "arguments": arguments,
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
