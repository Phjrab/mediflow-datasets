"""Atomic progress writes for new adaptation experiments, preserving embedded legacy code."""

import json
import time
import uuid
from pathlib import Path


def write_progress(path, value):
    path = Path(path)
    temporary = path.with_name(".json-" + uuid.uuid4().hex[:12] + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))
