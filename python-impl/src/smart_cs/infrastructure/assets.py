from __future__ import annotations

from pathlib import Path
import re
from uuid import uuid4


class LocalAssetStorage:
    """Store uploaded evidence separately from indexed Markdown knowledge."""

    ALLOWED_TYPES = {"image/jpeg": ".jpg", "image/png": ".png"}
    MAX_IMAGE_BYTES = 5 * 1024 * 1024

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(
        self, conversation_id: str, filename: str, content_type: str, content: bytes
    ) -> str:
        if content_type not in self.ALLOWED_TYPES:
            raise ValueError("Only JPEG and PNG evidence images are accepted")
        if not content or len(content) > self.MAX_IMAGE_BYTES:
            raise ValueError("Evidence image must be between 1 byte and 5 MB")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", conversation_id) or conversation_id in {".", ".."}:
            raise ValueError("Invalid conversation asset scope")
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).stem).strip("._") or "image"
        suffix = self.ALLOWED_TYPES[content_type]
        key = f"{conversation_id}/{uuid4().hex}-{stem}{suffix}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return key
