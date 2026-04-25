"""Context snapshot capture, storage, and serialization."""
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import tiktoken

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class ContextSnapshot:
    """A captured snapshot of an LLM context at a point in time."""

    snapshot_id: str          # unique UUID
    captured_at: str          # ISO 8601 UTC timestamp
    messages: list[dict]      # raw messages list (Anthropic/OpenAI format)
    token_count: int          # total tokens across all message content
    metadata: dict = field(default_factory=dict)


# ── Store ────────────────────────────────────────────────────────────────────

class SnapshotStore:
    """In-memory store for ContextSnapshot objects with serialization support."""

    def __init__(self) -> None:
        """Initialise an empty snapshot store."""
        self._store: dict[str, ContextSnapshot] = {}

    def capture(
        self, messages: list[dict], metadata: dict | None = None
    ) -> ContextSnapshot:
        """Capture and store a snapshot from a messages list.

        Args:
            messages: List of message dicts, each with 'role' and 'content'.
            metadata: Optional arbitrary metadata to attach to the snapshot.

        Returns:
            The stored ContextSnapshot.
        """
        snapshot_id = str(uuid.uuid4())
        token_count = _count_messages_tokens(messages)
        snapshot = ContextSnapshot(
            snapshot_id=snapshot_id,
            captured_at=datetime.now(timezone.utc).isoformat(),
            messages=messages,
            token_count=token_count,
            metadata=metadata or {},
        )
        self._store[snapshot_id] = snapshot
        logger.debug("Captured snapshot %s (%d tokens)", snapshot_id, token_count)
        return snapshot

    def get(self, snapshot_id: str) -> ContextSnapshot:
        """Retrieve a snapshot by its ID.

        Raises:
            KeyError: If snapshot_id is not in the store.
        """
        if snapshot_id not in self._store:
            raise KeyError(f"Snapshot {snapshot_id!r} not found in store")
        return self._store[snapshot_id]

    def list_all(self) -> list[ContextSnapshot]:
        """Return all stored snapshots in insertion order."""
        return list(self._store.values())

    def clear(self) -> None:
        """Remove all snapshots from the store."""
        self._store.clear()
        logger.debug("Snapshot store cleared")

    @staticmethod
    def to_dict(snapshot: ContextSnapshot) -> dict:
        """Serialize a snapshot to a JSON-safe dict.

        Args:
            snapshot: The ContextSnapshot to serialize.

        Returns:
            A plain dict suitable for json.dumps.
        """
        return {
            "snapshot_id": snapshot.snapshot_id,
            "captured_at": snapshot.captured_at,
            "messages": snapshot.messages,
            "token_count": snapshot.token_count,
            "metadata": snapshot.metadata,
        }

    @staticmethod
    def from_dict(data: dict) -> ContextSnapshot:
        """Deserialize a snapshot from a dict (e.g. loaded from JSON).

        Args:
            data: Dict previously produced by to_dict.

        Returns:
            A reconstructed ContextSnapshot.

        Raises:
            ValueError: If any required keys are missing from data.
        """
        required = {"snapshot_id", "captured_at", "messages", "token_count"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Snapshot dict missing required keys: {missing}")
        return ContextSnapshot(
            snapshot_id=data["snapshot_id"],
            captured_at=data["captured_at"],
            messages=data["messages"],
            token_count=data["token_count"],
            metadata=data.get("metadata", {}),
        )


# ── Internal helper ───────────────────────────────────────────────────────────

def _count_messages_tokens(messages: list[dict]) -> int:
    """Count tokens across all message content strings in a messages list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(_enc.encode(content))
        elif isinstance(content, list):
            # Anthropic block format: [{"type": "text", "text": "..."}]
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(_enc.encode(block["text"]))
    return total
