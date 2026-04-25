"""Tests for context_lens.engine.snapshots — no API calls."""
import pytest

from context_lens.engine.snapshots import ContextSnapshot, SnapshotStore


SAMPLE_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


# ── SnapshotStore.capture ─────────────────────────────────────────────────────

def test_capture_returns_snapshot():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    assert isinstance(snap, ContextSnapshot)


def test_capture_assigns_unique_ids():
    store = SnapshotStore()
    s1 = store.capture(SAMPLE_MESSAGES)
    s2 = store.capture(SAMPLE_MESSAGES)
    assert s1.snapshot_id != s2.snapshot_id


def test_capture_stores_messages():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    assert snap.messages == SAMPLE_MESSAGES


def test_capture_counts_tokens_nonzero():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    assert snap.token_count > 0


def test_capture_empty_messages_token_count_zero():
    store = SnapshotStore()
    snap = store.capture([])
    assert snap.token_count == 0


def test_capture_metadata_stored():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES, metadata={"step": "retrieval", "run_id": "42"})
    assert snap.metadata["step"] == "retrieval"
    assert snap.metadata["run_id"] == "42"


def test_capture_metadata_defaults_to_empty_dict():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    assert snap.metadata == {}


def test_capture_has_iso_timestamp():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    assert "T" in snap.captured_at   # ISO 8601 contains "T"
    assert snap.captured_at.endswith("+00:00") or snap.captured_at.endswith("Z")


# ── SnapshotStore.get ─────────────────────────────────────────────────────────

def test_get_returns_same_snapshot():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    retrieved = store.get(snap.snapshot_id)
    assert retrieved is snap


def test_get_missing_id_raises_key_error():
    store = SnapshotStore()
    with pytest.raises(KeyError, match="not found"):
        store.get("nonexistent-id")


# ── SnapshotStore.list_all ────────────────────────────────────────────────────

def test_list_all_empty_store():
    store = SnapshotStore()
    assert store.list_all() == []


def test_list_all_returns_all_snapshots():
    store = SnapshotStore()
    s1 = store.capture(SAMPLE_MESSAGES)
    s2 = store.capture(SAMPLE_MESSAGES)
    all_snaps = store.list_all()
    assert len(all_snaps) == 2
    assert s1 in all_snaps
    assert s2 in all_snaps


# ── SnapshotStore.clear ───────────────────────────────────────────────────────

def test_clear_empties_store():
    store = SnapshotStore()
    store.capture(SAMPLE_MESSAGES)
    store.clear()
    assert store.list_all() == []


def test_get_after_clear_raises():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    store.clear()
    with pytest.raises(KeyError):
        store.get(snap.snapshot_id)


# ── Serialization round-trip ──────────────────────────────────────────────────

def test_to_dict_contains_required_keys():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES)
    d = SnapshotStore.to_dict(snap)
    assert "snapshot_id" in d
    assert "captured_at" in d
    assert "messages" in d
    assert "token_count" in d
    assert "metadata" in d


def test_round_trip_serialization():
    store = SnapshotStore()
    snap = store.capture(SAMPLE_MESSAGES, metadata={"k": "v"})
    d = SnapshotStore.to_dict(snap)
    restored = SnapshotStore.from_dict(d)
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.captured_at == snap.captured_at
    assert restored.messages == snap.messages
    assert restored.token_count == snap.token_count
    assert restored.metadata == snap.metadata


def test_from_dict_missing_key_raises():
    bad = {"snapshot_id": "x", "captured_at": "t", "messages": []}  # missing token_count
    with pytest.raises(ValueError, match="missing required keys"):
        SnapshotStore.from_dict(bad)


def test_from_dict_metadata_defaults_to_empty():
    d = {
        "snapshot_id": "abc",
        "captured_at": "2026-01-01T00:00:00+00:00",
        "messages": [],
        "token_count": 0,
        # no 'metadata' key
    }
    snap = SnapshotStore.from_dict(d)
    assert snap.metadata == {}


# ── Token counting for block-format content ───────────────────────────────────

def test_capture_block_format_content_counts_tokens():
    """Anthropic block content format: [{"type": "text", "text": "..."}]."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello from block format."}],
        }
    ]
    store = SnapshotStore()
    snap = store.capture(messages)
    assert snap.token_count > 0
