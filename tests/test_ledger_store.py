"""State Mutation Layer tests: deep-merge, validation rejection, snapshot isolation, logging."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ledger_store import STORE, LedgerStore



def test_mutate_deep_merge_preserves_siblings(init_incident) -> None:  # type: ignore[no-untyped-def]
    init_incident()
    STORE.mutate({"mitigation": {"strategy_scores": {"ALT_SUPPLIER": 60.35}}})
    STORE.mutate({"mitigation": {"strategy_scores": {"AIR_FREIGHT": 65.28}}})
    scores = STORE.snapshot().mitigation.strategy_scores
    assert scores == {"ALT_SUPPLIER": 60.35, "AIR_FREIGHT": 65.28}


def test_mutate_rejects_bad_primitive(init_incident) -> None:  # type: ignore[no-untyped-def]
    init_incident()
    with pytest.raises(ValidationError):
        STORE.mutate({"metadata": {"loop_count": 99}})  # le=11


def test_snapshot_is_deep_copy(init_incident) -> None:  # type: ignore[no-untyped-def]
    init_incident()
    snap = STORE.snapshot()
    snap.metrics.revenue_at_risk_usd = 1.0  # mutate the copy
    assert STORE.snapshot().metrics.revenue_at_risk_usd == 75000.0


def test_revision_and_increment_loop(init_incident) -> None:  # type: ignore[no-untyped-def]
    init_incident()
    r0 = STORE.revision
    STORE.increment_loop()
    assert STORE.revision == r0 + 1
    assert STORE.snapshot().metadata.loop_count == 1


def test_append_raw_log_strips_control_chars(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CR/LF in adversarial input must not forge extra log lines."""
    log_file = tmp_path / "exec.log"
    monkeypatch.setattr("ledger_store.RAW_LOG_PATH", log_file)
    LedgerStore.append_raw_log("src", "line1\r\nFORGED line2")
    contents = log_file.read_text(encoding="utf-8")
    # Exactly one record line (the only newline is the trailing one the writer adds).
    assert contents.count("\n") == 1
    assert "FORGED" in contents  # content preserved, but on the same line
