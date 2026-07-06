"""Agent CLI tests — exit codes and the JSON ledger dump (offline)."""

from __future__ import annotations

import json

import agent_cli


def test_cli_within_authority_exit_0() -> None:
    rc = agent_cli.main(["--qty", "440", "--hitl", "approve", "--no-color"])
    assert rc == 0


def test_cli_reject_exit_1() -> None:
    rc = agent_cli.main(["--qty", "500", "--hitl", "reject", "--no-color"])
    assert rc == 1


def test_cli_json_dump_is_valid_ledger(capsys) -> None:  # type: ignore[no-untyped-def]
    rc = agent_cli.main(["--qty", "300", "--hitl", "approve", "--no-color", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    # The final block prints the State Ledger JSON; parse the last JSON object out of it.
    start = out.rfind("{", 0, out.find('"metadata"'))
    ledger = json.loads(out[start:])
    assert ledger["context"]["target_sku"] == "SKU-99"
    assert ledger["mitigation"]["active_strategy"] == "INTERNAL_TRANSFER"
