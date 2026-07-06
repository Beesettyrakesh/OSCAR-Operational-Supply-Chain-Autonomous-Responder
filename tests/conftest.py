"""
Shared pytest fixtures and configuration.

Forces the deterministic offline mode for every test (no live LLM calls, no API quota):
`GEMINI_API_KEY=""` selects the offline planner and `VENDOR_MODE=deterministic` selects the
scripted vendor. A helper fixture builds the standard SKU-99 incident on the shared store.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import pytest

# Set offline env before any project module is imported.
os.environ["GEMINI_API_KEY"] = ""
os.environ["VENDOR_MODE"] = "deterministic"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee deterministic offline mode for every test."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("VENDOR_MODE", "deterministic")


@pytest.fixture
def init_incident() -> Callable[..., Any]:
    """Return a helper that initializes the standard SKU-99 incident on the shared store.

    Mirrors the demo seed values; callers override `replacement_order_qty` / `delay_days`.
    """
    from ledger_store import STORE

    def _init(order_quantity: int = 500, delay_days: int = 9, **overrides: Any) -> Any:
        params: dict[str, Any] = {
            "target_sku": "SKU-99",
            "primary_supplier_id": "SUP-A",
            "active_contract_id": "CTR-4471",
            "current_purchase_order_id": "PO-88123",
            "impacted_plants": ["PLANT-2"],
            "inventory_days_remaining": 2,
            "production_shutdown_hours": 48,
            "revenue_at_risk_usd": 75000.0,
            "transferable_units": 350,
            "air_freight_available": True,
            "air_freight_capacity_units": 420,
            "replacement_order_qty": order_quantity,
            "delay_days": delay_days,
        }
        params.update(overrides)
        return STORE.init_incident(**params)

    return _init
