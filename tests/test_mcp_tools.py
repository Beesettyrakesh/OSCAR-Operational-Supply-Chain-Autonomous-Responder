"""MCP Observation Tool tests + verification of the 3-server split and facade."""

from __future__ import annotations

import mcp_server
import mcp_servers.erp_server as erp
import mcp_servers.inventory_server as inventory
import mcp_servers.logistics_server as logistics


def test_query_erp_hit() -> None:
    r = erp.query_erp("SKU-99")
    assert r["found"] is True
    assert r["primary_supplier_id"] == "SUP-A"
    assert r["unit_cost_usd"] == 42.50
    assert r["active_contract_id"] == "CTR-4471"


def test_query_erp_miss() -> None:
    assert erp.query_erp("NOPE")["found"] is False


def test_query_inventory() -> None:
    r = inventory.query_inventory("SKU-99")
    assert r["found"] is True
    assert r["inventory_days_remaining"] == 2
    assert "PLANT-2" in r["plant_balances"]


def test_query_shipment_tracking() -> None:
    r = logistics.query_shipment_tracking("PO-88123")
    assert r["found"] is True
    assert r["status"] == "DELAYED"
    assert r["delay_days"] == 9


def test_extract_contract_rules() -> None:
    r = erp.extract_contract_rules("CTR-4471")
    assert r["found"] is True
    assert r["contracted_penalty_rate"] == 0.03


def test_query_alternate_suppliers() -> None:
    alts = erp.query_alternate_suppliers("SKU-99")
    by_id = {a["supplier_id"]: a for a in alts}
    assert set(by_id) == {"SUP-B", "SUP-C"}
    assert by_id["SUP-C"]["unit_cost_usd"] == 44.00
    assert by_id["SUP-C"]["quoted_lead_time_days"] == 6
    assert by_id["SUP-B"]["unit_cost_usd"] == 46.75


def test_three_server_split_and_facade() -> None:
    """Each category server owns its tools; the facade re-exports all five (same callables)."""
    # Category ownership.
    assert hasattr(erp, "query_erp") and hasattr(erp, "query_alternate_suppliers")
    assert hasattr(erp, "extract_contract_rules")
    assert hasattr(inventory, "query_inventory")
    assert hasattr(logistics, "query_shipment_tracking")
    # Facade re-exports the same functions.
    assert mcp_server.query_erp is erp.query_erp
    assert mcp_server.query_inventory is inventory.query_inventory
    assert mcp_server.query_shipment_tracking is logistics.query_shipment_tracking
    assert mcp_server.extract_contract_rules is erp.extract_contract_rules
    assert mcp_server.query_alternate_suppliers is erp.query_alternate_suppliers
