"""
tools/
======
Observation-tool support package for the Autonomous Supply Chain Incident Commander.

Houses the deterministic mock enterprise datasets (ERP / inventory / shipment) that the
MCP Observation Tools query. Separated from `mcp_server.py` so the tool *transport* layer
stays decoupled from the mock *data* layer (Section 4.1).
"""

from tools.mock_data import ERP_DB, INVENTORY_DB, SHIPMENT_DB

__all__ = ["ERP_DB", "INVENTORY_DB", "SHIPMENT_DB"]
