import os
import sys
import pytest

# Ensure project root is on sys.path so tests can import the `src` package.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.schema.manager import SchemaManager


def test_schema_manager_offline_cache():
    # Create manager without DATABASE_URL (no reflection)
    mgr = SchemaManager(database_url=None)

    # Manually set a small schema cache to simulate reflection
    mgr.schema_cache = {
        "tables": {
            "customer": {
                "columns": {
                    "customer_id": {"type": "INTEGER", "primary_key": True, "nullable": False, "foreign_key": False},
                    "city": {"type": "VARCHAR", "primary_key": False, "nullable": True, "foreign_key": False},
                },
                "primary_key": ["customer_id"],
                "foreign_keys": [],
            },
            "orders": {
                "columns": {
                    "order_id": {"type": "INTEGER", "primary_key": True, "nullable": False, "foreign_key": False},
                    "customer_id": {"type": "INTEGER", "primary_key": False, "nullable": False, "foreign_key": True},
                    "order_date": {"type": "DATE", "primary_key": False, "nullable": True, "foreign_key": False},
                },
                "primary_key": ["order_id"],
                "foreign_keys": [{"column": "customer_id", "references_table": "customer", "references_column": "customer_id"}],
            }
        }
    }

    ctx = mgr.get_context(top_k=2)
    assert "schema" in ctx
    assert "top_k_schema_items" in ctx
    assert isinstance(ctx["top_k_schema_items"], list)
    assert any(item["table"] == "customer" for item in ctx["top_k_schema_items"]) or any(item["table"] == "orders" for item in ctx["top_k_schema_items"]) 