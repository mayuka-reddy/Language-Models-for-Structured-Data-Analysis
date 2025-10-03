"""Schema management and lightweight indexing for tables and columns.

This module provides a SchemaManager that can reflect a database schema
using SQLAlchemy, produce a compact schema graph brief, and return a
small top-k list of schema snippets suitable for prompt slots.
"""

from typing import Dict, List, Any, Optional
import os
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from loguru import logger


class SchemaManager:
    """Manages database schema information and generates small schema slices.

    Notes:
    - On construction this class will try to read DATABASE_URL from the
      environment and reflect the schema. For unit-testing you can
      construct the object and set `schema_cache` manually.
    - The top-k heuristic is intentionally simple: it prefers tables with
      more columns and includes basic column hints (type, pk/fk).
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.engine: Optional[Engine] = None
        self.schema_cache: Optional[Dict[str, Any]] = None

        # Try to initialize if a URL is available; do not raise here to
        # keep construction safe for offline code checks
        if self.database_url:
            try:
                self.engine = sa.create_engine(self.database_url)
                self.refresh_schema()
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to reflect DB schema at init: {}", exc)

    def refresh_schema(self) -> Dict[str, Any]:
        """Reflect the database schema into an internal cache.

        Returns the schema info dict.
        """
        if not self.engine:
            raise ValueError("No database engine available to refresh schema")

        metadata = sa.MetaData()
        metadata.reflect(bind=self.engine)

        schema_info: Dict[str, Any] = {"tables": {}, "relationships": []}

        for table_name, table in metadata.tables.items():
            schema_info["tables"][table_name] = {
                "columns": self._get_column_info(table),
                "primary_key": [c.name for c in table.primary_key],
                "foreign_keys": self._get_foreign_keys(table),
            }

        self.schema_cache = schema_info
        return schema_info

    def _get_column_info(self, table: sa.Table) -> Dict[str, Dict[str, Any]]:
        return {
            col.name: {
                "type": str(col.type),
                "nullable": bool(col.nullable),
                "primary_key": bool(col.primary_key),
                "foreign_key": bool(col.foreign_keys),
            }
            for col in table.columns
        }

    def _get_foreign_keys(self, table: sa.Table) -> List[Dict[str, str]]:
        fks: List[Dict[str, str]] = []
        for fk in table.foreign_keys:
            fks.append(
                {
                    "column": fk.parent.name,
                    "references_table": fk.column.table.name,
                    "references_column": fk.column.name,
                }
            )
        return fks

    def get_context(self, top_k: int = 5) -> Dict[str, Any]:
        """Return a context object suitable for prompt slots.

        The returned dict contains:
        - schema: full reflected schema cache
        - top_k_schema_items: a compact list of top-k table snippets
        - schema_graph_brief: string with FK relations
        """
        if not self.schema_cache:
            raise ValueError("Schema not initialized. Call refresh_schema first.")

        return {
            "schema": self.schema_cache,
            "top_k_schema_items": self._get_top_k_relevant(k=top_k),
            "schema_graph_brief": self._generate_schema_graph(),
        }

    def _get_top_k_relevant(self, k: int = 5) -> List[Dict[str, Any]]:
        """Return a small list of top-k schema snippets using a simple heuristic.

        Heuristic: prefer tables with more columns, and include up to 6 columns
        for each table with basic type hints. This is intentionally light-weight
        and deterministic for inclusion in prompts.
        """
        tables = self.schema_cache["tables"]

        # Rank tables by column count
        ranked = sorted(tables.items(), key=lambda t: len(t[1]["columns"]), reverse=True)

        snippets: List[Dict[str, Any]] = []
        for table_name, info in ranked[:k]:
            columns = info["columns"]
            col_preview = [{"name": n, "type": c["type"], "pk": c["primary_key"]} for n, c in list(columns.items())[:6]]
            snippets.append({"table": table_name, "columns": col_preview})

        return snippets

    def _generate_schema_graph(self) -> str:
        if not self.schema_cache:
            return ""

        lines: List[str] = []
        for table_name, table_info in self.schema_cache["tables"].items():
            for fk in table_info["foreign_keys"]:
                lines.append(f"{table_name}.{fk['column']} → {fk['references_table']}.{fk['references_column']}")

        return "\n".join(lines)

    # convenience helpers
    def list_tables(self) -> List[str]:
        return list(self.schema_cache["tables"].keys()) if self.schema_cache else []

    def get_table_columns(self, table: str) -> Dict[str, Any]:
        return self.schema_cache["tables"].get(table, {}).get("columns", {}) if self.schema_cache else {}