"""NL to SQL generation model and prompt management."""
# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import json
from loguru import logger


class DummyModel:
    """A tiny deterministic model used for tests and local validation.

    It maps a few known questions to canned JSON responses. This allows
    the rest of the code (prompting, parsing, execution plumbing) to be
    exercised without loading a large LLM during development.
    """

    def generate(self, prompt: str) -> str:
        # Very small heuristic-based responses for the sample retail schema
        if "Which city has the highest number of customers" in prompt:
            return json.dumps({
                "sql": "SELECT city, COUNT(*) as customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1",
                "used_tables": ["customer"],
                "used_columns": ["city"],
                "reason_short": "Count customers per city and pick the top",
                "confidence": 0.95
            })
        if "What was the total sales in August 2025" in prompt:
            return json.dumps({
                "sql": "SELECT SUM(total_amount) as total_sales FROM orders WHERE order_date >= '2025-08-01' AND order_date < '2025-09-01'",
                "used_tables": ["orders"],
                "used_columns": ["order_date", "total_amount"],
                "reason_short": "Sum total_amount for orders in August 2025",
                "confidence": 0.92
            })
        # default fallback: return an empty JSON shell
        return json.dumps({
            "sql": "SELECT 1 as one",
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Fallback response",
            "confidence": 0.1
        })


class NL2SQLGenerator:
    """Handles natural language to SQL conversion using a pluggable model.

    By default this will use DummyModel for quick local development. When a
    real model is required, pass an object with a `generate(prompt: str) -> str`
    method that returns the model's full text completion.
    """

    def __init__(self, model: Optional[Any] = None):
        self.model = model or DummyModel()
        self.few_shots = self._load_few_shots()

    def _load_few_shots(self) -> str:
        """Return compact few-shots covering joins, dates, and ranking."""
        return """
# Example 1: Simple aggregation with grouping
Q: Which city has the highest number of customers?
A: {
  "sql": "SELECT city, COUNT(*) AS customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1",
  "used_tables": ["customer"],
  "used_columns": ["city"],
  "reason_short": "Count customers per city, order by count descending",
  "confidence": 0.95
}

# Example 2: Date filtering with aggregation
Q: What was the total sales in August 2025?
A: {
  "sql": "SELECT SUM(total_amount) AS total_sales FROM orders WHERE order_date >= '2025-08-01' AND order_date < '2025-09-01'",
  "used_tables": ["orders"],
  "used_columns": ["order_date", "total_amount"],
  "reason_short": "Sum order amounts for August 2025",
  "confidence": 0.92
}

# Example 3: Join with aggregation and ranking
Q: List the top 5 products by revenue this quarter.
A: {
  "sql": "WITH product_revenue AS (SELECT p.product_id, p.name, SUM(oi.quantity * p.price) AS revenue FROM product p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id WHERE o.order_date >= DATE_TRUNC('quarter', CURRENT_DATE) GROUP BY p.product_id, p.name) SELECT * FROM product_revenue ORDER BY revenue DESC LIMIT 5",
  "used_tables": ["product", "order_items", "orders"],
  "used_columns": ["product_id", "name", "quantity", "price", "order_date"],
  "reason_short": "Join orders and products, compute revenue, filter to current quarter",
  "confidence": 0.88
}
"""
    
    def generate(self, question: str, schema_context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate SQL from natural language question using the underlying model.

        The model is expected to return a string containing a JSON object
        matching the required output schema. We parse that JSON and validate
        it before returning the dict.
        """
        prompt = self._construct_prompt(question, schema_context)

        # Delegate to the provided model's generate method which returns text
        completion_text = self.model.generate(prompt)

        try:
            result = json.loads(completion_text)
        except json.JSONDecodeError:
            # Try to find a JSON object inside the text
            start_idx = completion_text.find("{")
            end_idx = completion_text.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON object found in model output")
                raise ValueError("Model did not return a valid JSON object")
            sub = completion_text[start_idx:end_idx]
            result = json.loads(sub)

        # Validate required fields
        required_fields = ["sql", "used_tables", "used_columns", "reason_short", "confidence"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        return result
    
    def _construct_prompt(self, question: str, schema_context: Dict[str, Any]) -> str:
        """Construct the master prompt with explicit slots.

        Expected schema_context keys:
        - dialect: e.g., "PostgreSQL" (defaults to PostgreSQL)
        - top_k_schema_items: List[dict]
        - schema_graph_brief: str
        - value_hints: Optional[str]
        """
        dialect = schema_context.get("dialect", "PostgreSQL")
        top_k_items = schema_context.get("top_k_schema_items", [])
        schema_graph_brief = schema_context.get("schema_graph_brief", "")
        value_hints = schema_context.get("value_hints", "")

        def _fmt_top_k(items: List[Dict[str, Any]]) -> str:
            lines: List[str] = []
            for item in items:
                cols = ", ".join([f"{c['name']}:{c.get('type','')}" for c in item.get("columns", [])])
                lines.append(f"- {item.get('table')}: {cols}")
            return "\n".join(lines)

        return f"""You are a Text-to-SQL generator for BI-style analytics over RELATIONAL databases.

## Rules
- DIALECT: {dialect} (strictly follow this dialect’s syntax).
- SAFETY: Generate READ-ONLY queries. Do NOT use INSERT/UPDATE/DELETE/TRUNCATE/DROP/ALTER.
- SCOPE: Only use tables/columns that exist in the provided schema. Prefer foreign-key join paths.
- OUTPUT FORMAT: Return a single JSON object, nothing else:

{{
  "sql": "<final SQL>",
  "used_tables": ["..."],
  "used_columns": ["..."],
  "reason_short": "<1-2 line rationale, no step-by-step>",
  "confidence": 0.0-1.0
}}

## Context
- Business question: {question}
- Top-k retrieved schema items (by relevance):
{_fmt_top_k(top_k_items)}
- Full schema graph & FKs (abbrev):
{schema_graph_brief}
- Example rows / enums (if any):
{value_hints}

## Generation Guidance
- Before you write SQL, internally plan joins, filters, groupings, and aggregations. Keep the plan internal; only return the JSON above.
- If metrics are implied, apply GROUP BY/ORDER BY/LIMIT appropriately.
- If dates are mentioned, do not guess unspecified ranges.
- Prefer window functions for latest-per-group and ranking tasks.
- If ambiguous, choose the most conservative interpretation consistent with column names and FKs.

## Validation
- Must parse for {dialect}.
- Must reference only existing tables/columns listed in Context.
- Prefer sargable predicates.

## Few-shot patterns
{self.few_shots}
"""

    def repair(self, original_json: Dict[str, Any], execution_feedback: str) -> Dict[str, Any]:
        """Basic repair: return the same schema with updated SQL candidate.

        This is a placeholder that would normally prompt a model with the
        error/result summary and ask for a corrected SQL. For now we just
        echo back the original structure and include the feedback as a note
        in reason_short when possible.
        """
        repaired = dict(original_json)
        if "reason_short" in repaired:
            repaired["reason_short"] = (repaired.get("reason_short") or "").strip() + " | repair: " + execution_feedback[:120]
        repaired["confidence"] = max(0.0, float(repaired.get("confidence", 0.0)) * 0.9)
        return repaired