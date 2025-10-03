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
                """Return a compact few-shot string tailored to the sample retail schema."""
                return """
# Example 1: Simple aggregation with grouping
Q: Which city has the highest number of customers?
A: {
    "sql": "SELECT city, COUNT(*) as customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1",
    "used_tables": ["customer"],
    "used_columns": ["city"],
    "reason_short": "Count customers per city, order by count descending to find highest",
    "confidence": 0.95
}

# Example 2: Date filtering with joins and aggregation
Q: What was the total sales in August 2025?
A: {
    "sql": "SELECT SUM(total_amount) as total_sales FROM orders WHERE DATE_TRUNC('month', order_date) = '2025-08-01'",
    "used_tables": ["orders"],
    "used_columns": ["order_date", "total_amount"],
    "reason_short": "Sum order amounts for August 2025 using DATE_TRUNC for month comparison",
    "confidence": 0.92
}

# Example 3: Complex join with aggregation and ranking
Q: List the top 5 products by revenue this quarter.
A: {
    "sql": "WITH product_revenue AS (SELECT p.product_id, p.name, SUM(oi.quantity * p.price) as revenue FROM product p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id WHERE o.order_date >= DATE_TRUNC('quarter', CURRENT_DATE) GROUP BY p.product_id, p.name) SELECT * FROM product_revenue ORDER BY revenue DESC LIMIT 5",
    "used_tables": ["product", "order_items", "orders"],
    "used_columns": ["product_id", "name", "quantity", "price", "order_date"],
    "reason_short": "Join orders and products, calculate revenue (quantity * price), filter for current quarter",
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
        """Construct the prompt for SQL generation."""
        return f"""You are a Text-to-SQL generator for BI-style analytics over RELATIONAL databases.

## Rules
- DIALECT: PostgreSQL (strictly follow PostgreSQL syntax)
- SAFETY: Generate **READ-ONLY** queries. Do NOT use INSERT/UPDATE/DELETE/TRUNCATE/DROP/ALTER
- SCOPE: Only use tables/columns that exist in the provided schema
- OUTPUT FORMAT: Return a single JSON object with the following fields:
  - sql: The generated SQL query
  - used_tables: List of tables referenced
  - used_columns: List of columns referenced
  - reason_short: Brief explanation (1-2 lines)
  - confidence: Float between 0.0-1.0

## Context
Business question: {question}

Schema Graph:
{schema_context.get('schema_graph_brief', '')}

## Few-shot Examples
{self.few_shots}

Generate a response for the given question:
"""