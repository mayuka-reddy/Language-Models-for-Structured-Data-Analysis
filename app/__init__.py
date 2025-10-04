"""
NL-to-SQL Assistant
Backend application package for natural language to SQL conversion.
"""

__version__ = "0.1.0"
__author__ = "NL2SQL Team"

from .inference import NL2SQLInference
from .sql_executor import SQLExecutor
from .insights import InsightsGenerator
from .charts import ChartGenerator
from .metrics import ModelEvaluator

__all__ = [
    "NL2SQLInference",
    "SQLExecutor", 
    "InsightsGenerator",
    "ChartGenerator",
    "ModelEvaluator"
]