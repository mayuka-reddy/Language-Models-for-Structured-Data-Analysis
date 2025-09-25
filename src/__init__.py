"""
Language Models for Structured Data Analysis

A comprehensive system for converting natural language queries into structured data operations
using advanced prompting strategies, schema-aware RAG, and fine-tuned language models.
"""

__version__ = "0.1.0"
__author__ = "Structured Data Analysis Team"
__email__ = "team@structured-data-analysis.com"

from .query_processor import NaturalLanguageQueryProcessor

__all__ = [
    "NaturalLanguageQueryProcessor",
]
