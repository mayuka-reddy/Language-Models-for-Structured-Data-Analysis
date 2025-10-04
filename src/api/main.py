"""Main FastAPI application for the NLP to SQL service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn

from src.models.nl2sql import NL2SQLGenerator
from src.database.connection import DatabaseConnection
from src.schema.manager import SchemaManager

app = FastAPI(
    title="NLP to SQL API",
    description="Convert natural language questions to SQL queries",
    version="0.1.0"
)

class QueryRequest(BaseModel):
    """Request model for natural language queries."""
    question: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    """Response model for SQL queries."""
    sql: str
    used_tables: List[str]
    used_columns: List[str]
    reason_short: str
    confidence: float

class RepairRequest(BaseModel):
    """Request model for repairing a prior generation using execution feedback."""
    original: Dict[str, Any]
    execution_feedback: str

class DecomposeRequest(BaseModel):
    """Request model for decomposing a question."""
    question: str
    context: Optional[Dict[str, Any]] = None

def _resolve_schema_context(request_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve schema context from request or SchemaManager.

    Allows offline usage by accepting a precomputed schema_context via request.context["schema_context"].
    """
    if request_context and isinstance(request_context.get("schema_context"), dict):
        return request_context["schema_context"]
    # Fallback to live reflection if available
    schema_manager = SchemaManager()
    return schema_manager.get_context()

@app.post("/parse", response_model=QueryResponse)
async def parse_question(request: QueryRequest):
    """Convert natural language question to SQL."""
    try:
        # Initialize components (in production, these would be dependency-injected)
        generator = NL2SQLGenerator()
        schema_context = _resolve_schema_context(request.context)
        # Inject optional dialect into schema_context if provided
        if request.context and request.context.get("dialect"):
            schema_context = dict(schema_context)
            schema_context["dialect"] = request.context["dialect"]

        # Generate SQL from question
        result = generator.generate(
            question=request.question,
            schema_context=schema_context,
            **request.context or {}
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/execute")
async def execute_query(request: QueryRequest):
    """Execute the generated SQL query."""
    try:
        # First generate the SQL
        generator = NL2SQLGenerator()
        schema_context = _resolve_schema_context(request.context)
        if request.context and request.context.get("dialect"):
            schema_context = dict(schema_context)
            schema_context["dialect"] = request.context["dialect"]

        query_result = generator.generate(
            question=request.question,
            schema_context=schema_context,
            **request.context or {}
        )
        
        # Then execute it safely
        db = DatabaseConnection()
        results = db.execute_read_only(query_result["sql"] if isinstance(query_result, dict) else query_result.sql)
        
        return {
            "sql": (query_result["sql"] if isinstance(query_result, dict) else query_result.sql),
            "results": results,
            "metadata": {
                "used_tables": (query_result.get("used_tables") if isinstance(query_result, dict) else query_result.used_tables),
                "reason": (query_result.get("reason_short") if isinstance(query_result, dict) else query_result.reason_short)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/repair", response_model=QueryResponse)
async def repair_query(request: RepairRequest):
    """Repair a previously generated SQL using execution feedback."""
    try:
        generator = NL2SQLGenerator()
        repaired = generator.repair(request.original, request.execution_feedback)
        return repaired
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/decompose")
async def decompose_question(request: DecomposeRequest):
    """Optional: Decompose a question into minimal sub-steps (heuristic placeholder)."""
    try:
        # Minimal heuristic decomposer; in production, call a model
        steps = [request.question]
        required_tables_or_views: List[str] = []
        key_columns: List[str] = []
        notes = "heuristic decomposer"
        return {
            "steps": steps,
            "required_tables_or_views": required_tables_or_views,
            "key_columns": key_columns,
            "notes": notes
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "nl2sql-api",
        "version": "0.1.0"
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "NL-to-SQL Assistant API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)