"""Module for handling database connections and query execution."""
from typing import List, Dict, Any
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import os
from loguru import logger

class DatabaseConnection:
    """Handles database connections and safe query execution."""
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = self._create_engine()
        
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine from environment variables."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        return sa.create_engine(db_url)
    
    def execute_read_only(self, query: str) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query safely."""
        # Ensure the query is read-only
        query = query.strip()
        if not query.lower().startswith(("select", "with")):
            raise ValueError("Only SELECT queries are allowed")
        
        try:
            # Start a read-only transaction
            with self.engine.connect() as conn:
                # Set transaction to read-only
                conn.execute(sa.text("SET TRANSACTION READ ONLY"))
                conn.execute(sa.text("BEGIN"))
                
                try:
                    # Execute the query
                    result = conn.execute(sa.text(query))
                    
                    # Convert to list of dicts
                    columns = result.keys()
                    rows = [dict(zip(columns, row)) for row in result.fetchall()]
                    
                    conn.execute(sa.text("COMMIT"))
                    return rows
                
                except Exception as e:
                    conn.execute(sa.text("ROLLBACK"))
                    logger.error(f"Query execution failed: {str(e)}")
                    raise
        
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            raise