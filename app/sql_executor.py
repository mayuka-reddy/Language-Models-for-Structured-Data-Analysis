"""
SQL Executor Module
Handles database connections and safe SQL query execution.
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlparse
import logging

# Use standard logging if loguru is not available
try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Optional DuckDB import
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, using SQLite only")


class SQLExecutor:
    """
    Safe SQL query executor supporting SQLite and DuckDB.
    
    Provides secure query execution with result formatting,
    error handling, and basic query validation.
    """
    
    def __init__(self, db_path: str = "data/sample_retail.db", db_type: str = "sqlite"):
        """
        Initialize SQL executor.
        
        Args:
            db_path: Path to database file
            db_type: Database type ('sqlite' or 'duckdb')
        """
        self.db_path = db_path
        self.db_type = db_type.lower()
        self.connection = None
        
        # Create sample database if it doesn't exist
        if not Path(db_path).exists():
            self._create_sample_database()
        
        self._connect()
        logger.info(f"SQL Executor initialized with {db_type} database: {db_path}")
    
    def _connect(self):
        """Establish database connection."""
        try:
            if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
                self.connection = duckdb.connect(self.db_path)
            else:  # sqlite (fallback)
                if self.db_type == "duckdb" and not DUCKDB_AVAILABLE:
                    logger.warning("DuckDB requested but not available, falling back to SQLite")
                    self.db_type = "sqlite"
                    # Change extension to .db for SQLite
                    if self.db_path.endswith('.duckdb'):
                        self.db_path = self.db_path.replace('.duckdb', '.db')
                
                self.connection = sqlite3.connect(self.db_path)
                self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            logger.info(f"Database connection established ({self.db_type})")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _create_sample_database(self):
        """Create sample retail database with mock data."""
        logger.info("Creating sample retail database...")
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create temporary connection to set up database
        if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
            temp_conn = duckdb.connect(self.db_path)
        else:
            temp_conn = sqlite3.connect(self.db_path)
        
        try:
            cursor = temp_conn.cursor()
            
            # Create customers table
            cursor.execute("""
                CREATE TABLE customers (
                    customer_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    region TEXT,
                    signup_date DATE
                )
            """)
            
            # Create products table
            cursor.execute("""
                CREATE TABLE products (
                    product_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT,
                    price DECIMAL(10,2),
                    stock_quantity INTEGER
                )
            """)
            
            # Create orders table
            cursor.execute("""
                CREATE TABLE orders (
                    order_id INTEGER PRIMARY KEY,
                    customer_id INTEGER,
                    order_date DATE,
                    total_amount DECIMAL(10,2),
                    status TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            # Create order_items table
            cursor.execute("""
                CREATE TABLE order_items (
                    order_item_id INTEGER PRIMARY KEY,
                    order_id INTEGER,
                    product_id INTEGER,
                    quantity INTEGER,
                    unit_price DECIMAL(10,2),
                    FOREIGN KEY (order_id) REFERENCES orders(order_id),
                    FOREIGN KEY (product_id) REFERENCES products(product_id)
                )
            """)
            
            # Insert sample data
            self._insert_sample_data(cursor)
            
            temp_conn.commit()
            logger.info("Sample database created successfully")
            
        except Exception as e:
            logger.error(f"Error creating sample database: {e}")
            raise
        finally:
            temp_conn.close()
    
    def _insert_sample_data(self, cursor):
        """Insert sample data into tables."""
        # Sample customers
        customers_data = [
            (1, 'John Doe', 'john@email.com', 'North', '2023-01-15'),
            (2, 'Jane Smith', 'jane@email.com', 'South', '2023-02-20'),
            (3, 'Bob Johnson', 'bob@email.com', 'East', '2023-03-10'),
            (4, 'Alice Brown', 'alice@email.com', 'West', '2023-04-05'),
            (5, 'Charlie Wilson', 'charlie@email.com', 'North', '2023-05-12')
        ]
        
        cursor.executemany(
            "INSERT INTO customers (customer_id, name, email, region, signup_date) VALUES (?, ?, ?, ?, ?)",
            customers_data
        )
        
        # Sample products
        products_data = [
            (1, 'Laptop', 'Electronics', 999.99, 50),
            (2, 'Mouse', 'Electronics', 29.99, 200),
            (3, 'Keyboard', 'Electronics', 79.99, 150),
            (4, 'Monitor', 'Electronics', 299.99, 75),
            (5, 'Headphones', 'Electronics', 149.99, 100)
        ]
        
        cursor.executemany(
            "INSERT INTO products (product_id, name, category, price, stock_quantity) VALUES (?, ?, ?, ?, ?)",
            products_data
        )
        
        # Sample orders
        orders_data = [
            (1, 1, '2023-06-01', 1079.98, 'completed'),
            (2, 2, '2023-06-02', 329.98, 'completed'),
            (3, 3, '2023-06-03', 149.99, 'pending'),
            (4, 4, '2023-06-04', 79.99, 'completed'),
            (5, 5, '2023-06-05', 449.98, 'shipped')
        ]
        
        cursor.executemany(
            "INSERT INTO orders (order_id, customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?, ?)",
            orders_data
        )
        
        # Sample order items
        order_items_data = [
            (1, 1, 1, 1, 999.99),  # John bought 1 laptop
            (2, 1, 2, 2, 29.99),   # John bought 2 mice
            (3, 2, 4, 1, 299.99),  # Jane bought 1 monitor
            (4, 2, 2, 1, 29.99),   # Jane bought 1 mouse
            (5, 3, 5, 1, 149.99),  # Bob bought 1 headphones
            (6, 4, 3, 1, 79.99),   # Alice bought 1 keyboard
            (7, 5, 4, 1, 299.99),  # Charlie bought 1 monitor
            (8, 5, 5, 1, 149.99)   # Charlie bought 1 headphones
        ]
        
        cursor.executemany(
            "INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?, ?)",
            order_items_data
        )
    
    def execute_query(self, sql: str) -> Dict[str, Any]:
        """
        Execute SQL query safely and return results.
        
        Args:
            sql: SQL query string
            
        Returns:
            Dictionary containing results, metadata, and any errors
        """
        if not sql.strip():
            return {
                'success': False,
                'error': 'Empty SQL query',
                'data': None,
                'row_count': 0
            }
        
        # Validate query safety
        if not self._is_safe_query(sql):
            return {
                'success': False,
                'error': 'Unsafe query detected. Only SELECT statements are allowed.',
                'data': None,
                'row_count': 0
            }
        
        try:
            # Execute query
            if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
                result = self.connection.execute(sql).fetchdf()
                data = result.to_dict('records')
                row_count = len(result)
            else:  # sqlite
                cursor = self.connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                data = [dict(row) for row in rows]
                row_count = len(data)
            
            return {
                'success': True,
                'data': data,
                'row_count': row_count,
                'sql': sql
            }
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'row_count': 0,
                'sql': sql
            }
    
    def _is_safe_query(self, sql: str) -> bool:
        """
        Check if SQL query is safe (read-only).
        
        Args:
            sql: SQL query string
            
        Returns:
            True if query is safe, False otherwise
        """
        # Parse SQL to check for dangerous operations
        try:
            parsed = sqlparse.parse(sql.upper().strip())
            
            if not parsed:
                return False
            
            # Get first statement
            statement = parsed[0]
            
            # Check for dangerous keywords
            dangerous_keywords = [
                'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
                'TRUNCATE', 'REPLACE', 'MERGE', 'EXEC', 'EXECUTE'
            ]
            
            sql_upper = sql.upper()
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    return False
            
            # Must start with SELECT
            first_token = None
            for token in statement.flatten():
                if token.ttype is None and token.value.strip():
                    first_token = token.value.upper().strip()
                    break
            
            return first_token == 'SELECT'
            
        except Exception:
            # If parsing fails, be conservative
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Returns:
            Dictionary containing table and column information
        """
        try:
            if self.db_type == "duckdb" and DUCKDB_AVAILABLE:
                # Get table names
                tables_result = self.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
                
                schema_info = {'tables': {}}
                
                for (table_name,) in tables_result:
                    # Get column info
                    columns_result = self.connection.execute(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND table_schema = 'main'
                    """).fetchall()
                    
                    schema_info['tables'][table_name] = {
                        'columns': [
                            {
                                'name': col_name,
                                'type': data_type,
                                'nullable': is_nullable == 'YES'
                            }
                            for col_name, data_type, is_nullable in columns_result
                        ]
                    }
            
            else:  # sqlite
                cursor = self.connection.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                schema_info = {'tables': {}}
                
                for (table_name,) in tables:
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    schema_info['tables'][table_name] = {
                        'columns': [
                            {
                                'name': col[1],  # column name
                                'type': col[2],  # data type
                                'nullable': not col[3],  # not null flag
                                'primary_key': bool(col[5])  # pk flag
                            }
                            for col in columns
                        ]
                    }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return {'tables': {}}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Example usage and testing
if __name__ == "__main__":
    # Test the SQL executor
    executor = SQLExecutor()
    
    # Test queries
    test_queries = [
        "SELECT * FROM customers LIMIT 3",
        "SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region",
        "SELECT p.name, p.price FROM products p WHERE p.price > 100",
        "SELECT c.name, o.total_amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id"
    ]
    
    print("Testing SQL Executor:")
    print("=" * 50)
    
    # Get schema info
    schema = executor.get_schema_info()
    print("Database Schema:")
    for table_name, table_info in schema['tables'].items():
        columns = [col['name'] for col in table_info['columns']]
        print(f"  {table_name}: {', '.join(columns)}")
    print()
    
    # Test queries
    for sql in test_queries:
        print(f"Query: {sql}")
        result = executor.execute_query(sql)
        
        if result['success']:
            print(f"Rows returned: {result['row_count']}")
            if result['data']:
                # Show first few rows
                for i, row in enumerate(result['data'][:2]):
                    print(f"  Row {i+1}: {row}")
        else:
            print(f"Error: {result['error']}")
        
        print("-" * 30)
    
    executor.close()