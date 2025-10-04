"""
Tests for the SQL executor module.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sql_executor import SQLExecutor


class TestSQLExecutor:
    """Test cases for SQL executor."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def sql_executor(self, temp_db_path):
        """Create SQL executor with temporary database."""
        return SQLExecutor(temp_db_path, "sqlite")
    
    def test_initialization(self, sql_executor):
        """Test SQL executor initialization."""
        assert sql_executor.connection is not None
        assert sql_executor.db_type == "sqlite"
        assert Path(sql_executor.db_path).exists()
    
    def test_sample_database_creation(self, sql_executor):
        """Test that sample database is created with proper tables."""
        schema = sql_executor.get_schema_info()
        
        expected_tables = ['customers', 'products', 'orders', 'order_items']
        for table in expected_tables:
            assert table in schema['tables']
    
    def test_safe_query_validation(self, sql_executor):
        """Test query safety validation."""
        # Safe queries
        assert sql_executor._is_safe_query("SELECT * FROM customers")
        assert sql_executor._is_safe_query("SELECT name FROM customers WHERE id = 1")
        
        # Unsafe queries
        assert not sql_executor._is_safe_query("DROP TABLE customers")
        assert not sql_executor._is_safe_query("INSERT INTO customers VALUES (1, 'test')")
        assert not sql_executor._is_safe_query("UPDATE customers SET name = 'test'")
        assert not sql_executor._is_safe_query("DELETE FROM customers")
    
    def test_execute_valid_query(self, sql_executor):
        """Test execution of valid SELECT query."""
        result = sql_executor.execute_query("SELECT * FROM customers LIMIT 3")
        
        assert result['success'] is True
        assert isinstance(result['data'], list)
        assert result['row_count'] <= 3
        assert 'sql' in result
    
    def test_execute_invalid_query(self, sql_executor):
        """Test execution of invalid query."""
        result = sql_executor.execute_query("SELECT * FROM non_existent_table")
        
        assert result['success'] is False
        assert 'error' in result
        assert result['data'] is None
    
    def test_execute_unsafe_query(self, sql_executor):
        """Test execution of unsafe query."""
        result = sql_executor.execute_query("DROP TABLE customers")
        
        assert result['success'] is False
        assert 'unsafe' in result['error'].lower()
    
    def test_execute_empty_query(self, sql_executor):
        """Test execution of empty query."""
        result = sql_executor.execute_query("")
        
        assert result['success'] is False
        assert 'empty' in result['error'].lower()
    
    def test_schema_info_retrieval(self, sql_executor):
        """Test schema information retrieval."""
        schema = sql_executor.get_schema_info()
        
        assert isinstance(schema, dict)
        assert 'tables' in schema
        assert len(schema['tables']) > 0
        
        # Check customers table structure
        customers_table = schema['tables']['customers']
        assert 'columns' in customers_table
        
        # Check for expected columns
        column_names = [col['name'] for col in customers_table['columns']]
        expected_columns = ['customer_id', 'name', 'email', 'region']
        for col in expected_columns:
            assert col in column_names
    
    def test_aggregation_query(self, sql_executor):
        """Test aggregation query execution."""
        result = sql_executor.execute_query(
            "SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region"
        )
        
        assert result['success'] is True
        assert result['row_count'] > 0
        
        # Check that results have expected structure
        if result['data']:
            first_row = result['data'][0]
            assert 'region' in first_row
            assert 'customer_count' in first_row
    
    def test_join_query(self, sql_executor):
        """Test JOIN query execution."""
        result = sql_executor.execute_query("""
            SELECT c.name, o.total_amount 
            FROM customers c 
            JOIN orders o ON c.customer_id = o.customer_id 
            LIMIT 5
        """)
        
        assert result['success'] is True
        
        if result['data']:
            first_row = result['data'][0]
            assert 'name' in first_row
            assert 'total_amount' in first_row
    
    def test_connection_handling(self, temp_db_path):
        """Test database connection handling."""
        executor = SQLExecutor(temp_db_path)
        
        # Test that connection works
        result = executor.execute_query("SELECT 1 as test")
        assert result['success'] is True
        
        # Test connection close
        executor.close()
        
        # After closing, new queries should fail gracefully
        # (This depends on implementation - might reconnect automatically)
    
    def test_data_types_in_results(self, sql_executor):
        """Test that different data types are handled correctly."""
        result = sql_executor.execute_query("""
            SELECT 
                customer_id,
                name,
                signup_date
            FROM customers 
            LIMIT 1
        """)
        
        assert result['success'] is True
        
        if result['data']:
            row = result['data'][0]
            assert isinstance(row['customer_id'], int)
            assert isinstance(row['name'], str)
            # signup_date might be string or date depending on SQLite handling


if __name__ == "__main__":
    pytest.main([__file__])