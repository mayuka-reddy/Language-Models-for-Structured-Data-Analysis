"""
Tests for the metrics module.
"""

import pytest
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.metrics import ModelEvaluator


class TestModelEvaluator:
    """Test cases for model evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create model evaluator for testing."""
        return ModelEvaluator()
    
    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data."""
        return [
            {
                'question': 'Show me all customers',
                'expected_sql': 'SELECT * FROM customers',
                'schema': 'customers(customer_id, name, email, region)'
            },
            {
                'question': 'What are the total sales?',
                'expected_sql': 'SELECT SUM(total_amount) FROM orders',
                'schema': 'orders(order_id, customer_id, total_amount)'
            }
        ]
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.smoothing_function is not None
    
    def test_exact_match_check(self, evaluator):
        """Test exact match checking."""
        sql1 = "SELECT * FROM customers"
        sql2 = "SELECT * FROM customers"
        sql3 = "SELECT name FROM customers"
        
        assert evaluator._check_exact_match(sql1, sql2) is True
        assert evaluator._check_exact_match(sql1, sql3) is False
    
    def test_sql_normalization(self, evaluator):
        """Test SQL normalization."""
        sql1 = "  SELECT   *   FROM   customers  ; "
        sql2 = "SELECT * FROM customers"
        
        normalized1 = evaluator._normalize_sql(sql1)
        normalized2 = evaluator._normalize_sql(sql2)
        
        assert normalized1 == normalized2
        assert normalized1 == "SELECT * FROM CUSTOMERS"
    
    def test_sql_tokenization(self, evaluator):
        """Test SQL tokenization for BLEU score."""
        sql = "SELECT name, age FROM customers WHERE age > 25"
        tokens = evaluator._tokenize_sql(sql)
        
        expected_tokens = ['select', 'name', ',', 'age', 'from', 'customers', 'where', 'age', '>', '25']
        assert tokens == expected_tokens
    
    def test_bleu_score_calculation(self, evaluator):
        """Test BLEU score calculation."""
        predicted = "SELECT * FROM customers"
        expected = "SELECT * FROM customers"
        
        score = evaluator._calculate_bleu_score(predicted, expected)
        assert score == 1.0  # Perfect match
        
        # Test partial match
        predicted2 = "SELECT name FROM customers"
        score2 = evaluator._calculate_bleu_score(predicted2, expected)
        assert 0.0 < score2 < 1.0
    
    def test_schema_compliance_check(self, evaluator):
        """Test schema compliance checking."""
        sql = "SELECT * FROM customers"
        schema = "customers(customer_id, name, email)"
        
        # This should pass
        assert evaluator._check_schema_compliance(sql, schema) is True
        
        # Test with non-existent table
        sql_bad = "SELECT * FROM non_existent_table"
        assert evaluator._check_schema_compliance(sql_bad, schema) is False
    
    def test_execution_correctness_check(self, evaluator):
        """Test execution correctness checking."""
        # Test identical queries
        sql1 = "SELECT * FROM customers"
        sql2 = "SELECT * FROM customers"
        
        assert evaluator._check_execution_correctness(sql1, sql2) is True
        
        # Test different queries
        sql3 = "SELECT name FROM customers"
        result = evaluator._check_execution_correctness(sql1, sql3)
        assert isinstance(result, bool)
    
    def test_dummy_models_creation(self, evaluator):
        """Test creation of dummy models."""
        models = evaluator.create_dummy_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Test that models have generate_sql method
        for model_name, model in models.items():
            assert hasattr(model, 'generate_sql')
            
            # Test model generation
            result = model.generate_sql("Show me customers")
            assert isinstance(result, dict)
            assert 'sql' in result
    
    def test_test_dataset_creation(self, evaluator):
        """Test creation of test dataset."""
        test_data = evaluator.create_test_dataset()
        
        assert isinstance(test_data, list)
        assert len(test_data) > 0
        
        # Check structure of test cases
        for test_case in test_data:
            assert 'question' in test_case
            assert 'expected_sql' in test_case
            assert 'schema' in test_case
    
    def test_single_case_evaluation(self, evaluator):
        """Test evaluation of single test case."""
        # Create a simple dummy model
        class SimpleModel:
            def generate_sql(self, question, schema=""):
                return {'sql': 'SELECT * FROM customers'}
        
        model = SimpleModel()
        test_case = {
            'question': 'Show me all customers',
            'expected_sql': 'SELECT * FROM customers',
            'schema': 'customers(id, name)'
        }
        
        result = evaluator._evaluate_single_case(test_case, model)
        
        assert isinstance(result, dict)
        assert 'execution_correct' in result
        assert 'exact_match' in result
        assert 'schema_compliant' in result
        assert 'bleu_score' in result
        assert 'response_time' in result
        assert 'error' in result
    
    def test_model_evaluation(self, evaluator, sample_test_data):
        """Test full model evaluation."""
        # Create dummy models
        models = evaluator.create_dummy_models()
        
        # Take only first model for faster testing
        test_models = {list(models.keys())[0]: list(models.values())[0]}
        
        results_df = evaluator.evaluate_models(sample_test_data, test_models)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(test_models)
        
        # Check required columns
        required_columns = [
            'model', 'total_queries', 'execution_accuracy',
            'exact_match_accuracy', 'schema_compliance_rate',
            'avg_bleu_score', 'avg_response_time', 'error_rate'
        ]
        
        for col in required_columns:
            assert col in results_df.columns
    
    def test_comparison_report_generation(self, evaluator):
        """Test comparison report generation."""
        # Create sample results DataFrame
        sample_results = pd.DataFrame([
            {
                'model': 'test_model',
                'execution_accuracy': 0.8,
                'exact_match_accuracy': 0.6,
                'schema_compliance_rate': 0.9,
                'avg_bleu_score': 0.75,
                'avg_response_time': 1.2,
                'error_rate': 0.1
            }
        ])
        
        report = evaluator.generate_comparison_report(sample_results)
        
        assert isinstance(report, str)
        assert 'Model Comparison Report' in report
        assert 'test_model' in report
        assert 'Execution Accuracy' in report
    
    def test_error_handling(self, evaluator):
        """Test error handling in evaluation."""
        # Create a model that raises exceptions
        class ErrorModel:
            def generate_sql(self, question, schema=""):
                raise Exception("Test error")
        
        model = ErrorModel()
        test_case = {
            'question': 'Test question',
            'expected_sql': 'SELECT 1',
            'schema': 'test(id)'
        }
        
        result = evaluator._evaluate_single_case(test_case, model)
        
        assert result['error'] is True
        assert result['execution_correct'] is False
        assert result['bleu_score'] == 0.0
    
    def test_empty_inputs(self, evaluator):
        """Test handling of empty inputs."""
        # Test empty SQL
        assert evaluator._calculate_bleu_score("", "SELECT * FROM test") == 0.0
        assert evaluator._check_exact_match("", "SELECT * FROM test") is False
        
        # Test empty schema
        assert evaluator._check_schema_compliance("SELECT * FROM test", "") is False


if __name__ == "__main__":
    pytest.main([__file__])