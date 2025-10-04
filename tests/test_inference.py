"""
Tests for the inference module.
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import NL2SQLInference


class TestNL2SQLInference:
    """Test cases for NL2SQL inference engine."""
    
    @pytest.fixture
    def inference_engine(self):
        """Create inference engine for testing."""
        return NL2SQLInference()
    
    def test_initialization(self, inference_engine):
        """Test that inference engine initializes correctly."""
        assert inference_engine.model is not None
        assert inference_engine.tokenizer is not None
        assert inference_engine.config is not None
    
    def test_generate_sql_basic(self, inference_engine):
        """Test basic SQL generation."""
        question = "Show me all customers"
        result = inference_engine.generate_sql(question)
        
        assert isinstance(result, dict)
        assert 'sql' in result
        assert 'confidence' in result
        assert isinstance(result['sql'], str)
        assert isinstance(result['confidence'], float)
    
    def test_generate_sql_with_schema(self, inference_engine):
        """Test SQL generation with schema information."""
        question = "What are the total sales?"
        schema = "customers(id, name), orders(id, customer_id, amount)"
        
        result = inference_engine.generate_sql(question, schema)
        
        assert isinstance(result, dict)
        assert 'sql' in result
        assert 'input_text' in result
        assert schema in result['input_text']
    
    def test_empty_question(self, inference_engine):
        """Test handling of empty questions."""
        result = inference_engine.generate_sql("")
        
        assert 'error' in result
        assert result['sql'] == ''
        assert result['confidence'] == 0.0
    
    def test_batch_generation(self, inference_engine):
        """Test batch SQL generation."""
        questions = [
            "Show me all customers",
            "What are the total sales?",
            "Find top products"
        ]
        
        results = inference_engine.batch_generate(questions)
        
        assert len(results) == len(questions)
        for result in results:
            assert isinstance(result, dict)
            assert 'sql' in result
    
    def test_confidence_calculation(self, inference_engine):
        """Test confidence score calculation."""
        question = "Show me all customers"
        result = inference_engine.generate_sql(question)
        
        confidence = result['confidence']
        assert 0.0 <= confidence <= 1.0
    
    def test_input_preparation(self, inference_engine):
        """Test input text preparation."""
        question = "Show customers"
        schema = "customers(id, name)"
        
        input_text = inference_engine._prepare_input(question, schema)
        
        assert "translate English to SQL" in input_text
        assert question in input_text
        assert schema in input_text
    
    def test_config_loading(self):
        """Test configuration loading."""
        # Test with non-existent config file
        inference = NL2SQLInference("non_existent_config.yaml")
        
        # Should use default config
        assert inference.config['name'] == 't5-small'
        assert 'max_input_length' in inference.config


if __name__ == "__main__":
    pytest.main([__file__])