"""
NL-to-SQL Inference Module
Handles loading T5 model and generating SQL from natural language questions.
"""

import os
import yaml
from typing import Dict, List, Optional, Any
import logging

# Handle optional dependencies gracefully
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers not installed. Run: pip install transformers torch")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  torch not installed. Run: pip install torch")

try:
    from loguru import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class NL2SQLInference:
    """
    T5-based inference engine for converting natural language to SQL.
    
    This class handles model loading, tokenization, and SQL generation
    with configurable parameters and fallback mechanisms.
    """
    
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the inference engine.
        
        Args:
            config_path: Path to model configuration file
        """
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            raise ImportError(
                "Missing required dependencies. Install with:\n"
                "pip install transformers torch"
            )
        
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        logger.info(f"NL2SQL Inference initialized on {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('model', {})
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'name': 't5-small',
                'max_input_length': 512,
                'max_output_length': 256,
                'num_beams': 4,
                'temperature': 0.7
            }
    
    def _load_model(self):
        """Load T5 model and tokenizer."""
        model_name = self.config.get('name', 't5-small')
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def generate_sql(self, question: str, schema_info: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            schema_info: Optional database schema information
            
        Returns:
            Dictionary containing generated SQL and metadata
        """
        if not question.strip():
            return {
                'sql': '',
                'confidence': 0.0,
                'error': 'Empty question provided'
            }
        
        try:
            # Prepare input text
            input_text = self._prepare_input(question, schema_info)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.get('max_input_length', 512),
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.get('max_output_length', 256),
                    num_beams=self.config.get('num_beams', 4),
                    temperature=self.config.get('temperature', 0.7),
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode output
            generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(outputs[0])
            
            return {
                'sql': generated_sql,
                'confidence': confidence,
                'input_text': input_text,
                'model': self.config.get('name', 't5-small')
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            return {
                'sql': '',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _prepare_input(self, question: str, schema_info: Optional[str] = None) -> str:
        """
        Prepare input text for the model.
        
        Args:
            question: Natural language question
            schema_info: Optional schema information
            
        Returns:
            Formatted input text
        """
        # Basic prompt template
        if schema_info:
            input_text = f"translate English to SQL: {question} | schema: {schema_info}"
        else:
            input_text = f"translate English to SQL: {question}"
        
        return input_text
    
    def _calculate_confidence(self, output_ids: torch.Tensor) -> float:
        """
        Calculate confidence score for generated output.
        
        Args:
            output_ids: Generated token IDs
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simplified confidence calculation
        # In practice, you might use model logits or other metrics
        sql_length = len(output_ids)
        
        # Heuristic: longer, more complex queries might be less confident
        if sql_length < 10:
            return 0.5
        elif sql_length < 20:
            return 0.8
        else:
            return 0.9
    
    def batch_generate(self, questions: List[str], schema_info: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate SQL for multiple questions.
        
        Args:
            questions: List of natural language questions
            schema_info: Optional schema information
            
        Returns:
            List of generation results
        """
        results = []
        for question in questions:
            result = self.generate_sql(question, schema_info)
            results.append(result)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test the inference engine
    inference = NL2SQLInference()
    
    # Test questions
    test_questions = [
        "Show me all customers",
        "What are the total sales by region?",
        "Find the top 5 products by revenue",
        "How many orders were placed last month?"
    ]
    
    # Sample schema info
    schema = "Tables: customers(id, name, region), orders(id, customer_id, amount, date), products(id, name, price)"
    
    print("Testing NL2SQL Inference:")
    print("=" * 50)
    
    for question in test_questions:
        result = inference.generate_sql(question, schema)
        print(f"Question: {question}")
        print(f"SQL: {result['sql']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 30)