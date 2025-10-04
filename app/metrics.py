"""
Model Evaluation Metrics Module
Compares different NL-to-SQL models and generates evaluation metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import sqlparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score
import re
from loguru import logger
import time


class ModelEvaluator:
    """
    Evaluates and compares NL-to-SQL models using various metrics.
    
    Supports evaluation of execution correctness, exact match,
    schema compliance, and BLEU scores for model comparison.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.smoothing_function = SmoothingFunction().method1
        logger.info("Model Evaluator initialized")
    
    def evaluate_models(self, test_data: List[Dict[str, Any]], models: Dict[str, Any]) -> pd.DataFrame:
        """
        Evaluate multiple models on test data.
        
        Args:
            test_data: List of test cases with questions and expected SQL
            models: Dictionary of model names to model instances
            
        Returns:
            DataFrame with evaluation results for each model
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            model_results = {
                'model': model_name,
                'total_queries': len(test_data),
                'execution_correct': 0,
                'exact_match': 0,
                'schema_compliant': 0,
                'avg_bleu_score': 0.0,
                'avg_response_time': 0.0,
                'errors': 0
            }
            
            bleu_scores = []
            response_times = []
            
            for test_case in test_data:
                # Evaluate single test case
                eval_result = self._evaluate_single_case(test_case, model)
                
                # Update counters
                if eval_result['execution_correct']:
                    model_results['execution_correct'] += 1
                
                if eval_result['exact_match']:
                    model_results['exact_match'] += 1
                
                if eval_result['schema_compliant']:
                    model_results['schema_compliant'] += 1
                
                if eval_result['error']:
                    model_results['errors'] += 1
                
                bleu_scores.append(eval_result['bleu_score'])
                response_times.append(eval_result['response_time'])
            
            # Calculate averages and percentages
            total = len(test_data)
            model_results['execution_accuracy'] = model_results['execution_correct'] / total
            model_results['exact_match_accuracy'] = model_results['exact_match'] / total
            model_results['schema_compliance_rate'] = model_results['schema_compliant'] / total
            model_results['avg_bleu_score'] = np.mean(bleu_scores)
            model_results['avg_response_time'] = np.mean(response_times)
            model_results['error_rate'] = model_results['errors'] / total
            
            results.append(model_results)
        
        return pd.DataFrame(results)
    
    def _evaluate_single_case(self, test_case: Dict[str, Any], model: Any) -> Dict[str, Any]:
        """
        Evaluate a single test case.
        
        Args:
            test_case: Dictionary with 'question', 'expected_sql', and optional 'schema'
            model: Model instance with generate_sql method
            
        Returns:
            Dictionary with evaluation results
        """
        question = test_case['question']
        expected_sql = test_case['expected_sql']
        schema_info = test_case.get('schema', '')
        
        try:
            # Generate SQL with timing
            start_time = time.time()
            
            if hasattr(model, 'generate_sql'):
                result = model.generate_sql(question, schema_info)
                predicted_sql = result.get('sql', '') if isinstance(result, dict) else str(result)
            else:
                # Fallback for simple models
                predicted_sql = str(model(question))
            
            response_time = time.time() - start_time
            
            # Evaluate metrics
            execution_correct = self._check_execution_correctness(predicted_sql, expected_sql)
            exact_match = self._check_exact_match(predicted_sql, expected_sql)
            schema_compliant = self._check_schema_compliance(predicted_sql, schema_info)
            bleu_score = self._calculate_bleu_score(predicted_sql, expected_sql)
            
            return {
                'execution_correct': execution_correct,
                'exact_match': exact_match,
                'schema_compliant': schema_compliant,
                'bleu_score': bleu_score,
                'response_time': response_time,
                'error': False,
                'predicted_sql': predicted_sql
            }
            
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
            return {
                'execution_correct': False,
                'exact_match': False,
                'schema_compliant': False,
                'bleu_score': 0.0,
                'response_time': 0.0,
                'error': True,
                'predicted_sql': ''
            }
    
    def _check_execution_correctness(self, predicted_sql: str, expected_sql: str) -> bool:
        """
        Check if predicted SQL would produce same results as expected SQL.
        
        Note: This is a simplified version. In practice, you'd execute both
        queries against a test database and compare results.
        """
        # Simplified check - normalize and compare structure
        pred_normalized = self._normalize_sql(predicted_sql)
        expected_normalized = self._normalize_sql(expected_sql)
        
        # Basic structural similarity check
        pred_tokens = set(pred_normalized.split())
        expected_tokens = set(expected_normalized.split())
        
        # Calculate Jaccard similarity
        intersection = len(pred_tokens.intersection(expected_tokens))
        union = len(pred_tokens.union(expected_tokens))
        
        similarity = intersection / union if union > 0 else 0
        
        # Consider execution correct if similarity > 0.7
        return similarity > 0.7
    
    def _check_exact_match(self, predicted_sql: str, expected_sql: str) -> bool:
        """Check if predicted SQL exactly matches expected SQL (normalized)."""
        pred_normalized = self._normalize_sql(predicted_sql)
        expected_normalized = self._normalize_sql(expected_sql)
        
        return pred_normalized == expected_normalized
    
    def _check_schema_compliance(self, predicted_sql: str, schema_info: str) -> bool:
        """
        Check if predicted SQL complies with schema constraints.
        
        Args:
            predicted_sql: Generated SQL query
            schema_info: Schema information string
            
        Returns:
            True if schema compliant, False otherwise
        """
        if not predicted_sql.strip() or not schema_info.strip():
            return False
        
        try:
            # Parse SQL to extract table and column references
            parsed = sqlparse.parse(predicted_sql.upper())
            
            if not parsed:
                return False
            
            # Extract table names (simplified)
            sql_upper = predicted_sql.upper()
            
            # Find table names after FROM and JOIN keywords
            from_pattern = r'FROM\s+(\w+)'
            join_pattern = r'JOIN\s+(\w+)'
            
            tables_in_sql = set()
            tables_in_sql.update(re.findall(from_pattern, sql_upper))
            tables_in_sql.update(re.findall(join_pattern, sql_upper))
            
            # Extract table names from schema info
            schema_upper = schema_info.upper()
            # Simple pattern to find table names in schema description
            schema_tables = set(re.findall(r'(\w+)\s*\(', schema_upper))
            
            # Check if all tables in SQL exist in schema
            if tables_in_sql and not tables_in_sql.issubset(schema_tables):
                return False
            
            # Additional checks could include column validation, etc.
            return True
            
        except Exception:
            # If parsing fails, assume non-compliant
            return False
    
    def _calculate_bleu_score(self, predicted_sql: str, expected_sql: str) -> float:
        """
        Calculate BLEU score between predicted and expected SQL.
        
        Args:
            predicted_sql: Generated SQL query
            expected_sql: Expected SQL query
            
        Returns:
            BLEU score between 0 and 1
        """
        try:
            # Tokenize SQL queries
            predicted_tokens = self._tokenize_sql(predicted_sql)
            expected_tokens = self._tokenize_sql(expected_sql)
            
            if not predicted_tokens or not expected_tokens:
                return 0.0
            
            # Calculate BLEU score
            score = sentence_bleu(
                [expected_tokens],
                predicted_tokens,
                smoothing_function=self.smoothing_function
            )
            
            return score
            
        except Exception:
            return 0.0
    
    def _normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL query for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL string
        """
        if not sql:
            return ""
        
        # Convert to uppercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', sql.strip().upper())
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        # Sort clauses for better comparison (simplified)
        # This is a basic implementation - more sophisticated normalization
        # would parse the SQL AST and normalize clause order
        
        return normalized
    
    def _tokenize_sql(self, sql: str) -> List[str]:
        """
        Tokenize SQL query for BLEU score calculation.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of tokens
        """
        if not sql:
            return []
        
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', sql.lower())
        return [token for token in tokens if token.strip()]
    
    def create_dummy_models(self) -> Dict[str, Any]:
        """
        Create dummy models for testing evaluation framework.
        
        Returns:
            Dictionary of dummy model instances
        """
        class DummyModel:
            def __init__(self, name: str, accuracy: float = 0.8):
                self.name = name
                self.accuracy = accuracy
            
            def generate_sql(self, question: str, schema_info: str = "") -> Dict[str, Any]:
                # Simulate different model behaviors
                if "customers" in question.lower():
                    if self.name == "baseline":
                        sql = "SELECT * FROM customers"
                    elif self.name == "improved":
                        sql = "SELECT * FROM customers LIMIT 10"
                    else:  # advanced
                        sql = "SELECT customer_id, name, region FROM customers ORDER BY name"
                
                elif "sales" in question.lower() or "revenue" in question.lower():
                    if self.name == "baseline":
                        sql = "SELECT SUM(amount) FROM orders"
                    elif self.name == "improved":
                        sql = "SELECT SUM(total_amount) as total_sales FROM orders"
                    else:  # advanced
                        sql = "SELECT region, SUM(total_amount) as total_sales FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY region"
                
                elif "top" in question.lower() or "best" in question.lower():
                    if self.name == "baseline":
                        sql = "SELECT * FROM products ORDER BY price DESC"
                    elif self.name == "improved":
                        sql = "SELECT * FROM products ORDER BY price DESC LIMIT 5"
                    else:  # advanced
                        sql = "SELECT p.name, SUM(oi.quantity * oi.unit_price) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id, p.name ORDER BY revenue DESC LIMIT 5"
                
                else:
                    # Default query
                    sql = "SELECT COUNT(*) FROM customers"
                
                # Add some randomness based on accuracy
                if np.random.random() > self.accuracy:
                    # Introduce errors for lower accuracy models
                    sql = sql.replace("SELECT", "SELCT")  # Typo
                
                return {
                    'sql': sql,
                    'confidence': self.accuracy + np.random.normal(0, 0.1)
                }
        
        return {
            'baseline_model': DummyModel("baseline", 0.6),
            'improved_model': DummyModel("improved", 0.8),
            'advanced_model': DummyModel("advanced", 0.9)
        }
    
    def create_test_dataset(self) -> List[Dict[str, Any]]:
        """
        Create a test dataset for model evaluation.
        
        Returns:
            List of test cases with questions and expected SQL
        """
        return [
            {
                'question': 'Show me all customers',
                'expected_sql': 'SELECT * FROM customers',
                'schema': 'customers(customer_id, name, email, region)'
            },
            {
                'question': 'What are the total sales?',
                'expected_sql': 'SELECT SUM(total_amount) as total_sales FROM orders',
                'schema': 'orders(order_id, customer_id, total_amount, order_date)'
            },
            {
                'question': 'Find the top 5 products by revenue',
                'expected_sql': 'SELECT p.name, SUM(oi.quantity * oi.unit_price) as revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_id, p.name ORDER BY revenue DESC LIMIT 5',
                'schema': 'products(product_id, name, price), order_items(order_id, product_id, quantity, unit_price)'
            },
            {
                'question': 'How many customers are in each region?',
                'expected_sql': 'SELECT region, COUNT(*) as customer_count FROM customers GROUP BY region',
                'schema': 'customers(customer_id, name, region)'
            },
            {
                'question': 'What is the average order value?',
                'expected_sql': 'SELECT AVG(total_amount) as avg_order_value FROM orders',
                'schema': 'orders(order_id, customer_id, total_amount)'
            }
        ]
    
    def generate_comparison_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a text report comparing model performance.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Formatted comparison report
        """
        report = ["Model Comparison Report", "=" * 50, ""]
        
        # Overall summary
        report.append("Overall Performance Summary:")
        report.append("-" * 30)
        
        for _, row in results_df.iterrows():
            report.append(f"{row['model']}:")
            report.append(f"  Execution Accuracy: {row['execution_accuracy']:.1%}")
            report.append(f"  Exact Match: {row['exact_match_accuracy']:.1%}")
            report.append(f"  Schema Compliance: {row['schema_compliance_rate']:.1%}")
            report.append(f"  BLEU Score: {row['avg_bleu_score']:.3f}")
            report.append(f"  Avg Response Time: {row['avg_response_time']:.3f}s")
            report.append(f"  Error Rate: {row['error_rate']:.1%}")
            report.append("")
        
        # Best performing model
        best_model = results_df.loc[results_df['execution_accuracy'].idxmax()]
        report.append(f"Best Performing Model: {best_model['model']}")
        report.append(f"  Execution Accuracy: {best_model['execution_accuracy']:.1%}")
        report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 15)
        
        if best_model['execution_accuracy'] < 0.8:
            report.append("• Consider improving model training or prompting strategies")
        
        if results_df['avg_response_time'].max() > 2.0:
            report.append("• Optimize model inference for better response times")
        
        if results_df['schema_compliance_rate'].min() < 0.9:
            report.append("• Improve schema awareness in model training")
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Test the model evaluator
    evaluator = ModelEvaluator()
    
    print("Testing Model Evaluator:")
    print("=" * 50)
    
    # Create dummy models and test data
    models = evaluator.create_dummy_models()
    test_data = evaluator.create_test_dataset()
    
    print(f"Created {len(models)} dummy models")
    print(f"Created {len(test_data)} test cases")
    print()
    
    # Evaluate models
    results_df = evaluator.evaluate_models(test_data, models)
    
    print("Evaluation Results:")
    print(results_df.round(3))
    print()
    
    # Generate comparison report
    report = evaluator.generate_comparison_report(results_df)
    print(report)