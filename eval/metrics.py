"""
Evaluation metrics for NL-to-SQL systems.
Owner: Sri Gopi Sarath Gode

Implements comprehensive evaluation metrics:
1. Execution Correctness (ExecCorrect)
2. Exact Match (EM)
3. Schema Compliance
4. BLEU Score for natural language explanations
"""

import re
import json
import sqlite3
import sqlparse
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None

try:
    import sqlglot
except ImportError:
    sqlglot = None


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    execution_correct: bool
    exact_match: bool
    schema_compliant: bool
    bleu_score: float
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'execution_correct': self.execution_correct,
            'exact_match': self.exact_match,
            'schema_compliant': self.schema_compliant,
            'bleu_score': self.bleu_score,
            'error_message': self.error_message,
            'execution_time': self.execution_time
        }


class SQLExecutor:
    """Safe SQL execution for evaluation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self._connect()
    
    def _connect(self) -> None:
        """Connect to database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def execute_query(self, sql: str, timeout: int = 30) -> Tuple[List[Dict], Optional[str]]:
        """
        Execute SQL query safely with timeout.
        
        Returns:
            Tuple of (results, error_message)
        """
        if not self._is_safe_query(sql):
            return [], "Unsafe query detected"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            
            # Fetch results
            rows = cursor.fetchall()
            results = [dict(row) for row in rows]
            
            return results, None
            
        except Exception as e:
            return [], str(e)
    
    def _is_safe_query(self, sql: str) -> bool:
        """Check if query is safe (read-only)."""
        # Convert to uppercase for checking
        sql_upper = sql.upper().strip()
        
        # Dangerous keywords
        dangerous_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
            'TRUNCATE', 'REPLACE', 'MERGE', 'CALL', 'EXEC'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        
        return True
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get database schema information."""
        schema_info = {'tables': {}}
        
        try:
            # Get table names
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get column info for each table
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                schema_info['tables'][table] = {
                    'columns': {
                        col[1]: {  # col[1] is column name
                            'type': col[2],  # col[2] is data type
                            'nullable': not col[3],  # col[3] is not null
                            'primary_key': bool(col[5])  # col[5] is pk
                        }
                        for col in columns
                    }
                }
            
            return schema_info
            
        except Exception as e:
            print(f"Error getting schema info: {e}")
            return {'tables': {}}


class SchemaValidator:
    """Validates SQL queries against database schema."""
    
    def __init__(self, schema_info: Dict[str, Any]):
        self.schema_info = schema_info
    
    def validate_query(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL query against schema.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Parse SQL
            parsed = sqlparse.parse(sql)[0]
            
            # Extract table and column references
            tables_used, columns_used = self._extract_references(parsed)
            
            # Validate tables exist
            for table in tables_used:
                if table not in self.schema_info['tables']:
                    errors.append(f"Table '{table}' does not exist")
            
            # Validate columns exist in their tables
            for table, columns in columns_used.items():
                if table in self.schema_info['tables']:
                    table_columns = self.schema_info['tables'][table]['columns']
                    for column in columns:
                        if column not in table_columns:
                            errors.append(f"Column '{column}' does not exist in table '{table}'")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"SQL parsing error: {str(e)}")
            return False, errors
    
    def _extract_references(self, parsed_sql) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract table and column references from parsed SQL."""
        tables_used = []
        columns_used = {}
        
        # This is a simplified extraction - in production would use a proper SQL parser
        sql_text = str(parsed_sql).upper()
        
        # Extract table names after FROM and JOIN
        from_pattern = r'FROM\s+(\w+)'
        join_pattern = r'JOIN\s+(\w+)'
        
        tables_used.extend(re.findall(from_pattern, sql_text))
        tables_used.extend(re.findall(join_pattern, sql_text))
        
        # Remove duplicates
        tables_used = list(set(tables_used))
        
        # For simplicity, assume all columns belong to first table
        # In production, would need proper SQL AST parsing
        if tables_used:
            columns_used[tables_used[0]] = []
        
        return tables_used, columns_used


class NL2SQLEvaluator:
    """Main evaluator for NL-to-SQL systems."""
    
    def __init__(self, db_path: str, schema_info: Optional[Dict[str, Any]] = None):
        self.executor = SQLExecutor(db_path)
        
        if schema_info is None:
            schema_info = self.executor.get_schema_info()
        
        self.schema_validator = SchemaValidator(schema_info)
        self.smoothing_function = SmoothingFunction().method1 if SmoothingFunction else None
    
    def evaluate_single(
        self, 
        predicted_sql: str, 
        ground_truth_sql: str,
        question: Optional[str] = None,
        predicted_explanation: Optional[str] = None,
        ground_truth_explanation: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a single prediction."""
        
        # 1. Execution Correctness
        exec_correct = self._check_execution_correctness(predicted_sql, ground_truth_sql)
        
        # 2. Exact Match
        exact_match = self._check_exact_match(predicted_sql, ground_truth_sql)
        
        # 3. Schema Compliance
        schema_compliant, schema_errors = self.schema_validator.validate_query(predicted_sql)
        
        # 4. BLEU Score (if explanations provided)
        bleu_score = 0.0
        if predicted_explanation and ground_truth_explanation:
            bleu_score = self._calculate_bleu(predicted_explanation, ground_truth_explanation)
        
        # Collect any error messages
        error_message = None
        if not schema_compliant:
            error_message = "; ".join(schema_errors)
        
        return EvaluationResult(
            execution_correct=exec_correct,
            exact_match=exact_match,
            schema_compliant=schema_compliant,
            bleu_score=bleu_score,
            error_message=error_message
        )
    
    def evaluate_batch(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of predictions."""
        results = []
        
        for pred in predictions:
            result = self.evaluate_single(
                predicted_sql=pred['predicted_sql'],
                ground_truth_sql=pred['ground_truth_sql'],
                question=pred.get('question'),
                predicted_explanation=pred.get('predicted_explanation'),
                ground_truth_explanation=pred.get('ground_truth_explanation')
            )
            results.append(result)
        
        # Aggregate metrics
        total = len(results)
        if total == 0:
            return {}
        
        metrics = {
            'execution_accuracy': sum(r.execution_correct for r in results) / total,
            'exact_match_accuracy': sum(r.exact_match for r in results) / total,
            'schema_compliance_rate': sum(r.schema_compliant for r in results) / total,
            'average_bleu': sum(r.bleu_score for r in results) / total,
            'total_samples': total
        }
        
        # Add detailed results
        metrics['detailed_results'] = [r.to_dict() for r in results]
        
        return metrics
    
    def _check_execution_correctness(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Check if predicted SQL produces same results as ground truth."""
        try:
            # Execute both queries
            pred_results, pred_error = self.executor.execute_query(predicted_sql)
            gt_results, gt_error = self.executor.execute_query(ground_truth_sql)
            
            # If either query failed, not correct
            if pred_error or gt_error:
                return False
            
            # Compare results (order-independent)
            return self._compare_query_results(pred_results, gt_results)
            
        except Exception:
            return False
    
    def _compare_query_results(self, results1: List[Dict], results2: List[Dict]) -> bool:
        """Compare two query result sets."""
        if len(results1) != len(results2):
            return False
        
        # Convert to comparable format (handle different column orders)
        def normalize_result(result):
            return tuple(sorted(result.items()))
        
        normalized1 = {normalize_result(r) for r in results1}
        normalized2 = {normalize_result(r) for r in results2}
        
        return normalized1 == normalized2
    
    def _check_exact_match(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """Check if SQL queries are exactly the same (normalized)."""
        return self._normalize_sql(predicted_sql) == self._normalize_sql(ground_truth_sql)
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', sql.strip().lower())
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        return normalized
    
    def _calculate_bleu(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score between predicted and reference text."""
        if not sentence_bleu:
            return 0.0
        
        try:
            # Tokenize
            reference_tokens = reference.lower().split()
            predicted_tokens = predicted.lower().split()
            
            # Calculate BLEU score
            score = sentence_bleu(
                [reference_tokens], 
                predicted_tokens,
                smoothing_function=self.smoothing_function
            )
            
            return score
            
        except Exception:
            return 0.0


def create_sample_evaluation_data(output_path: str) -> None:
    """Create sample evaluation data for testing."""
    sample_data = [
        {
            "question": "Which city has the highest number of customers?",
            "predicted_sql": "SELECT city, COUNT(*) as customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1",
            "ground_truth_sql": "SELECT city, COUNT(*) as count FROM customer GROUP BY city ORDER BY count DESC LIMIT 1",
            "predicted_explanation": "This query groups customers by city and counts them, then orders by count descending to find the city with most customers.",
            "ground_truth_explanation": "Groups customers by city, counts them, and returns the city with the highest count."
        },
        {
            "question": "What was the total sales in August 2025?",
            "predicted_sql": "SELECT SUM(total_amount) as total_sales FROM orders WHERE order_date >= '2025-08-01' AND order_date < '2025-09-01'",
            "ground_truth_sql": "SELECT SUM(total_amount) FROM orders WHERE EXTRACT(MONTH FROM order_date) = 8 AND EXTRACT(YEAR FROM order_date) = 2025",
            "predicted_explanation": "Sums the total_amount for all orders in August 2025 using date range filtering.",
            "ground_truth_explanation": "Calculates total sales for August 2025 by summing order amounts."
        }
    ]
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample evaluation data created at {output_path}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    create_sample_evaluation_data("eval/sample_predictions.json")
    
    # Note: This would require an actual database to run
    # evaluator = NL2SQLEvaluator("data/processed/olist.db")
    # 
    # with open("eval/sample_predictions.json", 'r') as f:
    #     predictions = json.load(f)
    # 
    # results = evaluator.evaluate_batch(predictions)
    # print("Evaluation Results:", json.dumps(results, indent=2))