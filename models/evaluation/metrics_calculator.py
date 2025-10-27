"""
Comprehensive metrics calculator for NL-to-SQL model evaluation.
Consolidates functionality from app/metrics.py and eval/metrics.py

Implements comprehensive evaluation metrics:
1. Execution Correctness (ExecCorrect)
2. Exact Match (EM)
3. Schema Compliance
4. BLEU Score for SQL similarity
5. Response time analysis
"""

import re
import json
import sqlite3
import sqlparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import time

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    NLTK_AVAILABLE = True
except ImportError:
    sentence_bleu = None
    SmoothingFunction = None
    NLTK_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    execution_correct: bool
    exact_match: bool
    schema_compliant: bool
    bleu_score: float
    response_time: float
    confidence_score: float
    error_message: Optional[str] = None
    predicted_sql: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'execution_correct': self.execution_correct,
            'exact_match': self.exact_match,
            'schema_compliant': self.schema_compliant,
            'bleu_score': self.bleu_score,
            'response_time': self.response_time,
            'confidence_score': self.confidence_score,
            'error_message': self.error_message,
            'predicted_sql': self.predicted_sql
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
            logger.error(f"Error getting schema info: {e}")
            return {'tables': {}}


class MetricsCalculator:
    """
    Comprehensive metrics calculator for NL-to-SQL evaluation.
    
    Supports evaluation of execution correctness, exact match,
    schema compliance, and BLEU scores for model comparison.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize metrics calculator."""
        self.db_path = db_path
        self.executor = SQLExecutor(db_path) if db_path else None
        self.schema_info = self.executor.get_schema_info() if self.executor else {}
        
        if NLTK_AVAILABLE:
            self.smoothing_function = SmoothingFunction().method1
        else:
            self.smoothing_function = None
            
        logger.info("Metrics Calculator initialized")
    
    def calculate_execution_correctness(self, predicted_sql: str, ground_truth_sql: str) -> bool:
        """
        Check if predicted SQL produces same results as ground truth.
        
        Args:
            predicted_sql: Generated SQL query
            ground_truth_sql: Expected SQL query
            
        Returns:
            True if execution results match, False otherwise
        """
        if not self.executor:
            # Fallback to structural similarity if no database
            return self._check_structural_similarity(predicted_sql, ground_truth_sql)
        
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
    
    def calculate_bleu_scores(self, predictions: List[str], references: List[str]) -> float:
        """
        Calculate BLEU scores between predicted and reference SQL queries.
        
        Args:
            predictions: List of predicted SQL queries
            references: List of reference SQL queries
            
        Returns:
            Average BLEU score
        """
        if not NLTK_AVAILABLE or not predictions or not references:
            return 0.0
        
        scores = []
        for pred, ref in zip(predictions, references):
            score = self._calculate_single_bleu(pred, ref)
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def evaluate_schema_compliance(self, sql_queries: List[str], schema: Optional[Dict] = None) -> float:
        """
        Evaluate schema compliance for a list of SQL queries.
        
        Args:
            sql_queries: List of SQL queries to evaluate
            schema: Optional schema information (uses self.schema_info if None)
            
        Returns:
            Compliance rate (0.0 to 1.0)
        """
        if not sql_queries:
            return 0.0
        
        schema_to_use = schema or self.schema_info
        if not schema_to_use:
            return 0.0
        
        compliant_count = 0
        for sql in sql_queries:
            if self._check_schema_compliance(sql, schema_to_use):
                compliant_count += 1
        
        return compliant_count / len(sql_queries)
    
    def measure_response_times(self, technique: str, queries: List[str], model_func) -> List[float]:
        """
        Measure response times for a list of queries using a specific technique.
        
        Args:
            technique: Name of the technique being measured
            queries: List of queries to process
            model_func: Function that takes a query and returns a result
            
        Returns:
            List of response times in seconds
        """
        response_times = []
        
        logger.info(f"Measuring response times for {technique} on {len(queries)} queries")
        
        for i, query in enumerate(queries):
            start_time = time.perf_counter()  # More precise timing
            try:
                _ = model_func(query)
                response_time = time.perf_counter() - start_time
                response_times.append(response_time)
                
                if (i + 1) % 10 == 0:  # Log progress every 10 queries
                    logger.info(f"Processed {i + 1}/{len(queries)} queries for {technique}")
                    
            except Exception as e:
                logger.warning(f"Error processing query '{query}' with {technique}: {e}")
                response_times.append(float('inf'))  # Mark as failed
        
        # Calculate statistics
        valid_times = [t for t in response_times if t != float('inf')]
        if valid_times:
            avg_time = np.mean(valid_times)
            median_time = np.median(valid_times)
            logger.info(f"{technique} - Avg: {avg_time:.3f}s, Median: {median_time:.3f}s, Success: {len(valid_times)}/{len(queries)}")
        
        return response_times
    
    def generate_comprehensive_report(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report from results.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary containing comprehensive metrics and analysis
        """
        if not evaluation_results:
            return {}
        
        total = len(evaluation_results)
        
        # Calculate aggregate metrics
        metrics = {
            'total_samples': total,
            'execution_accuracy': sum(r.execution_correct for r in evaluation_results) / total,
            'exact_match_accuracy': sum(r.exact_match for r in evaluation_results) / total,
            'schema_compliance_rate': sum(r.schema_compliant for r in evaluation_results) / total,
            'average_bleu_score': sum(r.bleu_score for r in evaluation_results) / total,
            'average_response_time': sum(r.response_time for r in evaluation_results) / total,
            'average_confidence': sum(r.confidence_score for r in evaluation_results) / total,
            'error_rate': sum(1 for r in evaluation_results if r.error_message) / total
        }
        
        # Add distribution statistics
        response_times = [r.response_time for r in evaluation_results if r.response_time != float('inf')]
        bleu_scores = [r.bleu_score for r in evaluation_results]
        confidence_scores = [r.confidence_score for r in evaluation_results]
        
        if response_times:
            metrics['response_time_stats'] = {
                'min': min(response_times),
                'max': max(response_times),
                'median': np.median(response_times),
                'std': np.std(response_times),
                'p95': np.percentile(response_times, 95),
                'p99': np.percentile(response_times, 99)
            }
        else:
            metrics['response_time_stats'] = {'min': 0, 'max': 0, 'median': 0, 'std': 0, 'p95': 0, 'p99': 0}
        
        metrics['bleu_score_stats'] = {
            'min': min(bleu_scores) if bleu_scores else 0,
            'max': max(bleu_scores) if bleu_scores else 0,
            'median': np.median(bleu_scores) if bleu_scores else 0,
            'std': np.std(bleu_scores) if bleu_scores else 0,
            'p25': np.percentile(bleu_scores, 25) if bleu_scores else 0,
            'p75': np.percentile(bleu_scores, 75) if bleu_scores else 0
        }
        
        metrics['confidence_stats'] = {
            'min': min(confidence_scores) if confidence_scores else 0,
            'max': max(confidence_scores) if confidence_scores else 0,
            'median': np.median(confidence_scores) if confidence_scores else 0,
            'std': np.std(confidence_scores) if confidence_scores else 0,
            'calibration_score': self._calculate_confidence_calibration(evaluation_results)
        }
        
        # Add query complexity analysis
        metrics['complexity_analysis'] = self._analyze_query_complexity(evaluation_results)
        
        # Add performance by query type
        metrics['performance_by_type'] = self._analyze_performance_by_type(evaluation_results)
        
        # Add detailed results (optional, can be large)
        metrics['detailed_results'] = [r.to_dict() for r in evaluation_results]
        
        return metrics
    
    def evaluate_single_prediction(
        self, 
        predicted_sql: str, 
        ground_truth_sql: str,
        question: Optional[str] = None,
        confidence_score: float = 0.0,
        response_time: float = 0.0
    ) -> EvaluationResult:
        """Evaluate a single prediction comprehensively."""
        
        error_message = None
        
        try:
            # 1. Execution Correctness
            exec_correct = self.calculate_execution_correctness(predicted_sql, ground_truth_sql)
            
            # 2. Exact Match
            exact_match = self._check_exact_match(predicted_sql, ground_truth_sql)
            
            # 3. Schema Compliance
            schema_compliant = self._check_schema_compliance(predicted_sql, self.schema_info)
            
            # 4. BLEU Score
            bleu_score = self._calculate_single_bleu(predicted_sql, ground_truth_sql)
            
        except Exception as e:
            logger.error(f"Error evaluating prediction: {e}")
            error_message = str(e)
            exec_correct = False
            exact_match = False
            schema_compliant = False
            bleu_score = 0.0
        
        return EvaluationResult(
            execution_correct=exec_correct,
            exact_match=exact_match,
            schema_compliant=schema_compliant,
            bleu_score=bleu_score,
            response_time=response_time,
            confidence_score=confidence_score,
            predicted_sql=predicted_sql,
            error_message=error_message
        )
    
    def batch_evaluate_predictions(
        self,
        predictions: List[str],
        ground_truths: List[str],
        questions: Optional[List[str]] = None,
        confidence_scores: Optional[List[float]] = None,
        response_times: Optional[List[float]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple predictions in batch for efficiency.
        
        Args:
            predictions: List of predicted SQL queries
            ground_truths: List of ground truth SQL queries
            questions: Optional list of questions
            confidence_scores: Optional list of confidence scores
            response_times: Optional list of response times
            
        Returns:
            List of evaluation results
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        # Prepare optional parameters
        questions = questions or [None] * len(predictions)
        confidence_scores = confidence_scores or [0.0] * len(predictions)
        response_times = response_times or [0.0] * len(predictions)
        
        results = []
        total = len(predictions)
        
        logger.info(f"Batch evaluating {total} predictions")
        
        for i, (pred, gt, q, conf, rt) in enumerate(zip(
            predictions, ground_truths, questions, confidence_scores, response_times
        )):
            result = self.evaluate_single_prediction(pred, gt, q, conf, rt)
            results.append(result)
            
            if (i + 1) % 50 == 0:  # Log progress every 50 evaluations
                logger.info(f"Evaluated {i + 1}/{total} predictions")
        
        logger.info(f"Batch evaluation completed: {total} predictions processed")
        return results
    
    def calculate_statistical_significance(
        self,
        results_a: List[EvaluationResult],
        results_b: List[EvaluationResult],
        metric: str = 'execution_correct'
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance between two sets of results.
        
        Args:
            results_a: First set of evaluation results
            results_b: Second set of evaluation results
            metric: Metric to compare ('execution_correct', 'bleu_score', etc.)
            
        Returns:
            Dictionary with statistical test results
        """
        try:
            from scipy import stats
            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False
            
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, cannot calculate statistical significance")
            return {'error': 'scipy not available'}
        
        # Extract metric values
        if metric == 'execution_correct':
            values_a = [1.0 if r.execution_correct else 0.0 for r in results_a]
            values_b = [1.0 if r.execution_correct else 0.0 for r in results_b]
        elif metric == 'bleu_score':
            values_a = [r.bleu_score for r in results_a]
            values_b = [r.bleu_score for r in results_b]
        elif metric == 'response_time':
            values_a = [r.response_time for r in results_a if r.response_time != float('inf')]
            values_b = [r.response_time for r in results_b if r.response_time != float('inf')]
        else:
            return {'error': f'Unsupported metric: {metric}'}
        
        if len(values_a) == 0 or len(values_b) == 0:
            return {'error': 'No valid values for comparison'}
        
        # Perform statistical tests
        try:
            # T-test for means
            t_stat, t_pvalue = stats.ttest_ind(values_a, values_b)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) + 
                                 (len(values_b) - 1) * np.var(values_b, ddof=1)) / 
                                (len(values_a) + len(values_b) - 2))
            
            cohens_d = (np.mean(values_a) - np.mean(values_b)) / pooled_std if pooled_std > 0 else 0
            
            return {
                'metric': metric,
                'sample_size_a': len(values_a),
                'sample_size_b': len(values_b),
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'std_a': np.std(values_a),
                'std_b': np.std(values_b),
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'u_statistic': u_stat,
                'u_pvalue': u_pvalue,
                'cohens_d': cohens_d,
                'significant_at_05': min(t_pvalue, u_pvalue) < 0.05,
                'significant_at_01': min(t_pvalue, u_pvalue) < 0.01
            }
            
        except Exception as e:
            return {'error': f'Statistical test failed: {e}'}
    
    def _check_structural_similarity(self, predicted_sql: str, expected_sql: str) -> bool:
        """
        Check structural similarity when database execution is not available.
        """
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
    
    def _check_schema_compliance(self, predicted_sql: str, schema_info: Dict[str, Any]) -> bool:
        """
        Check if predicted SQL complies with schema constraints.
        """
        if not predicted_sql.strip() or not schema_info:
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
            if 'tables' in schema_info:
                schema_tables = set(schema_info['tables'].keys())
                
                # Check if all tables in SQL exist in schema
                if tables_in_sql and not tables_in_sql.issubset(schema_tables):
                    return False
            
            return True
            
        except Exception:
            # If parsing fails, assume non-compliant
            return False
    
    def _calculate_single_bleu(self, predicted_sql: str, expected_sql: str) -> float:
        """
        Calculate BLEU score between predicted and expected SQL.
        """
        if not NLTK_AVAILABLE:
            return 0.0
        
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
        """
        if not sql:
            return ""
        
        # Convert to uppercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', sql.strip().upper())
        
        # Remove trailing semicolon
        normalized = normalized.rstrip(';')
        
        return normalized
    
    def _tokenize_sql(self, sql: str) -> List[str]:
        """
        Tokenize SQL query for BLEU score calculation.
        """
        if not sql:
            return []
        
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\w+|[^\w\s]', sql.lower())
        return [token for token in tokens if token.strip()]
    
    def _calculate_confidence_calibration(self, evaluation_results: List[EvaluationResult]) -> float:
        """
        Calculate confidence calibration score (how well confidence correlates with accuracy).
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Calibration score between 0 and 1 (higher is better)
        """
        if len(evaluation_results) < 2:
            return 0.0
        
        confidences = [r.confidence_score for r in evaluation_results]
        accuracies = [1.0 if r.execution_correct else 0.0 for r in evaluation_results]
        
        try:
            # Calculate correlation between confidence and accuracy
            correlation = np.corrcoef(confidences, accuracies)[0, 1]
            return max(0.0, correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _analyze_query_complexity(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Analyze performance by query complexity.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary with complexity analysis
        """
        complexity_analysis = {
            'simple_queries': {'count': 0, 'accuracy': 0.0},
            'medium_queries': {'count': 0, 'accuracy': 0.0},
            'complex_queries': {'count': 0, 'accuracy': 0.0}
        }
        
        for result in evaluation_results:
            # Determine complexity based on SQL length and keywords
            sql = result.predicted_sql.upper()
            complexity = self._determine_query_complexity(sql)
            
            complexity_analysis[f'{complexity}_queries']['count'] += 1
            if result.execution_correct:
                complexity_analysis[f'{complexity}_queries']['accuracy'] += 1
        
        # Calculate accuracy rates
        for complexity_type in complexity_analysis:
            count = complexity_analysis[complexity_type]['count']
            if count > 0:
                accuracy = complexity_analysis[complexity_type]['accuracy'] / count
                complexity_analysis[complexity_type]['accuracy'] = accuracy
        
        return complexity_analysis
    
    def _determine_query_complexity(self, sql: str) -> str:
        """
        Determine query complexity based on SQL features.
        
        Args:
            sql: SQL query string
            
        Returns:
            Complexity level: 'simple', 'medium', or 'complex'
        """
        if not sql:
            return 'simple'
        
        sql_upper = sql.upper()
        
        # Count complex features
        complex_features = 0
        
        # Check for joins
        if 'JOIN' in sql_upper:
            complex_features += 2
        
        # Check for subqueries
        if sql_upper.count('SELECT') > 1:
            complex_features += 2
        
        # Check for aggregations
        agg_functions = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'GROUP BY']
        for func in agg_functions:
            if func in sql_upper:
                complex_features += 1
                break
        
        # Check for window functions
        window_keywords = ['OVER', 'PARTITION BY', 'ROW_NUMBER', 'RANK']
        for keyword in window_keywords:
            if keyword in sql_upper:
                complex_features += 2
                break
        
        # Check for multiple conditions
        if sql_upper.count('WHERE') > 0 and ('AND' in sql_upper or 'OR' in sql_upper):
            complex_features += 1
        
        # Classify based on features
        if complex_features >= 4:
            return 'complex'
        elif complex_features >= 2:
            return 'medium'
        else:
            return 'simple'
    
    def _analyze_performance_by_type(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Analyze performance by query type (SELECT, INSERT, etc.).
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary with performance by query type
        """
        type_analysis = {}
        
        for result in evaluation_results:
            query_type = self._determine_query_type(result.predicted_sql)
            
            if query_type not in type_analysis:
                type_analysis[query_type] = {
                    'count': 0,
                    'execution_correct': 0,
                    'exact_match': 0,
                    'avg_bleu_score': 0.0,
                    'avg_response_time': 0.0
                }
            
            stats = type_analysis[query_type]
            stats['count'] += 1
            
            if result.execution_correct:
                stats['execution_correct'] += 1
            if result.exact_match:
                stats['exact_match'] += 1
            
            stats['avg_bleu_score'] += result.bleu_score
            if result.response_time != float('inf'):
                stats['avg_response_time'] += result.response_time
        
        # Calculate averages
        for query_type in type_analysis:
            stats = type_analysis[query_type]
            count = stats['count']
            
            if count > 0:
                stats['execution_accuracy'] = stats['execution_correct'] / count
                stats['exact_match_accuracy'] = stats['exact_match'] / count
                stats['avg_bleu_score'] = stats['avg_bleu_score'] / count
                stats['avg_response_time'] = stats['avg_response_time'] / count
        
        return type_analysis
    
    def _determine_query_type(self, sql: str) -> str:
        """
        Determine the type of SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            Query type string
        """
        if not sql:
            return 'unknown'
        
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            # Further classify SELECT queries
            if 'GROUP BY' in sql_upper or any(agg in sql_upper for agg in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']):
                return 'select_aggregate'
            elif 'JOIN' in sql_upper:
                return 'select_join'
            else:
                return 'select_simple'
        elif sql_upper.startswith('INSERT'):
            return 'insert'
        elif sql_upper.startswith('UPDATE'):
            return 'update'
        elif sql_upper.startswith('DELETE'):
            return 'delete'
        elif sql_upper.startswith('CREATE'):
            return 'create'
        else:
            return 'unknown'


# Example usage and testing
if __name__ == "__main__":
    # Test the metrics calculator
    calculator = MetricsCalculator()
    
    print("Testing Metrics Calculator:")
    print("=" * 50)
    
    # Test single evaluation
    predicted = "SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY COUNT(*) DESC LIMIT 1"
    ground_truth = "SELECT city, COUNT(*) as count FROM customers GROUP BY city ORDER BY count DESC LIMIT 1"
    
    result = calculator.evaluate_single_prediction(predicted, ground_truth)
    print("Single evaluation result:")
    print(result.to_dict())
    print()
    
    # Test batch evaluation
    predictions = [predicted, "SELECT * FROM customers"]
    references = [ground_truth, "SELECT * FROM customers"]
    
    bleu_score = calculator.calculate_bleu_scores(predictions, references)
    print(f"BLEU score: {bleu_score:.3f}")