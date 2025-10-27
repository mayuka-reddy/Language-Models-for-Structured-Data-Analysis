"""
Multi-technique comparison module for NL-to-SQL models.
Provides comprehensive comparison and analysis of different ML techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import json
import time
from pathlib import Path

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class TechniqueResult:
    """Container for individual technique evaluation results."""
    technique_name: str
    predictions: List[str]
    execution_times: List[float]
    confidence_scores: List[float]
    metadata: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Container for comparison results between techniques."""
    technique_results: Dict[str, TechniqueResult]
    comparative_metrics: Dict[str, Any]
    best_technique: str
    recommendations: List[str]


class ModelComparator:
    """
    Comprehensive model and technique comparator.
    
    Evaluates and compares multiple ML techniques including:
    - Prompting strategies (CoT, Few-Shot, Self-Consistency, Least-to-Most)
    - RAG pipeline performance
    - Fine-tuned model results
    """
    
    def __init__(self, metrics_calculator=None):
        """
        Initialize the comparator.
        
        Args:
            metrics_calculator: Optional metrics calculator instance
        """
        self.metrics_calculator = metrics_calculator
        self.comparison_history = []
    
    def compare_techniques(
        self,
        test_queries: List[Dict[str, Any]],
        techniques: Dict[str, Callable],
        ground_truth: List[str]
    ) -> ComparisonResult:
        """
        Compare multiple techniques on a set of test queries.
        
        Args:
            test_queries: List of test queries with metadata
            techniques: Dictionary of technique names to callable functions
            ground_truth: List of expected SQL queries
            
        Returns:
            ComparisonResult with detailed analysis
        """
        logger.info(f"Comparing {len(techniques)} techniques on {len(test_queries)} queries")
        
        technique_results = {}
        
        # Evaluate each technique
        for technique_name, technique_func in techniques.items():
            logger.info(f"Evaluating technique: {technique_name}")
            
            predictions = []
            execution_times = []
            confidence_scores = []
            
            for query_data in test_queries:
                # Measure execution time
                start_time = time.time()
                
                try:
                    # Call technique function
                    result = technique_func(query_data)
                    
                    # Extract prediction and confidence
                    if isinstance(result, dict):
                        prediction = result.get('sql', '')
                        confidence = result.get('confidence', 0.5)
                    else:
                        prediction = str(result)
                        confidence = 0.5
                    
                    execution_time = time.time() - start_time
                    
                    predictions.append(prediction)
                    execution_times.append(execution_time)
                    confidence_scores.append(confidence)
                    
                except Exception as e:
                    logger.error(f"Error in {technique_name}: {e}")
                    predictions.append("")
                    execution_times.append(float('inf'))
                    confidence_scores.append(0.0)
            
            # Store results
            technique_results[technique_name] = TechniqueResult(
                technique_name=technique_name,
                predictions=predictions,
                execution_times=execution_times,
                confidence_scores=confidence_scores,
                metadata={'total_queries': len(test_queries)}
            )
        
        # Calculate comparative metrics
        comparative_metrics = self._calculate_comparative_metrics(
            technique_results, ground_truth
        )
        
        # Determine best technique
        best_technique = self._determine_best_technique(comparative_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(comparative_metrics, technique_results)
        
        # Create comparison result
        comparison_result = ComparisonResult(
            technique_results=technique_results,
            comparative_metrics=comparative_metrics,
            best_technique=best_technique,
            recommendations=recommendations
        )
        
        # Store in history
        self.comparison_history.append(comparison_result)
        
        return comparison_result
    
    def _calculate_comparative_metrics(
        self,
        technique_results: Dict[str, TechniqueResult],
        ground_truth: List[str]
    ) -> Dict[str, Any]:
        """Calculate comprehensive comparative metrics."""
        
        metrics = {}
        
        for technique_name, result in technique_results.items():
            technique_metrics = {}
            
            # Basic accuracy metrics
            if self.metrics_calculator:
                # Use comprehensive metrics calculator if available
                evaluation_results = []
                for pred, gt in zip(result.predictions, ground_truth):
                    eval_result = self.metrics_calculator.evaluate_single_prediction(pred, gt)
                    evaluation_results.append(eval_result)
                
                # Aggregate metrics
                total = len(evaluation_results)
                technique_metrics.update({
                    'execution_accuracy': sum(r.execution_correct for r in evaluation_results) / total,
                    'exact_match_accuracy': sum(r.exact_match for r in evaluation_results) / total,
                    'schema_compliance_rate': sum(r.schema_compliant for r in evaluation_results) / total,
                    'avg_bleu_score': sum(r.bleu_score for r in evaluation_results) / total,
                })
            else:
                # Fallback to simple metrics
                technique_metrics.update({
                    'execution_accuracy': self._calculate_simple_accuracy(result.predictions, ground_truth),
                    'exact_match_accuracy': self._calculate_exact_match(result.predictions, ground_truth),
                    'schema_compliance_rate': 0.8,  # Placeholder
                    'avg_bleu_score': 0.7,  # Placeholder
                })
            
            # Performance metrics
            technique_metrics.update({
                'avg_response_time': np.mean(result.execution_times),
                'median_response_time': np.median(result.execution_times),
                'response_time_std': np.std(result.execution_times),
                'avg_confidence': np.mean(result.confidence_scores),
                'confidence_std': np.std(result.confidence_scores),
                'success_rate': sum(1 for t in result.execution_times if t != float('inf')) / len(result.execution_times),
                'total_queries': len(result.predictions)
            })
            
            # Quality metrics
            technique_metrics.update({
                'prediction_length_avg': np.mean([len(p) for p in result.predictions if p]),
                'non_empty_predictions': sum(1 for p in result.predictions if p.strip()),
                'confidence_accuracy_correlation': self._calculate_confidence_correlation(
                    result.confidence_scores, result.predictions, ground_truth
                )
            })
            
            metrics[technique_name] = technique_metrics
        
        return metrics
    
    def _calculate_simple_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate simple accuracy based on string similarity."""
        if not predictions or not ground_truth:
            return 0.0
        
        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            # Normalize both strings
            pred_norm = pred.strip().upper().replace(';', '')
            gt_norm = gt.strip().upper().replace(';', '')
            
            # Simple similarity check
            if pred_norm == gt_norm:
                correct += 1
            elif pred_norm and gt_norm:
                # Check token overlap
                pred_tokens = set(pred_norm.split())
                gt_tokens = set(gt_norm.split())
                
                if pred_tokens and gt_tokens:
                    overlap = len(pred_tokens.intersection(gt_tokens))
                    similarity = overlap / len(pred_tokens.union(gt_tokens))
                    if similarity > 0.7:  # Threshold for "correct"
                        correct += 1
        
        return correct / len(predictions)
    
    def _calculate_exact_match(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate exact match accuracy."""
        if not predictions or not ground_truth:
            return 0.0
        
        exact_matches = sum(
            1 for pred, gt in zip(predictions, ground_truth)
            if pred.strip().upper() == gt.strip().upper()
        )
        
        return exact_matches / len(predictions)
    
    def _calculate_confidence_correlation(
        self,
        confidence_scores: List[float],
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """Calculate correlation between confidence and accuracy."""
        if len(confidence_scores) != len(predictions) or len(predictions) != len(ground_truth):
            return 0.0
        
        # Calculate per-prediction accuracy
        accuracies = []
        for pred, gt in zip(predictions, ground_truth):
            pred_norm = pred.strip().upper()
            gt_norm = gt.strip().upper()
            accuracies.append(1.0 if pred_norm == gt_norm else 0.0)
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(confidence_scores, accuracies)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _determine_best_technique(self, comparative_metrics: Dict[str, Any]) -> str:
        """Determine the best performing technique based on weighted metrics."""
        
        if not comparative_metrics:
            return "unknown"
        
        # Define weights for different metrics (higher weight = more important)
        weights = {
            'execution_accuracy': 0.3,
            'exact_match_accuracy': 0.2,
            'schema_compliance_rate': 0.15,
            'avg_bleu_score': 0.1,
            'avg_response_time': -0.1,  # Negative because lower is better
            'success_rate': 0.1,
            'avg_confidence': 0.05
        }
        
        technique_scores = {}
        
        for technique_name, metrics in comparative_metrics.items():
            score = 0.0
            
            for metric, weight in weights.items():
                if metric in metrics:
                    value = metrics[metric]
                    
                    # Normalize response time (invert and scale)
                    if metric == 'avg_response_time':
                        # Convert to score (lower time = higher score)
                        max_time = max(m.get('avg_response_time', 1) for m in comparative_metrics.values())
                        normalized_value = 1 - (value / max_time) if max_time > 0 else 1
                        score += weight * normalized_value
                    else:
                        score += weight * value
            
            technique_scores[technique_name] = score
        
        # Return technique with highest score
        best_technique = max(technique_scores.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Best technique determined: {best_technique}")
        logger.info(f"Technique scores: {technique_scores}")
        
        return best_technique
    
    def _generate_recommendations(
        self,
        comparative_metrics: Dict[str, Any],
        technique_results: Dict[str, TechniqueResult]
    ) -> List[str]:
        """Generate actionable recommendations based on comparison results."""
        
        recommendations = []
        
        # Analyze overall performance
        avg_accuracy = np.mean([m.get('execution_accuracy', 0) for m in comparative_metrics.values()])
        avg_response_time = np.mean([m.get('avg_response_time', 0) for m in comparative_metrics.values()])
        
        if avg_accuracy < 0.7:
            recommendations.append(
                "Overall accuracy is below 70%. Consider improving training data quality or model architecture."
            )
        
        if avg_response_time > 2.0:
            recommendations.append(
                "Average response time exceeds 2 seconds. Consider model optimization or caching strategies."
            )
        
        # Technique-specific recommendations
        for technique_name, metrics in comparative_metrics.items():
            accuracy = metrics.get('execution_accuracy', 0)
            response_time = metrics.get('avg_response_time', 0)
            confidence = metrics.get('avg_confidence', 0)
            
            if accuracy < 0.6:
                recommendations.append(
                    f"{technique_name}: Low accuracy ({accuracy:.1%}). "
                    f"Consider technique-specific improvements or parameter tuning."
                )
            
            if response_time > 3.0:
                recommendations.append(
                    f"{technique_name}: High response time ({response_time:.1f}s). "
                    f"Optimize for faster inference."
                )
            
            if confidence < 0.5:
                recommendations.append(
                    f"{technique_name}: Low confidence scores ({confidence:.2f}). "
                    f"Review confidence calibration."
                )
        
        # Best practices recommendations
        best_accuracy_technique = max(
            comparative_metrics.items(),
            key=lambda x: x[1].get('execution_accuracy', 0)
        )[0]
        
        fastest_technique = min(
            comparative_metrics.items(),
            key=lambda x: x[1].get('avg_response_time', float('inf'))
        )[0]
        
        recommendations.append(
            f"For highest accuracy, use {best_accuracy_technique}. "
            f"For fastest response, use {fastest_technique}."
        )
        
        return recommendations
    
    def generate_comparison_report(self, comparison_result: ComparisonResult) -> str:
        """Generate a comprehensive comparison report."""
        
        report_lines = [
            "ML Technique Comparison Report",
            "=" * 50,
            "",
            f"Techniques Evaluated: {len(comparison_result.technique_results)}",
            f"Best Performing Technique: {comparison_result.best_technique}",
            "",
            "Performance Summary:",
            "-" * 30
        ]
        
        # Add technique summaries
        for technique_name, metrics in comparison_result.comparative_metrics.items():
            report_lines.extend([
                f"\n{technique_name}:",
                f"  Execution Accuracy: {metrics.get('execution_accuracy', 0):.1%}",
                f"  Exact Match: {metrics.get('exact_match_accuracy', 0):.1%}",
                f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s",
                f"  Success Rate: {metrics.get('success_rate', 0):.1%}",
                f"  Avg Confidence: {metrics.get('avg_confidence', 0):.2f}"
            ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "Recommendations:",
            "-" * 15
        ])
        
        for i, rec in enumerate(comparison_result.recommendations, 1):
            report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)
    
    def save_comparison_results(self, comparison_result: ComparisonResult, output_path: str) -> None:
        """Save comparison results to file."""
        
        # Prepare data for serialization
        serializable_data = {
            'best_technique': comparison_result.best_technique,
            'comparative_metrics': comparison_result.comparative_metrics,
            'recommendations': comparison_result.recommendations,
            'technique_summaries': {}
        }
        
        # Add technique summaries
        for name, result in comparison_result.technique_results.items():
            serializable_data['technique_summaries'][name] = {
                'total_predictions': len(result.predictions),
                'avg_execution_time': np.mean(result.execution_times),
                'avg_confidence': np.mean(result.confidence_scores),
                'metadata': result.metadata
            }
        
        # Save to JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Comparison results saved to {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Test the comparator
    comparator = ModelComparator()
    
    print("Testing Model Comparator:")
    print("=" * 50)
    
    # Mock test data
    test_queries = [
        {"question": "Show all customers", "schema": "customers(id, name, city)"},
        {"question": "Count orders by region", "schema": "orders(id, region, amount)"},
        {"question": "Find top products", "schema": "products(id, name, sales)"}
    ]
    
    ground_truth = [
        "SELECT * FROM customers",
        "SELECT region, COUNT(*) FROM orders GROUP BY region",
        "SELECT * FROM products ORDER BY sales DESC LIMIT 10"
    ]
    
    # Mock technique functions
    def mock_technique_1(query_data):
        return {
            'sql': f"SELECT * FROM table WHERE condition",
            'confidence': 0.8
        }
    
    def mock_technique_2(query_data):
        return {
            'sql': f"SELECT column FROM table",
            'confidence': 0.9
        }
    
    techniques = {
        'Chain-of-Thought': mock_technique_1,
        'Few-Shot': mock_technique_2
    }
    
    # Run comparison
    result = comparator.compare_techniques(test_queries, techniques, ground_truth)
    
    # Generate and print report
    report = comparator.generate_comparison_report(result)
    print(report)
    
    # Save results
    comparator.save_comparison_results(result, "models/evaluation/comparison_results.json")
    
    print("\nComparison testing completed successfully!")