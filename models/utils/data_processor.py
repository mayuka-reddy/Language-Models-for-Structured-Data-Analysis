"""
Data preparation utilities for NL-to-SQL training and evaluation.
Handles data loading, preprocessing, validation, and augmentation for the 
Olist Brazilian E-Commerce dataset.

Dataset Information:
- Olist Brazilian E-Commerce Public Dataset (Kaggle)
- ~100,000 customer orders from 2016-2018
- Two main views:
  * Item-Level Dataset (112,650 rows, 37 columns)
  * Order-Level Dataset (98,666 rows, 13 columns)
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality analysis results."""
    total_samples: int
    valid_samples: int
    invalid_samples: int
    avg_question_length: float
    avg_sql_length: float
    unique_questions: int
    duplicate_questions: int
    schema_coverage: Dict[str, int]
    quality_issues: List[str]


@dataclass
class OlistDatasetInfo:
    """Information about the Olist Brazilian E-Commerce dataset."""
    item_level_rows: int = 112650
    item_level_columns: int = 37
    order_level_rows: int = 98666
    order_level_columns: int = 13
    date_range: str = "2016-2018"
    
    # Key tables and columns for NL-to-SQL generation
    key_tables: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.key_tables is None:
            self.key_tables = {
                "item_level": [
                    "order_id", "product_id", "seller_id", "customer_id",
                    "product_category_name", "product_weight_g", "product_length_cm",
                    "product_height_cm", "product_width_cm", "price", "freight_value",
                    "total_payment_value", "payment_type", "payment_installments",
                    "order_status", "order_purchase_timestamp", "order_delivered_timestamp",
                    "delivery_days", "delivery_delay_days", "review_score", "n_reviews"
                ],
                "order_level": [
                    "order_id", "customer_id", "order_status", "order_purchase_timestamp",
                    "total_payment_value", "total_price", "total_freight", "avg_review_score",
                    "n_reviews", "n_sellers", "n_products", "avg_delivery_days", "avg_delivery_delay_days"
                ]
            }


class DataProcessor:
    """
    Comprehensive data processor for NL-to-SQL datasets.
    
    Handles data loading, validation, preprocessing, and augmentation
    for the Olist Brazilian E-Commerce dataset training and evaluation.
    """
    
    def __init__(self, output_dir: str = "models/training/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.olist_info = OlistDatasetInfo()
    
    def load_and_validate_data(self, data_path: str) -> Tuple[List[Dict[str, Any]], DataQualityReport]:
        """
        Load and validate NL-to-SQL dataset.
        
        Args:
            data_path: Path to the dataset file
            
        Returns:
            Tuple of (validated_data, quality_report)
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        data = self._load_data_file(data_path)
        
        # Validate and clean data
        validated_data = []
        quality_issues = []
        
        for i, item in enumerate(data):
            validation_result = self._validate_data_item(item, i)
            
            if validation_result['is_valid']:
                validated_data.append(validation_result['cleaned_item'])
            else:
                quality_issues.extend(validation_result['issues'])
        
        # Generate quality report
        quality_report = self._generate_quality_report(data, validated_data, quality_issues)
        
        logger.info(f"Data validation complete: {len(validated_data)}/{len(data)} samples valid")
        
        return validated_data, quality_report
    
    def _load_data_file(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from various file formats."""
        file_path = Path(data_path)
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {data_path}")
            return []
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            
            elif file_path.suffix.lower() == '.jsonl':
                data = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
            
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
                return df.to_dict('records')
            
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            return []
    
    def _validate_data_item(self, item: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate and clean a single data item."""
        
        issues = []
        is_valid = True
        cleaned_item = item.copy()
        
        # Check required fields
        required_fields = ['question', 'sql']
        for field in required_fields:
            if field not in item or not item[field]:
                issues.append(f"Item {index}: Missing or empty '{field}' field")
                is_valid = False
        
        if not is_valid:
            return {'is_valid': False, 'issues': issues, 'cleaned_item': None}
        
        # Clean and validate question
        question = str(item['question']).strip()
        if len(question) < 5:
            issues.append(f"Item {index}: Question too short ({len(question)} chars)")
            is_valid = False
        elif len(question) > 500:
            issues.append(f"Item {index}: Question too long ({len(question)} chars)")
            # Truncate but keep as valid
            question = question[:500]
        
        cleaned_item['question'] = question
        
        # Clean and validate SQL
        sql = str(item['sql']).strip()
        if not self._is_valid_sql(sql):
            issues.append(f"Item {index}: Invalid SQL syntax")
            is_valid = False
        
        # Clean SQL formatting
        cleaned_sql = self._clean_sql(sql)
        cleaned_item['sql'] = cleaned_sql
        
        # Validate schema context if present
        if 'schema_context' in item:
            schema_validation = self._validate_schema_context(item['schema_context'], index)
            if not schema_validation['is_valid']:
                issues.extend(schema_validation['issues'])
            else:
                cleaned_item['schema_context'] = schema_validation['cleaned_schema']
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'cleaned_item': cleaned_item if is_valid else None
        }
    
    def _is_valid_sql(self, sql: str) -> bool:
        """Basic SQL syntax validation."""
        if not sql or not sql.strip():
            return False
        
        sql_upper = sql.upper().strip()
        
        # Must start with SELECT, INSERT, UPDATE, DELETE, or WITH
        valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH']
        if not any(sql_upper.startswith(start) for start in valid_starts):
            return False
        
        # Basic syntax checks
        if sql_upper.startswith('SELECT'):
            if 'FROM' not in sql_upper and 'DUAL' not in sql_upper:
                return False
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            return False
        
        return True
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL formatting."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', sql.strip())
        
        # Remove trailing semicolon
        cleaned = cleaned.rstrip(';')
        
        # Normalize keywords (optional - keep original case for now)
        # This could be expanded to normalize SQL formatting
        
        return cleaned
    
    def _validate_schema_context(self, schema_context: Any, index: int) -> Dict[str, Any]:
        """Validate schema context information."""
        
        issues = []
        is_valid = True
        cleaned_schema = schema_context
        
        if isinstance(schema_context, dict):
            # Validate expected schema fields
            if 'relevant_tables' in schema_context:
                tables = schema_context['relevant_tables']
                if not isinstance(tables, list):
                    issues.append(f"Item {index}: relevant_tables should be a list")
                    is_valid = False
                elif not tables:
                    issues.append(f"Item {index}: relevant_tables is empty")
            
            if 'relevant_columns' in schema_context:
                columns = schema_context['relevant_columns']
                if not isinstance(columns, list):
                    issues.append(f"Item {index}: relevant_columns should be a list")
                    is_valid = False
        
        elif isinstance(schema_context, str):
            # Convert string schema to structured format
            cleaned_schema = self._parse_schema_string(schema_context)
        
        else:
            issues.append(f"Item {index}: Invalid schema_context type")
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'cleaned_schema': cleaned_schema
        }
    
    def _parse_schema_string(self, schema_str: str) -> Dict[str, Any]:
        """Parse schema string into structured format."""
        
        # Simple parser for schema strings like "table1(col1, col2), table2(col3, col4)"
        tables = []
        columns = []
        
        # Extract table definitions
        table_pattern = r'(\w+)\s*\(([^)]+)\)'
        matches = re.findall(table_pattern, schema_str)
        
        for table_name, cols_str in matches:
            tables.append(table_name)
            table_columns = [col.strip() for col in cols_str.split(',')]
            columns.extend([f"{table_name}.{col}" for col in table_columns])
        
        return {
            'relevant_tables': tables,
            'relevant_columns': columns,
            'schema_string': schema_str
        }
    
    def _generate_quality_report(
        self,
        original_data: List[Dict[str, Any]],
        validated_data: List[Dict[str, Any]],
        quality_issues: List[str]
    ) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        
        total_samples = len(original_data)
        valid_samples = len(validated_data)
        invalid_samples = total_samples - valid_samples
        
        if validated_data:
            # Calculate statistics
            question_lengths = [len(item['question']) for item in validated_data]
            sql_lengths = [len(item['sql']) for item in validated_data]
            
            avg_question_length = np.mean(question_lengths)
            avg_sql_length = np.mean(sql_lengths)
            
            # Count unique questions
            questions = [item['question'].lower() for item in validated_data]
            unique_questions = len(set(questions))
            duplicate_questions = len(questions) - unique_questions
            
            # Analyze schema coverage
            schema_coverage = {}
            for item in validated_data:
                if 'schema_context' in item and isinstance(item['schema_context'], dict):
                    tables = item['schema_context'].get('relevant_tables', [])
                    for table in tables:
                        schema_coverage[table] = schema_coverage.get(table, 0) + 1
        
        else:
            avg_question_length = 0.0
            avg_sql_length = 0.0
            unique_questions = 0
            duplicate_questions = 0
            schema_coverage = {}
        
        return DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            avg_question_length=avg_question_length,
            avg_sql_length=avg_sql_length,
            unique_questions=unique_questions,
            duplicate_questions=duplicate_questions,
            schema_coverage=schema_coverage,
            quality_issues=quality_issues
        )
    
    def augment_data(self, data: List[Dict[str, Any]], augmentation_factor: float = 1.5) -> List[Dict[str, Any]]:
        """
        Augment training data with variations.
        
        Args:
            data: Original dataset
            augmentation_factor: Factor by which to increase dataset size
            
        Returns:
            Augmented dataset
        """
        logger.info(f"Augmenting data with factor {augmentation_factor}")
        
        augmented_data = data.copy()
        target_size = int(len(data) * augmentation_factor)
        
        while len(augmented_data) < target_size:
            # Select random item to augment
            original_item = np.random.choice(data)
            
            # Apply augmentation techniques
            augmented_item = self._apply_augmentation(original_item)
            
            if augmented_item:
                augmented_data.append(augmented_item)
        
        logger.info(f"Data augmentation complete: {len(data)} -> {len(augmented_data)} samples")
        
        return augmented_data
    
    def _apply_augmentation(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply augmentation techniques to a single item."""
        
        augmentation_techniques = [
            self._augment_question_paraphrase,
            self._augment_sql_formatting,
            self._augment_column_aliases
        ]
        
        # Randomly select augmentation technique
        technique = np.random.choice(augmentation_techniques)
        
        try:
            return technique(item)
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return None
    
    def _augment_question_paraphrase(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create paraphrased version of the question."""
        
        question = item['question']
        
        # Simple paraphrasing rules
        paraphrases = {
            'show me': 'display',
            'find': 'get',
            'what is': 'what are',
            'how many': 'count',
            'list all': 'show all'
        }
        
        paraphrased = question.lower()
        for original, replacement in paraphrases.items():
            if original in paraphrased:
                paraphrased = paraphrased.replace(original, replacement)
                break
        
        # Capitalize first letter
        paraphrased = paraphrased[0].upper() + paraphrased[1:] if paraphrased else question
        
        augmented_item = item.copy()
        augmented_item['question'] = paraphrased
        
        return augmented_item
    
    def _augment_sql_formatting(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Create differently formatted version of SQL."""
        
        sql = item['sql']
        
        # Apply formatting variations
        formatting_options = [
            lambda s: s.upper(),  # All uppercase
            lambda s: s.lower(),  # All lowercase
            lambda s: re.sub(r'\s+', ' ', s),  # Normalize whitespace
            lambda s: s.replace(',', ', ')  # Add space after commas
        ]
        
        formatter = np.random.choice(formatting_options)
        formatted_sql = formatter(sql)
        
        augmented_item = item.copy()
        augmented_item['sql'] = formatted_sql
        
        return augmented_item
    
    def _augment_column_aliases(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Add or modify column aliases in SQL."""
        
        sql = item['sql']
        
        # Simple alias addition for SELECT statements
        if sql.upper().startswith('SELECT') and 'AS ' not in sql.upper():
            # Add alias to first column if it's a function
            if any(func in sql.upper() for func in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
                # Find the first function and add alias
                for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
                    pattern = f'{func}\\([^)]+\\)'
                    match = re.search(pattern, sql, re.IGNORECASE)
                    if match:
                        alias_name = f"{func.lower()}_result"
                        aliased = f"{match.group()} AS {alias_name}"
                        sql = sql.replace(match.group(), aliased)
                        break
        
        augmented_item = item.copy()
        augmented_item['sql'] = sql
        
        return augmented_item
    
    def split_data(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            data: Complete dataset
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Shuffle data
        shuffled_data = data.copy()
        np.random.shuffle(shuffled_data)
        
        # Calculate split indices
        total_size = len(shuffled_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # Split data
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def save_processed_data(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Save processed data to files.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Optional test dataset
            
        Returns:
            Dictionary of dataset names to file paths
        """
        
        file_paths = {}
        
        # Save training data
        train_path = self.output_dir / "train_pairs.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        file_paths['train'] = str(train_path)
        
        # Save validation data
        val_path = self.output_dir / "eval_pairs.json"
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        file_paths['validation'] = str(val_path)
        
        # Save test data if provided
        if test_data:
            test_path = self.output_dir / "test_pairs.json"
            with open(test_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            file_paths['test'] = str(test_path)
        
        logger.info(f"Processed data saved to {self.output_dir}")
        
        return file_paths
    
    def create_olist_training_examples(self) -> List[Dict[str, Any]]:
        """
        Create NL-to-SQL training examples based on the Olist dataset schema.
        
        Returns:
            List of training examples with questions and SQL queries
        """
        logger.info("Creating Olist-specific NL-to-SQL training examples")
        
        training_examples = []
        
        # Item-level dataset examples
        item_level_examples = [
            {
                "question": "What categories have the highest freight costs?",
                "sql": "SELECT product_category_name, AVG(freight_value) as avg_freight FROM item_level GROUP BY product_category_name ORDER BY avg_freight DESC LIMIT 10",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["product_category_name", "freight_value"],
                    "query_type": "aggregate",
                    "difficulty_level": "medium"
                }
            },
            {
                "question": "Do late shipments decrease review scores?",
                "sql": "SELECT CASE WHEN delivery_delay_days > 0 THEN 'Late' ELSE 'On Time' END as delivery_status, AVG(review_score) as avg_score FROM item_level WHERE review_score IS NOT NULL GROUP BY CASE WHEN delivery_delay_days > 0 THEN 'Late' ELSE 'On Time' END",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["delivery_delay_days", "review_score"],
                    "query_type": "aggregate",
                    "difficulty_level": "hard"
                }
            },
            {
                "question": "Which payment type is most popular?",
                "sql": "SELECT payment_type, COUNT(*) as payment_count FROM item_level GROUP BY payment_type ORDER BY payment_count DESC LIMIT 1",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["payment_type"],
                    "query_type": "aggregate",
                    "difficulty_level": "easy"
                }
            },
            {
                "question": "What is the average delivery time for each product category?",
                "sql": "SELECT product_category_name, AVG(delivery_days) as avg_delivery_days FROM item_level WHERE delivery_days IS NOT NULL GROUP BY product_category_name ORDER BY avg_delivery_days",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["product_category_name", "delivery_days"],
                    "query_type": "aggregate",
                    "difficulty_level": "medium"
                }
            },
            {
                "question": "Show me orders with the highest total payment value",
                "sql": "SELECT order_id, total_payment_value FROM item_level ORDER BY total_payment_value DESC LIMIT 10",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["order_id", "total_payment_value"],
                    "query_type": "select",
                    "difficulty_level": "easy"
                }
            },
            {
                "question": "What are the dimensions of the heaviest products?",
                "sql": "SELECT product_id, product_weight_g, product_length_cm, product_height_cm, product_width_cm FROM item_level WHERE product_weight_g IS NOT NULL ORDER BY product_weight_g DESC LIMIT 5",
                "schema_context": {
                    "relevant_tables": ["item_level"],
                    "relevant_columns": ["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"],
                    "query_type": "select",
                    "difficulty_level": "medium"
                }
            }
        ]
        
        # Order-level dataset examples
        order_level_examples = [
            {
                "question": "How many orders were delivered successfully?",
                "sql": "SELECT COUNT(*) as delivered_orders FROM order_level WHERE order_status = 'delivered'",
                "schema_context": {
                    "relevant_tables": ["order_level"],
                    "relevant_columns": ["order_status"],
                    "query_type": "aggregate",
                    "difficulty_level": "easy"
                }
            },
            {
                "question": "What is the average number of products per order?",
                "sql": "SELECT AVG(n_products) as avg_products_per_order FROM order_level",
                "schema_context": {
                    "relevant_tables": ["order_level"],
                    "relevant_columns": ["n_products"],
                    "query_type": "aggregate",
                    "difficulty_level": "easy"
                }
            },
            {
                "question": "Which orders have multiple sellers?",
                "sql": "SELECT order_id, n_sellers FROM order_level WHERE n_sellers > 1 ORDER BY n_sellers DESC",
                "schema_context": {
                    "relevant_tables": ["order_level"],
                    "relevant_columns": ["order_id", "n_sellers"],
                    "query_type": "select",
                    "difficulty_level": "medium"
                }
            },
            {
                "question": "Show the monthly order trends for 2017",
                "sql": "SELECT DATE_TRUNC('month', order_purchase_timestamp) as month, COUNT(*) as order_count FROM order_level WHERE EXTRACT(year FROM order_purchase_timestamp) = 2017 GROUP BY DATE_TRUNC('month', order_purchase_timestamp) ORDER BY month",
                "schema_context": {
                    "relevant_tables": ["order_level"],
                    "relevant_columns": ["order_purchase_timestamp"],
                    "query_type": "aggregate",
                    "difficulty_level": "hard"
                }
            },
            {
                "question": "What is the correlation between delivery delay and review scores?",
                "sql": "SELECT CASE WHEN avg_delivery_delay_days > 0 THEN 'Delayed' ELSE 'On Time' END as delivery_status, AVG(avg_review_score) as avg_score, COUNT(*) as order_count FROM order_level WHERE avg_review_score IS NOT NULL GROUP BY CASE WHEN avg_delivery_delay_days > 0 THEN 'Delayed' ELSE 'On Time' END",
                "schema_context": {
                    "relevant_tables": ["order_level"],
                    "relevant_columns": ["avg_delivery_delay_days", "avg_review_score"],
                    "query_type": "aggregate",
                    "difficulty_level": "hard"
                }
            }
        ]
        
        # Combine all examples
        training_examples.extend(item_level_examples)
        training_examples.extend(order_level_examples)
        
        logger.info(f"Created {len(training_examples)} Olist training examples")
        
        return training_examples
    
    def generate_olist_schema_context(self) -> Dict[str, Any]:
        """
        Generate schema context information for the Olist dataset.
        
        Returns:
            Dictionary containing schema information for RAG and prompting
        """
        
        schema_context = {
            "dataset_name": "Olist Brazilian E-Commerce",
            "dataset_description": "Brazilian e-commerce platform data with customer orders from 2016-2018",
            "total_orders": "~100,000",
            "date_range": "2016-2018",
            "views": {
                "item_level": {
                    "description": "Item-level data per order with detailed product and shipping information",
                    "rows": self.olist_info.item_level_rows,
                    "columns": self.olist_info.item_level_columns,
                    "key_columns": self.olist_info.key_tables["item_level"]
                },
                "order_level": {
                    "description": "Order-level aggregated data with summary statistics per order",
                    "rows": self.olist_info.order_level_rows,
                    "columns": self.olist_info.order_level_columns,
                    "key_columns": self.olist_info.key_tables["order_level"]
                }
            },
            "common_query_patterns": [
                "Product category analysis",
                "Delivery performance metrics",
                "Payment method preferences",
                "Review score correlations",
                "Geographic distribution",
                "Temporal trends",
                "Seller performance",
                "Customer behavior"
            ],
            "sample_questions": [
                "What categories have the highest freight costs?",
                "Do late shipments decrease review scores?",
                "Which payment type is most popular?",
                "How many orders were delivered successfully?",
                "What is the average number of products per order?"
            ]
        }
        
        return schema_context


# Example usage and testing
if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    
    print("Testing Data Processor:")
    print("=" * 50)
    
    # Create Olist training examples
    olist_examples = processor.create_olist_training_examples()
    print(f"Created {len(olist_examples)} Olist training examples")
    
    # Generate schema context
    schema_context = processor.generate_olist_schema_context()
    print(f"Schema context: {schema_context['dataset_name']}")
    
    # Use first few examples for testing
    sample_data = olist_examples[:5]
    
    # Test validation
    validated_data, quality_report = processor.load_and_validate_data("dummy_path")
    print(f"Quality report: {quality_report}")
    
    # Test augmentation
    augmented_data = processor.augment_data(sample_data, 2.0)
    print(f"Augmented data size: {len(augmented_data)}")
    
    # Test data splitting
    train_data, val_data, test_data = processor.split_data(sample_data)
    print(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Test saving
    file_paths = processor.save_processed_data(train_data, val_data, test_data)
    print(f"Saved files: {file_paths}")
    
    print("\nData processor testing completed successfully!")