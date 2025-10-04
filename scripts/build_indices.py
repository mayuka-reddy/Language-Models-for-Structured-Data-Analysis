"""
Build RAG search indices for schema-aware retrieval.
Owner: Prem Shah (RAG) & Sharath Gode (Scripts)

Creates BM25 and vector indices from database schema for efficient retrieval.
"""

import os
import json
import argparse
from pathlib import Path
from loguru import logger

# Import RAG components
from rag.pipeline import SchemaAwareRAG
from src.schema.manager import SchemaManager


def build_indices_from_database(
    database_url: str,
    output_dir: str = "rag/indices",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    """Build RAG indices from live database connection."""
    logger.info(f"Building indices from database: {database_url}")
    
    # Initialize schema manager
    schema_manager = SchemaManager(database_url)
    
    try:
        # Refresh schema from database
        schema_context = schema_manager.get_context()
        logger.info(f"Loaded schema with {len(schema_context['schema']['tables'])} tables")
        
        # Initialize RAG pipeline
        rag = SchemaAwareRAG(schema_context)
        
        # Save indices
        rag.save_indices(output_dir)
        
        logger.info(f"Successfully built and saved indices to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to build indices from database: {e}")
        raise


def build_indices_from_schema_file(
    schema_file: str,
    output_dir: str = "rag/indices",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> None:
    """Build RAG indices from schema JSON file."""
    logger.info(f"Building indices from schema file: {schema_file}")
    
    # Load schema from file
    with open(schema_file, 'r') as f:
        schema_context = json.load(f)
    
    logger.info(f"Loaded schema with {len(schema_context.get('tables', {}))} tables")
    
    # Wrap in expected format if needed
    if 'schema' not in schema_context:
        schema_context = {'schema': schema_context}
    
    # Initialize RAG pipeline
    rag = SchemaAwareRAG(schema_context)
    
    # Save indices
    rag.save_indices(output_dir)
    
    logger.info(f"Successfully built and saved indices to {output_dir}")


def create_sample_schema() -> dict:
    """Create sample schema for testing."""
    return {
        'tables': {
            'customers': {
                'columns': {
                    'customer_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'customer_unique_id': {'type': 'VARCHAR', 'primary_key': False, 'nullable': False},
                    'customer_zip_code_prefix': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'customer_city': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'customer_state': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['customer_id'],
                'foreign_keys': []
            },
            'orders': {
                'columns': {
                    'order_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'customer_id': {'type': 'INTEGER', 'primary_key': False, 'nullable': False, 'foreign_key': True},
                    'order_status': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'order_purchase_timestamp': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'order_approved_at': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'order_delivered_carrier_date': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'order_delivered_customer_date': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'order_estimated_delivery_date': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['order_id'],
                'foreign_keys': [
                    {'column': 'customer_id', 'references_table': 'customers', 'references_column': 'customer_id'}
                ]
            },
            'order_items': {
                'columns': {
                    'order_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False, 'foreign_key': True},
                    'order_item_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'product_id': {'type': 'INTEGER', 'primary_key': False, 'nullable': False, 'foreign_key': True},
                    'seller_id': {'type': 'INTEGER', 'primary_key': False, 'nullable': False, 'foreign_key': True},
                    'shipping_limit_date': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'price': {'type': 'DECIMAL', 'primary_key': False, 'nullable': False},
                    'freight_value': {'type': 'DECIMAL', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['order_id', 'order_item_id'],
                'foreign_keys': [
                    {'column': 'order_id', 'references_table': 'orders', 'references_column': 'order_id'},
                    {'column': 'product_id', 'references_table': 'products', 'references_column': 'product_id'},
                    {'column': 'seller_id', 'references_table': 'sellers', 'references_column': 'seller_id'}
                ]
            },
            'products': {
                'columns': {
                    'product_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'product_category_name': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'product_name_lenght': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'product_description_lenght': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'product_photos_qty': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'product_weight_g': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'product_length_cm': {'type': 'DECIMAL', 'primary_key': False, 'nullable': True},
                    'product_height_cm': {'type': 'DECIMAL', 'primary_key': False, 'nullable': True},
                    'product_width_cm': {'type': 'DECIMAL', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['product_id'],
                'foreign_keys': []
            },
            'sellers': {
                'columns': {
                    'seller_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'seller_zip_code_prefix': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'seller_city': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'seller_state': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['seller_id'],
                'foreign_keys': []
            },
            'order_reviews': {
                'columns': {
                    'review_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'order_id': {'type': 'INTEGER', 'primary_key': False, 'nullable': False, 'foreign_key': True},
                    'review_score': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'review_comment_title': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'review_comment_message': {'type': 'TEXT', 'primary_key': False, 'nullable': True},
                    'review_creation_date': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True},
                    'review_answer_timestamp': {'type': 'TIMESTAMP', 'primary_key': False, 'nullable': True}
                },
                'primary_key': ['review_id'],
                'foreign_keys': [
                    {'column': 'order_id', 'references_table': 'orders', 'references_column': 'order_id'}
                ]
            },
            'order_payments': {
                'columns': {
                    'order_id': {'type': 'INTEGER', 'primary_key': True, 'nullable': False, 'foreign_key': True},
                    'payment_sequential': {'type': 'INTEGER', 'primary_key': True, 'nullable': False},
                    'payment_type': {'type': 'VARCHAR', 'primary_key': False, 'nullable': True},
                    'payment_installments': {'type': 'INTEGER', 'primary_key': False, 'nullable': True},
                    'payment_value': {'type': 'DECIMAL', 'primary_key': False, 'nullable': False}
                },
                'primary_key': ['order_id', 'payment_sequential'],
                'foreign_keys': [
                    {'column': 'order_id', 'references_table': 'orders', 'references_column': 'order_id'}
                ]
            }
        }
    }


def test_indices(indices_dir: str) -> None:
    """Test the built indices with sample queries."""
    logger.info(f"Testing indices in {indices_dir}")
    
    # Load indices
    schema_cards_file = Path(indices_dir) / "schema_cards.json"
    if not schema_cards_file.exists():
        logger.error(f"Schema cards file not found: {schema_cards_file}")
        return
    
    with open(schema_cards_file, 'r') as f:
        schema_cards = json.load(f)
    
    logger.info(f"Loaded {len(schema_cards)} schema cards")
    
    # Test queries
    test_queries = [
        "Which city has the most customers?",
        "What is the total order value?",
        "How many products are there?",
        "What are the payment methods?",
        "Show me customer information"
    ]
    
    # Simple text-based search test
    for query in test_queries:
        query_lower = query.lower()
        matches = []
        
        for card in schema_cards:
            if any(word in card['content'].lower() for word in query_lower.split()):
                matches.append(card['name'])
        
        logger.info(f"Query: '{query}' -> Matches: {matches[:3]}")  # Top 3 matches


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build RAG indices for schema-aware retrieval")
    
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL for live schema reflection"
    )
    
    parser.add_argument(
        "--schema-file",
        type=str,
        help="Path to schema JSON file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rag/indices",
        help="Output directory for indices"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
    )
    
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample schema and indices for testing"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the built indices"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if args.create_sample:
            # Create sample schema
            logger.info("Creating sample schema and indices...")
            sample_schema = create_sample_schema()
            
            # Save sample schema
            schema_file = Path(args.output_dir).parent / "sample_schema.json"
            with open(schema_file, 'w') as f:
                json.dump(sample_schema, f, indent=2)
            
            # Build indices from sample schema
            build_indices_from_schema_file(str(schema_file), args.output_dir, args.embedding_model)
            
        elif args.database_url:
            # Build from database
            build_indices_from_database(args.database_url, args.output_dir, args.embedding_model)
            
        elif args.schema_file:
            # Build from schema file
            build_indices_from_schema_file(args.schema_file, args.output_dir, args.embedding_model)
            
        else:
            # Try to find schema file or use sample
            schema_candidates = [
                "data/processed/schema.json",
                "data/schema.json",
                "schema.json"
            ]
            
            schema_file = None
            for candidate in schema_candidates:
                if Path(candidate).exists():
                    schema_file = candidate
                    break
            
            if schema_file:
                logger.info(f"Found schema file: {schema_file}")
                build_indices_from_schema_file(schema_file, args.output_dir, args.embedding_model)
            else:
                logger.info("No schema source provided, creating sample indices...")
                args.create_sample = True
                # Recursively call with sample creation
                return main()
        
        # Test indices if requested
        if args.test:
            test_indices(args.output_dir)
        
        logger.info("Index building completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to build indices: {e}")
        raise


if __name__ == "__main__":
    main()