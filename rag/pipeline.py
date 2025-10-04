"""
Schema-aware RAG pipeline for NL-to-SQL generation.
Owner: Prem Shah

Implements hybrid retrieval combining:
1. BM25 keyword search over schema metadata
2. Dense embeddings for semantic similarity
3. Schema-aware reranking and context assembly
"""

from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass

# Placeholder imports - would be real implementations
try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
    import numpy as np
except ImportError:
    # Graceful fallback for development
    SentenceTransformer = None
    BM25Okapi = None
    faiss = None
    np = None


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source_type: str  # 'table', 'column', 'relationship'


class SchemaCardBuilder:
    """Builds searchable schema cards from database metadata."""
    
    def __init__(self, schema_context: Dict[str, Any]):
        self.schema_context = schema_context
        self.cards = self._build_cards()
    
    def _build_cards(self) -> List[Dict[str, Any]]:
        """Build schema cards for tables, columns, and relationships."""
        cards = []
        
        if 'tables' not in self.schema_context:
            return cards
        
        # Table-level cards
        for table_name, table_info in self.schema_context['tables'].items():
            card = {
                'id': f"table_{table_name}",
                'type': 'table',
                'name': table_name,
                'content': f"Table: {table_name}",
                'columns': list(table_info.get('columns', {}).keys()),
                'primary_keys': table_info.get('primary_key', []),
                'foreign_keys': table_info.get('foreign_keys', [])
            }
            cards.append(card)
        
        # Column-level cards
        for table_name, table_info in self.schema_context['tables'].items():
            for col_name, col_info in table_info.get('columns', {}).items():
                card = {
                    'id': f"column_{table_name}_{col_name}",
                    'type': 'column',
                    'name': f"{table_name}.{col_name}",
                    'content': f"Column: {table_name}.{col_name} ({col_info.get('type', 'unknown')})",
                    'table': table_name,
                    'data_type': col_info.get('type', 'unknown'),
                    'nullable': col_info.get('nullable', True),
                    'is_primary_key': col_info.get('primary_key', False),
                    'is_foreign_key': col_info.get('foreign_key', False)
                }
                cards.append(card)
        
        return cards
    
    def get_cards_by_type(self, card_type: str) -> List[Dict[str, Any]]:
        """Get cards filtered by type."""
        return [card for card in self.cards if card['type'] == card_type]
    
    def search_cards(self, query: str, card_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple text search over cards."""
        query_lower = query.lower()
        results = []
        
        for card in self.cards:
            if card_type and card['type'] != card_type:
                continue
            
            if query_lower in card['content'].lower() or query_lower in card['name'].lower():
                results.append(card)
        
        return results


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense embeddings."""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.bm25_index = None
        self.vector_index = None
        self.documents = []
        self.document_embeddings = None
        
        # Initialize embedding model if available
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
            except Exception:
                print(f"Warning: Could not load embedding model {embedding_model_name}")
    
    def build_indices(self, schema_cards: List[Dict[str, Any]]) -> None:
        """Build BM25 and vector indices from schema cards."""
        self.documents = schema_cards
        
        # Build BM25 index
        if BM25Okapi:
            corpus = [card['content'].split() for card in schema_cards]
            self.bm25_index = BM25Okapi(corpus)
        
        # Build vector index
        if self.embedding_model and faiss and np:
            texts = [card['content'] for card in schema_cards]
            embeddings = self.embedding_model.encode(texts)
            self.document_embeddings = embeddings
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.vector_index.add(embeddings.astype('float32'))
    
    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using BM25 keyword search."""
        if not self.bm25_index:
            return []
        
        query_tokens = query.split()
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append(RetrievalResult(
                    content=self.documents[idx]['content'],
                    score=float(scores[idx]),
                    metadata=self.documents[idx],
                    source_type=self.documents[idx]['type']
                ))
        
        return results
    
    def retrieve_dense(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve using dense embeddings."""
        if not self.embedding_model or not self.vector_index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search vector index
        scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append(RetrievalResult(
                    content=self.documents[idx]['content'],
                    score=float(score),
                    metadata=self.documents[idx],
                    source_type=self.documents[idx]['type']
                ))
        
        return results
    
    def hybrid_retrieve(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[RetrievalResult]:
        """Combine BM25 and dense retrieval with weighted scoring."""
        bm25_results = self.retrieve_bm25(query, top_k)
        dense_results = self.retrieve_dense(query, top_k)
        
        # Combine and rerank
        combined_results = {}
        
        # Add BM25 results
        for result in bm25_results:
            doc_id = result.metadata['id']
            combined_results[doc_id] = result
            combined_results[doc_id].score *= (1 - alpha)  # Weight BM25 score
        
        # Add dense results
        for result in dense_results:
            doc_id = result.metadata['id']
            if doc_id in combined_results:
                # Combine scores
                combined_results[doc_id].score += result.score * alpha
            else:
                result.score *= alpha  # Weight dense score
                combined_results[doc_id] = result
        
        # Sort by combined score
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        return final_results[:top_k]


class SchemaAwareRAG:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, schema_context: Dict[str, Any]):
        self.schema_context = schema_context
        self.card_builder = SchemaCardBuilder(schema_context)
        self.retriever = HybridRetriever()
        
        # Build indices
        self.retriever.build_indices(self.card_builder.cards)
    
    def retrieve_context(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Retrieve relevant schema context for a query."""
        # Get hybrid retrieval results
        results = self.retriever.hybrid_retrieve(query, top_k)
        
        # Organize results by type
        context = {
            'relevant_tables': [],
            'relevant_columns': [],
            'relationships': [],
            'retrieval_scores': []
        }
        
        for result in results:
            if result.source_type == 'table':
                context['relevant_tables'].append(result.metadata)
            elif result.source_type == 'column':
                context['relevant_columns'].append(result.metadata)
            
            context['retrieval_scores'].append({
                'content': result.content,
                'score': result.score,
                'type': result.source_type
            })
        
        return context
    
    def augment_prompt(self, base_prompt: str, query: str, top_k: int = 5) -> str:
        """Augment a base prompt with retrieved schema context."""
        context = self.retrieve_context(query, top_k)
        
        # Build context string
        context_str = "## Retrieved Schema Context:\n"
        
        if context['relevant_tables']:
            context_str += "### Relevant Tables:\n"
            for table in context['relevant_tables'][:3]:  # Limit to top 3
                context_str += f"- {table['name']}: {len(table['columns'])} columns\n"
        
        if context['relevant_columns']:
            context_str += "### Relevant Columns:\n"
            for col in context['relevant_columns'][:5]:  # Limit to top 5
                context_str += f"- {col['name']} ({col['data_type']})\n"
        
        # Insert context into prompt
        augmented_prompt = f"{base_prompt}\n\n{context_str}\n"
        return augmented_prompt
    
    def save_indices(self, output_dir: str) -> None:
        """Save built indices to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save schema cards
        with open(output_path / "schema_cards.json", 'w') as f:
            json.dump(self.card_builder.cards, f, indent=2)
        
        # Save vector embeddings if available
        if self.retriever.document_embeddings is not None and np:
            np.save(output_path / "embeddings.npy", self.retriever.document_embeddings)
        
        print(f"RAG indices saved to {output_dir}")
    
    def load_indices(self, input_dir: str) -> None:
        """Load pre-built indices from disk."""
        input_path = Path(input_dir)
        
        # Load schema cards
        cards_file = input_path / "schema_cards.json"
        if cards_file.exists():
            with open(cards_file, 'r') as f:
                self.card_builder.cards = json.load(f)
            
            # Rebuild indices
            self.retriever.build_indices(self.card_builder.cards)
            print(f"RAG indices loaded from {input_dir}")
        else:
            print(f"No indices found at {input_dir}")


# Example usage and testing
if __name__ == "__main__":
    # Mock schema for testing
    mock_schema = {
        'tables': {
            'customer': {
                'columns': {
                    'customer_id': {'type': 'INTEGER', 'primary_key': True},
                    'city': {'type': 'VARCHAR', 'nullable': False},
                    'state': {'type': 'VARCHAR', 'nullable': False}
                },
                'primary_key': ['customer_id'],
                'foreign_keys': []
            },
            'orders': {
                'columns': {
                    'order_id': {'type': 'INTEGER', 'primary_key': True},
                    'customer_id': {'type': 'INTEGER', 'foreign_key': True},
                    'total_amount': {'type': 'DECIMAL', 'nullable': False}
                },
                'primary_key': ['order_id'],
                'foreign_keys': [{'column': 'customer_id', 'references_table': 'customer', 'references_column': 'customer_id'}]
            }
        }
    }
    
    # Test RAG pipeline
    rag = SchemaAwareRAG(mock_schema)
    
    # Test retrieval
    query = "Which city has the most customers?"
    context = rag.retrieve_context(query)
    print("Retrieved context:", context)
    
    # Test prompt augmentation
    base_prompt = "Generate SQL for the following question:"
    augmented = rag.augment_prompt(base_prompt, query)
    print("Augmented prompt:", augmented)