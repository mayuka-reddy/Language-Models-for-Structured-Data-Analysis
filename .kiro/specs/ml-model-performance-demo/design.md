# Design Document

## Overview

The ML Model Performance Demo system will create a comprehensive demonstration of multiple machine learning techniques for NL-to-SQL conversion. The design focuses on implementing a unified performance evaluation notebook that showcases prompting strategies, RAG pipeline, and supervised fine-tuning with complete training and evaluation workflows.

## Architecture

```
models/
├── performance_demo.ipynb          # Main demonstration notebook
├── training/
│   ├── sft_trainer.py             # Enhanced SFT implementation
│   ├── model_checkpoints/         # Saved trained models
│   └── training_data/             # Processed training datasets
├── evaluation/
│   ├── metrics_calculator.py      # Comprehensive metrics
│   ├── comparator.py             # Multi-technique comparison
│   └── visualizer.py             # Performance visualization
├── techniques/
│   ├── prompting_strategies.py    # Enhanced prompting engine
│   ├── rag_pipeline.py           # Schema-aware RAG
│   └── baseline_models.py        # Baseline implementations
└── utils/
    ├── data_processor.py         # Data preparation utilities
    ├── config_manager.py         # Centralized configuration
    └── environment_setup.py      # Environment validation
```

## Components and Interfaces

### 1. Performance Demo Notebook (`models/performance_demo.ipynb`)

**Purpose:** Central demonstration notebook showcasing all ML techniques

**Key Sections:**
- Environment setup and dependency validation
- Olist Brazilian E-Commerce dataset preparation and analysis
- Prompting strategies evaluation with e-commerce domain queries (CoT, Few-Shot, Self-Consistency, Least-to-Most)
- RAG pipeline demonstration with e-commerce schema retrieval metrics
- Supervised fine-tuning with Brazilian e-commerce domain training workflow
- Comparative analysis and unique insights for business intelligence applications
- Model performance visualization for e-commerce query patterns

**Interface:**
```python
class PerformanceDemo:
    def setup_environment() -> bool
    def load_and_analyze_data() -> Dict[str, Any]
    def evaluate_prompting_strategies() -> Dict[str, float]
    def demonstrate_rag_pipeline() -> Dict[str, Any]
    def run_supervised_fine_tuning() -> str  # Returns model path
    def compare_all_techniques() -> pd.DataFrame
    def generate_insights() -> List[str]
```

### 2. Enhanced SFT Trainer (`models/training/sft_trainer.py`)

**Purpose:** Complete supervised fine-tuning implementation with model saving

**Key Features:**
- Full parameter fine-tuning using HuggingFace transformers
- Training progress tracking and visualization
- Model checkpointing and saving
- Evaluation metrics calculation
- Training data quality analysis

**Interface:**
```python
class EnhancedSFTTrainer:
    def prepare_training_data() -> Dataset
    def train_model() -> TrainingResults
    def save_model(path: str) -> bool
    def evaluate_model() -> Dict[str, float]
    def plot_training_curves() -> None
```

### 3. Comprehensive Metrics Calculator (`models/evaluation/metrics_calculator.py`)

**Purpose:** Calculate and compare performance metrics across all techniques

**Metrics Included:**
- Execution correctness
- BLEU scores
- Schema compliance
- Query complexity handling
- Response time analysis
- Confidence calibration

**Interface:**
```python
class MetricsCalculator:
    def calculate_execution_correctness(predictions: List[str], ground_truth: List[str]) -> float
    def calculate_bleu_scores(predictions: List[str], references: List[str]) -> float
    def evaluate_schema_compliance(sql_queries: List[str], schema: Dict) -> float
    def measure_response_times(technique: str, queries: List[str]) -> List[float]
    def generate_comprehensive_report() -> Dict[str, Any]
```

### 4. RAG Pipeline Enhancement (`models/techniques/rag_pipeline.py`)

**Purpose:** Schema-aware retrieval with performance evaluation

**Key Features:**
- Hybrid BM25 + dense retrieval
- Schema card generation and indexing
- Retrieval accuracy measurement
- Context relevance scoring

**Interface:**
```python
class EnhancedRAGPipeline:
    def build_schema_index() -> None
    def retrieve_context(query: str) -> RetrievalResult
    def evaluate_retrieval_quality() -> Dict[str, float]
    def measure_context_relevance() -> float
```

## Data Models

### Training Data Structure
```python
@dataclass
class TrainingExample:
    question: str
    sql: str
    schema_context: Dict[str, Any]
    difficulty_level: str  # 'easy', 'medium', 'hard'
    query_type: str       # 'select', 'aggregate', 'join', 'nested'
    domain_category: str  # 'product_analysis', 'delivery_metrics', 'payment_trends', 'review_analysis'

@dataclass
class OlistDatasetInfo:
    """Olist Brazilian E-Commerce dataset information"""
    item_level_rows: int = 112650
    item_level_columns: int = 37
    order_level_rows: int = 98666
    order_level_columns: int = 13
    date_range: str = "2016-2018"
    total_orders: int = 100000
```

### Performance Result Structure
```python
@dataclass
class PerformanceResult:
    technique_name: str
    execution_correctness: float
    bleu_score: float
    schema_compliance: float
    avg_response_time: float
    confidence_score: float
    sample_outputs: List[str]
```

### Training Metrics Structure
```python
@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    eval_loss: float
    learning_rate: float
    gradient_norm: float
    timestamp: str
```

## Error Handling

### Model Loading Errors
- Graceful fallback to smaller models if GPU memory insufficient
- Clear error messages for missing dependencies
- Automatic retry with different configurations

### Training Errors
- Checkpoint recovery for interrupted training
- Memory optimization for large models
- Validation of training data quality

### Evaluation Errors
- Handling of malformed SQL outputs
- Timeout management for slow queries
- Graceful degradation for missing ground truth

## Testing Strategy

### Unit Tests
- Individual component functionality
- Metrics calculation accuracy
- Data processing correctness

### Integration Tests
- End-to-end notebook execution
- Model training and saving workflow
- Cross-technique comparison accuracy

### Performance Tests
- Memory usage monitoring
- Response time benchmarking
- Scalability testing with large datasets

### Validation Tests
- Model output quality verification
- Training convergence validation
- Retrieval accuracy assessment