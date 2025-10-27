# Models Directory

This directory contains the complete ML model performance demonstration system for NL-to-SQL conversion, organized into specialized modules for training, evaluation, techniques, and utilities.

## Directory Structure

```
models/
├── training/                    # Supervised fine-tuning module
│   ├── sft_trainer.py          # Enhanced SFT trainer implementation
│   ├── model_checkpoints/      # Saved trained models
│   └── training_data/          # Processed training datasets
├── evaluation/                  # Comprehensive evaluation module
│   ├── metrics_calculator.py   # Execution correctness, BLEU, schema compliance
│   ├── comparator.py          # Multi-technique comparison framework
│   └── visualizer.py          # Performance visualization and dashboards
├── techniques/                  # ML technique implementations
│   ├── prompting_strategies.py # CoT, Few-Shot, Self-Consistency, Least-to-Most
│   └── rag_pipeline.py        # Schema-aware RAG with hybrid retrieval
├── utils/                      # Utilities and configuration
│   ├── data_processor.py      # Data preparation and augmentation
│   ├── config_manager.py      # Centralized configuration management
│   ├── environment_setup.py   # Environment validation
│   └── validate_environment.py # Validation script
├── cache/                      # Model cache directory
└── README.md                   # This file
```

## Quick Start

### 1. Environment Setup

First, validate your environment:

```bash
python models/utils/validate_environment.py
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy and configure the environment file:

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, HuggingFace, Weights & Biases)
```

### 4. Run Performance Demo

The main performance demonstration is available in the Jupyter notebook:

```bash
jupyter notebook models/performance_demo.ipynb
```

## Core Components

### Training Module (`training/`)

**Enhanced SFT Trainer** (`sft_trainer.py`):
- Complete supervised fine-tuning pipeline using HuggingFace transformers
- Support for T5, CodeT5, and other seq2seq models
- Training progress tracking and model checkpointing
- Weights & Biases integration for experiment tracking

**Usage**:
```python
from models.training.sft_trainer import EnhancedSFTTrainer, SFTConfig

config = SFTConfig(model_name="t5-small", num_train_epochs=3)
trainer = EnhancedSFTTrainer(config)
trainer.train_model()
```

### Evaluation Module (`evaluation/`)

**Metrics Calculator** (`metrics_calculator.py`):
- Execution correctness evaluation
- BLEU score calculation for SQL similarity
- Schema compliance checking
- Response time measurement

**Model Comparator** (`comparator.py`):
- Multi-technique performance comparison
- Statistical analysis and recommendations
- Automated best technique selection

**Performance Visualizer** (`visualizer.py`):
- Training curve plotting
- Comparative performance charts
- Interactive dashboards with Plotly
- Confusion matrices and radar charts

### Techniques Module (`techniques/`)

**Prompting Strategies** (`prompting_strategies.py`):
- **Chain-of-Thought**: Step-by-step reasoning with enhanced templates
- **Few-Shot Learning**: Domain-specific examples with structured prompts
- **Self-Consistency**: Multiple reasoning paths with voting mechanism
- **Least-to-Most**: Complex query decomposition

**RAG Pipeline** (`rag_pipeline.py`):
- Schema-aware retrieval with hybrid BM25 + dense embeddings
- Schema card generation and indexing
- Retrieval quality evaluation and context relevance scoring

### Utilities Module (`utils/`)

**Data Processor** (`data_processor.py`):
- Data loading, validation, and quality analysis
- Data augmentation with paraphrasing and formatting variations
- Train/validation/test splitting

**Configuration Manager** (`config_manager.py`):
- Centralized configuration for all components
- Environment variable overrides
- Configuration validation and merging

**Environment Setup** (`environment_setup.py`):
- Comprehensive environment validation
- Package version checking
- GPU availability and memory analysis
- Model loading capability testing

## Configuration

The system uses a centralized configuration approach. Main configuration sections:

- **Model Config**: Model selection, tokenizer settings, cache directories
- **Training Config**: Learning rates, batch sizes, training parameters
- **Data Config**: Dataset paths, augmentation settings, split ratios
- **Evaluation Config**: Metrics selection, database paths, output directories
- **Technique Config**: Enabled techniques, strategy-specific parameters

Example configuration loading:

```python
from models.utils.config_manager import ConfigManager

config_manager = ConfigManager()
config = config_manager.load_config()
```

## Performance Metrics

The system evaluates techniques across multiple dimensions:

### Accuracy Metrics
- **Execution Correctness**: Whether generated SQL produces correct results
- **Exact Match**: Normalized SQL string matching
- **Schema Compliance**: Adherence to database schema constraints
- **BLEU Score**: Semantic similarity between generated and reference SQL

### Performance Metrics
- **Response Time**: Query processing latency
- **Confidence Calibration**: Correlation between confidence and accuracy
- **Success Rate**: Percentage of successful query generations

### Quality Metrics
- **Query Complexity Handling**: Performance across different query types
- **Error Analysis**: Categorization and frequency of error types
- **Robustness**: Performance consistency across diverse inputs

## Model Support

### Supported Base Models
- **T5 Family**: t5-small, t5-base, t5-large
- **Flan-T5**: google/flan-t5-small, google/flan-t5-base
- **CodeT5**: Salesforce/codet5-small, Salesforce/codet5-base
- **Custom Models**: Any T5-compatible seq2seq model

### Model Selection Guidelines

| Model | Parameters | Memory | Speed | Accuracy | Use Case |
|-------|------------|--------|-------|----------|----------|
| t5-small | 60M | ~1GB | Fast | Good | Development, prototyping |
| t5-base | 220M | ~2GB | Medium | Better | Production (balanced) |
| flan-t5-small | 80M | ~1GB | Fast | Good | Instruction following |
| flan-t5-base | 250M | ~2GB | Medium | Better | Production (high quality) |

## Environment Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space

### Recommended Setup
- Python 3.9+
- 8GB+ RAM
- GPU with 4GB+ VRAM
- 10GB+ disk space

### Optional Enhancements
- CUDA-compatible GPU for faster training
- Weights & Biases account for experiment tracking
- OpenAI API key for advanced model comparisons

## Troubleshooting

### Common Issues

**Environment Validation Failures**:
```bash
# Run validation script for detailed diagnostics
python models/utils/validate_environment.py
```

**Memory Issues**:
- Reduce batch size in training configuration
- Use smaller base models (t5-small instead of t5-base)
- Enable gradient checkpointing for training

**Model Loading Errors**:
- Check internet connection for model downloads
- Verify HuggingFace token for gated models
- Clear model cache if corrupted: `rm -rf models/cache/*`

**Training Failures**:
- Validate training data format and quality
- Check GPU memory availability
- Review training logs for specific error messages

### Performance Optimization

**For Training**:
- Use mixed precision training (fp16=True)
- Enable gradient accumulation for larger effective batch sizes
- Use learning rate scheduling and warmup

**For Inference**:
- Keep models loaded in memory between requests
- Use batch processing for multiple queries
- Consider model quantization for deployment

## Contributing

When adding new techniques or improvements:

1. Follow the established directory structure
2. Add comprehensive docstrings and type hints
3. Include unit tests for new functionality
4. Update configuration schemas as needed
5. Add examples and documentation

## License

This project is part of the Language Models for Structured Data Analysis research initiative.