# Requirements Document

## Introduction

This feature focuses on creating a comprehensive ML model performance demonstration system for the NL-to-SQL project using the Olist Brazilian E-Commerce Public Dataset. The system will showcase multiple ML techniques including prompting strategies, RAG (Retrieval-Augmented Generation), and fine-tuning with complete performance evaluation and training results in a unified notebook environment. The demonstration will use realistic e-commerce queries based on the Olist dataset, which contains ~100,000 customer orders from 2016-2018 with both item-level (112,650 rows, 37 columns) and order-level (98,666 rows, 13 columns) views.

## Glossary

- **NL2SQL_System**: The natural language to SQL conversion system
- **Performance_Notebook**: Jupyter notebook demonstrating model performance and training results
- **RAG_Pipeline**: Retrieval-Augmented Generation pipeline for schema-aware context retrieval
- **SFT_Module**: Supervised Fine-Tuning module for model training
- **Prompting_Engine**: System implementing multiple prompting strategies (CoT, Few-Shot, Self-Consistency, Least-to-Most)
- **Model_Evaluator**: Component for measuring and comparing model performance metrics
- **Training_Manager**: System for managing model training and saving trained models

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to see comprehensive performance results from multiple ML techniques, so that I can understand which approach works best for NL-to-SQL conversion.

#### Acceptance Criteria

1. WHEN the Performance_Notebook is executed, THE NL2SQL_System SHALL demonstrate results from at least 4 different prompting techniques
2. WHEN the Performance_Notebook runs RAG evaluation, THE RAG_Pipeline SHALL show retrieval accuracy and context relevance metrics
3. WHEN fine-tuning is performed, THE SFT_Module SHALL save the trained model and display training metrics
4. WHEN model comparison is requested, THE Model_Evaluator SHALL provide execution correctness, BLEU scores, and schema compliance metrics
5. WHERE multiple techniques are evaluated, THE Performance_Notebook SHALL display comparative performance visualizations

### Requirement 2

**User Story:** As a developer, I want a clean and organized project structure, so that I can easily navigate and maintain the codebase.

#### Acceptance Criteria

1. THE NL2SQL_System SHALL organize all model-related code in the models directory
2. THE NL2SQL_System SHALL remove unnecessary files and folders to maintain clean structure
3. WHEN environment setup is needed, THE NL2SQL_System SHALL provide proper .env configuration for API keys
4. THE NL2SQL_System SHALL consolidate duplicate functionality across modules
5. WHERE configuration is needed, THE NL2SQL_System SHALL use centralized config files

### Requirement 3

**User Story:** As a data scientist, I want to see unique insights and research findings, so that I can understand the effectiveness of different ML approaches.

#### Acceptance Criteria

1. WHEN the Performance_Notebook analyzes results, THE Model_Evaluator SHALL generate unique insights about technique effectiveness
2. THE Performance_Notebook SHALL include code cells showing research-specific analysis
3. WHEN comparing techniques, THE Model_Evaluator SHALL identify which approach works best for different query types
4. THE Performance_Notebook SHALL demonstrate at least one complete end-to-end technique with perfect implementation
5. WHERE training data is used, THE Training_Manager SHALL show Olist e-commerce data quality and distribution analysis

### Requirement 4

**User Story:** As a practitioner, I want to see complete model training and evaluation workflows, so that I can reproduce and extend the results.

#### Acceptance Criteria

1. WHEN fine-tuning is performed, THE SFT_Module SHALL complete the full training pipeline and save the model
2. THE Training_Manager SHALL display training loss curves and validation metrics
3. WHEN RAG is evaluated, THE RAG_Pipeline SHALL show retrieval performance and context quality
4. THE Performance_Notebook SHALL include model inference examples with confidence scores
5. WHERE evaluation metrics are calculated, THE Model_Evaluator SHALL provide comprehensive performance statistics