# ADR-001: Prompting Strategies for NL-to-SQL Generation

**Status:** Accepted  
**Date:** 2025-01-01  
**Owner:** Kushal Adhyaru  

## Context

We need to implement advanced prompting strategies to improve the accuracy and reliability of natural language to SQL conversion. Different types of questions may benefit from different reasoning approaches.

## Decision

We will implement four distinct prompting strategies:

### 1. Chain-of-Thought (CoT)
- **Use Case:** Complex multi-table queries requiring step-by-step reasoning
- **Approach:** Guide the model through explicit reasoning steps
- **Template:** Step-by-step breakdown of query construction

### 2. Few-Shot Learning
- **Use Case:** Domain-specific terminology and common query patterns
- **Approach:** Provide relevant examples before the target question
- **Template:** 3-5 curated examples with question-SQL pairs

### 3. Self-Consistency
- **Use Case:** Reducing hallucinations and improving reliability
- **Approach:** Generate multiple reasoning paths and select the most consistent
- **Template:** Multiple independent generations with voting mechanism

### 4. Least-to-Most Decomposition
- **Use Case:** Complex nested queries and aggregations
- **Approach:** Break down complex problems into simpler sub-problems
- **Template:** Hierarchical problem decomposition

## Implementation Details

### Strategy Selection
- Default to Chain-of-Thought for general queries
- Use Few-Shot for domain-specific questions
- Apply Self-Consistency for high-stakes queries
- Use Least-to-Most for complex analytical questions

### Template Management
- YAML-based templates for easy modification
- Parameterized prompts with schema context injection
- Version control for prompt evolution

### Evaluation Criteria
- Execution correctness on test dataset
- Query complexity handling
- Response time and computational cost
- Consistency across similar questions

## Consequences

### Positive
- Improved accuracy through specialized reasoning approaches
- Better handling of complex queries
- Reduced hallucinations through self-consistency
- Modular design allows easy strategy comparison

### Negative
- Increased complexity in prompt management
- Higher computational cost for self-consistency
- Need for strategy selection logic
- More templates to maintain

## Alternatives Considered

1. **Single Universal Prompt:** Simpler but less effective for diverse query types
2. **Dynamic Prompt Generation:** More flexible but harder to control and debug
3. **Model-Specific Prompts:** Better optimization but less portable

## Implementation Plan

1. **Week 1:** Implement base strategy interface and CoT
2. **Week 2:** Add Few-Shot and Self-Consistency strategies  
3. **Week 3:** Implement Least-to-Most and evaluation framework

## Success Metrics

- 15% improvement in execution correctness over baseline
- Successful handling of complex multi-table queries
- Consistent performance across different question types
- Sub-5 second response time for all strategies