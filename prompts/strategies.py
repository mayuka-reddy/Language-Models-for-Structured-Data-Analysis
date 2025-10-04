"""
Prompting strategies implementation for NL-to-SQL generation.
Owner: Kushal Adhyaru

Implements four advanced prompting strategies:
1. Chain-of-Thought (CoT)
2. Few-Shot Learning
3. Self-Consistency
4. Least-to-Most Decomposition
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import json
import yaml
from pathlib import Path


class PromptStrategy(ABC):
    """Base class for prompting strategies."""
    
    @abstractmethod
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate a prompt using this strategy."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response for this strategy."""
        pass


class ChainOfThoughtStrategy(PromptStrategy):
    """Chain-of-Thought prompting for step-by-step reasoning."""
    
    def __init__(self, template_path: str = "prompts/templates/chain_of_thought.yaml"):
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        """Load YAML template file."""
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Default CoT template if file doesn't exist."""
        return {
            "system_prompt": "You are an expert SQL analyst. Think step by step.",
            "reasoning_steps": [
                "1. Understand the question",
                "2. Identify required tables and columns", 
                "3. Plan the SQL structure",
                "4. Write the SQL query"
            ]
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate CoT prompt with step-by-step reasoning."""
        return f"""
{self.template['system_prompt']}

Question: {question}

Schema Context:
{schema_context.get('schema_graph_brief', '')}

Please think through this step by step:
{chr(10).join(self.template['reasoning_steps'])}

Provide your reasoning and then the final JSON response.
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from CoT response."""
        # Find JSON object in response
        start_idx = response.rfind("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        raise ValueError("No valid JSON found in CoT response")


class FewShotStrategy(PromptStrategy):
    """Few-shot learning with domain-specific examples."""
    
    def __init__(self, template_path: str = "prompts/templates/few_shot.yaml"):
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        return {
            "examples": [
                {
                    "question": "Which city has the highest number of customers?",
                    "sql": "SELECT city, COUNT(*) as customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1"
                }
            ]
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate few-shot prompt with examples."""
        examples_text = ""
        for i, example in enumerate(self.template['examples'], 1):
            examples_text += f"\nExample {i}:\nQ: {example['question']}\nA: {example['sql']}\n"
        
        return f"""
You are an expert SQL generator. Learn from these examples:
{examples_text}

Schema Context:
{schema_context.get('schema_graph_brief', '')}

Now generate SQL for:
Q: {question}
A: """
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse few-shot response."""
        # Simple parsing - assume response is just SQL
        return {
            "sql": response.strip(),
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Generated using few-shot learning",
            "confidence": 0.8
        }


class SelfConsistencyStrategy(PromptStrategy):
    """Self-consistency with multiple reasoning paths."""
    
    def __init__(self, num_samples: int = 3):
        self.num_samples = num_samples
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate prompt for self-consistency sampling."""
        return f"""
Generate {self.num_samples} different approaches to solve this SQL problem:

Question: {question}
Schema: {schema_context.get('schema_graph_brief', '')}

For each approach, provide reasoning and SQL. Then select the best one.
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse self-consistency response."""
        # Placeholder - would implement voting mechanism
        return {
            "sql": "SELECT 1",
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Selected via self-consistency",
            "confidence": 0.9
        }


class LeastToMostStrategy(PromptStrategy):
    """Least-to-most decomposition for complex queries."""
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate decomposition prompt."""
        return f"""
Break down this complex question into simpler sub-problems:

Question: {question}
Schema: {schema_context.get('schema_graph_brief', '')}

1. Identify the main components
2. Solve each component step by step
3. Combine into final SQL query
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse decomposition response."""
        # Placeholder - would implement decomposition logic
        return {
            "sql": "SELECT 1",
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Generated via decomposition",
            "confidence": 0.85
        }


class PromptStrategyManager:
    """Manages and coordinates different prompting strategies."""
    
    def __init__(self):
        self.strategies = {
            "chain_of_thought": ChainOfThoughtStrategy(),
            "few_shot": FewShotStrategy(),
            "self_consistency": SelfConsistencyStrategy(),
            "least_to_most": LeastToMostStrategy()
        }
    
    def get_strategy(self, strategy_name: str) -> PromptStrategy:
        """Get a specific prompting strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return self.strategies[strategy_name]
    
    def compare_strategies(self, question: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare all strategies on a single question."""
        results = {}
        for name, strategy in self.strategies.items():
            try:
                prompt = strategy.generate_prompt(question, schema_context)
                # In real implementation, would call model here
                results[name] = {"prompt": prompt, "status": "ready"}
            except Exception as e:
                results[name] = {"error": str(e), "status": "failed"}
        return results