"""
Enhanced prompting strategies implementation for NL-to-SQL generation.
Owner: Kushal Adhyaru

Implements five advanced prompting strategies:
1. Zero-Shot Prompting (Baseline)
2. Chain-of-Thought (CoT)
3. Few-Shot Learning
4. Self-Consistency
5. Least-to-Most Decomposition
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


class ZeroShotStrategy(PromptStrategy):
    """Zero-shot baseline prompting - direct question to SQL without examples."""
    
    def __init__(self, template_path: str = "prompts/templates/zero_shot.yaml"):
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        """Load YAML template file."""
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Default zero-shot template - serves as baseline."""
        return {
            "system_prompt": "You are an expert SQL analyst. Convert the natural language question into a valid SQL query using the provided database schema.",
            "instruction": "Generate a SQL query that accurately answers the question. Use only the tables and columns available in the schema.",
            "output_format": {
                "sql": "Complete SQL query",
                "confidence": "Confidence score (0-1)"
            }
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate zero-shot prompt - direct and minimal."""
        schema_info = self._format_schema_context(schema_context)
        
        return f"""
{self.template['system_prompt']}

Database Schema:
{schema_info}

Question: {question}

{self.template['instruction']}

Provide your response in JSON format:
{{
    "sql": "SELECT ...",
    "confidence": 0.85
}}
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse zero-shot response."""
        try:
            # Try to parse JSON
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if 'sql' not in result:
                    result['sql'] = self._extract_sql_fallback(response)
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback
        return {
            "sql": self._extract_sql_fallback(response),
            "confidence": 0.5,
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Zero-shot baseline generation"
        }
    
    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Format schema context concisely."""
        if not schema_context or 'tables' not in schema_context:
            return "No schema information available"
        
        schema_lines = []
        for table_name, table_info in schema_context['tables'].items():
            columns = table_info.get('columns', {})
            column_list = [f"{col}({info.get('type', 'unknown')})" for col, info in columns.items()]
            schema_lines.append(f"- {table_name}: {', '.join(column_list)}")
        
        return "\n".join(schema_lines)
    
    def _extract_sql_fallback(self, response: str) -> str:
        """Extract SQL from unstructured response."""
        import re
        sql_pattern = r'(SELECT\s+.*?(?:;|$))'
        matches = re.findall(sql_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            return matches[0].strip().rstrip(';')
        
        return "SELECT 1"


class ChainOfThoughtStrategy(PromptStrategy):
    """Enhanced Chain-of-Thought prompting for step-by-step reasoning."""
    
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
        """Enhanced CoT template with better reasoning steps and e-commerce focus."""
        return {
            "system_prompt": "You are an expert SQL analyst with deep knowledge of e-commerce databases. Think through each problem step by step, showing your reasoning process clearly.",
            "reasoning_steps": [
                "Step 1: Analyze the natural language question to understand what business information is being requested",
                "Step 2: Identify which tables and columns from the e-commerce schema are needed to answer the question", 
                "Step 3: Determine the appropriate SQL operations (SELECT, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, etc.)",
                "Step 4: Plan the query structure, including any necessary joins between tables and business logic",
                "Step 5: Consider data quality issues and edge cases (NULL values, date ranges, etc.)",
                "Step 6: Write the complete SQL query with proper syntax and optimization",
                "Step 7: Verify the query logic matches the original business question"
            ],
            "output_format": {
                "reasoning": "Detailed step-by-step explanation following the 7-step process",
                "sql": "Final optimized SQL query",
                "confidence": "Confidence score (0-1)",
                "business_context": "Brief explanation of business relevance"
            },
            "examples": [
                {
                    "question": "What are the top 3 cities by total order value in 2017?",
                    "reasoning": "Step 1: Need to find cities with highest total order values for 2017\nStep 2: Need orders table (for order values and dates) and customers table (for city)\nStep 3: Need JOIN, WHERE for date filter, GROUP BY city, SUM order values, ORDER BY DESC, LIMIT 3\nStep 4: JOIN orders with customers on customer_id, filter by year 2017\nStep 5: Handle NULL dates and ensure proper date filtering\nStep 6: Write complete query with proper joins and aggregation\nStep 7: Verify this answers 'top 3 cities by total order value in 2017'",
                    "sql": "SELECT c.customer_city, SUM(p.payment_value) as total_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_payments p ON o.order_id = p.order_id WHERE YEAR(o.order_purchase_timestamp) = 2017 GROUP BY c.customer_city ORDER BY total_value DESC LIMIT 3"
                }
            ]
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate enhanced CoT prompt with detailed reasoning framework and examples."""
        schema_info = self._format_schema_context(schema_context)
        
        # Include example if available
        example_text = ""
        if 'examples' in self.template and self.template['examples']:
            example = self.template['examples'][0]  # Use first example
            example_text = f"""
Example of Chain-of-Thought reasoning:

Question: {example['question']}
Reasoning: {example['reasoning']}
SQL: {example['sql']}

Now apply the same systematic approach to your question:
"""
        
        return f"""
{self.template['system_prompt']}

Question: {question}

Available E-commerce Schema:
{schema_info}

{example_text}

Please analyze this step by step following this framework:
{chr(10).join(self.template['reasoning_steps'])}

Provide your complete reasoning process and then the final SQL query in JSON format:
{{
    "reasoning": "Your detailed step-by-step analysis following the 7-step framework",
    "sql": "SELECT ...",
    "confidence": 0.95,
    "used_tables": ["table1", "table2"],
    "used_columns": ["col1", "col2"],
    "business_context": "Brief explanation of the business value of this query"
}}
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extract and validate JSON from CoT response."""
        try:
            # Find JSON object in response
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure required fields exist
                if 'sql' not in result:
                    result['sql'] = ""
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                if 'reasoning' not in result:
                    result['reasoning'] = "No reasoning provided"
                
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback parsing
        return {
            "sql": self._extract_sql_fallback(response),
            "confidence": 0.3,
            "reasoning": "Failed to parse structured response",
            "used_tables": [],
            "used_columns": []
        }
    
    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Format schema context for better readability."""
        if not schema_context or 'tables' not in schema_context:
            return "No schema information available"
        
        schema_lines = []
        for table_name, table_info in schema_context['tables'].items():
            columns = table_info.get('columns', {})
            column_list = [f"{col}({info.get('type', 'unknown')})" for col, info in columns.items()]
            schema_lines.append(f"- {table_name}: {', '.join(column_list)}")
        
        return "\n".join(schema_lines)
    
    def _extract_sql_fallback(self, response: str) -> str:
        """Fallback SQL extraction from unstructured response."""
        # Look for SQL keywords
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY']
        lines = response.split('\n')
        
        for line in lines:
            line_upper = line.strip().upper()
            if any(keyword in line_upper for keyword in sql_keywords):
                return line.strip()
        
        return ""


class FewShotStrategy(PromptStrategy):
    """Enhanced few-shot learning with domain-specific e-commerce examples."""
    
    def __init__(self, template_path: str = "prompts/templates/few_shot.yaml"):
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Enhanced few-shot template with diverse e-commerce examples."""
        return {
            "system_prompt": "You are an expert SQL analyst specializing in e-commerce data analysis. Learn from the provided examples and generate accurate SQL queries for similar questions.",
            "instruction": "Study these domain-specific examples carefully. Notice the patterns in how different types of business questions are translated to SQL queries.",
            "examples": [
                {
                    "category": "customer_analysis",
                    "question": "Which city has the highest number of customers?",
                    "sql": "SELECT customer_city, COUNT(*) as customer_count FROM customers GROUP BY customer_city ORDER BY customer_count DESC LIMIT 1",
                    "explanation": "Groups customers by city, counts them, and returns the city with the highest count",
                    "complexity": "simple"
                },
                {
                    "category": "product_analysis",
                    "question": "What is the total revenue for each product category?",
                    "sql": "SELECT p.product_category_name, SUM(oi.price + oi.freight_value) as total_revenue FROM products p JOIN order_items oi ON p.product_id = oi.product_id GROUP BY p.product_category_name ORDER BY total_revenue DESC",
                    "explanation": "Joins products with order items, calculates revenue including freight, groups by category",
                    "complexity": "medium"
                },
                {
                    "category": "delivery_analysis", 
                    "question": "What is the average delivery time by state?",
                    "sql": "SELECT c.customer_state, AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) as avg_delivery_days FROM orders o JOIN customers c ON o.customer_id = c.customer_id WHERE o.order_delivered_customer_date IS NOT NULL GROUP BY c.customer_state ORDER BY avg_delivery_days",
                    "explanation": "Calculates date difference for delivery time, groups by state, excludes undelivered orders",
                    "complexity": "medium"
                },
                {
                    "category": "payment_analysis",
                    "question": "Which payment method is most popular for high-value orders (>200)?",
                    "sql": "SELECT payment_type, COUNT(*) as usage_count, AVG(payment_value) as avg_payment_value FROM order_payments WHERE payment_value > 200 GROUP BY payment_type ORDER BY usage_count DESC",
                    "explanation": "Filters high-value payments, groups by payment type, shows usage statistics",
                    "complexity": "medium"
                }
            ]
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate enhanced few-shot prompt with categorized domain-specific examples."""
        schema_info = self._format_schema_context(schema_context)
        
        # Select most relevant examples based on question type
        relevant_examples = self._select_relevant_examples(question)
        
        examples_text = ""
        for i, example in enumerate(relevant_examples, 1):
            examples_text += f"""
Example {i} ({example['category']} - {example['complexity']}):
Question: {example['question']}
SQL: {example['sql']}
Explanation: {example['explanation']}
"""
        
        return f"""
{self.template['system_prompt']}

{self.template['instruction']}

E-commerce Schema Information:
{schema_info}

{examples_text}

Now generate SQL for this new question:
Question: {question}

Analyze the question type and select the most relevant example pattern. Provide your response in JSON format:
{{
    "sql": "SELECT ...",
    "explanation": "Brief explanation of the query logic and business context",
    "confidence": 0.85,
    "category": "customer_analysis|product_analysis|temporal_analysis|delivery_analysis|payment_analysis",
    "complexity": "simple|medium|complex"
}}
"""
    
    def _select_relevant_examples(self, question: str) -> List[Dict[str, Any]]:
        """Select the most relevant examples based on question content."""
        question_lower = question.lower()
        all_examples = self.template.get('examples', [])
        
        # Categorize question
        if any(word in question_lower for word in ['customer', 'city', 'state', 'user']):
            category_priority = ['customer_analysis', 'delivery_analysis']
        elif any(word in question_lower for word in ['product', 'category', 'item', 'revenue', 'sales']):
            category_priority = ['product_analysis', 'payment_analysis']
        elif any(word in question_lower for word in ['delivery', 'shipping', 'time', 'days']):
            category_priority = ['delivery_analysis', 'temporal_analysis']
        elif any(word in question_lower for word in ['payment', 'pay', 'method', 'value']):
            category_priority = ['payment_analysis', 'product_analysis']
        else:
            category_priority = ['customer_analysis', 'product_analysis']
        
        # Select examples prioritizing relevant categories
        selected = []
        for category in category_priority:
            for example in all_examples:
                if example.get('category') == category and example not in selected:
                    selected.append(example)
                    if len(selected) >= 3:  # Limit to 3 most relevant examples
                        break
            if len(selected) >= 3:
                break
        
        # Fill remaining slots with other examples if needed
        while len(selected) < 3 and len(selected) < len(all_examples):
            for example in all_examples:
                if example not in selected:
                    selected.append(example)
                    break
        
        return selected[:3]  # Return max 3 examples
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse few-shot response with fallback handling."""
        try:
            # Try to parse JSON first
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Ensure required fields
                if 'sql' not in result:
                    result['sql'] = self._extract_sql_fallback(response)
                if 'confidence' not in result:
                    result['confidence'] = 0.8
                
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback parsing
        return {
            "sql": self._extract_sql_fallback(response),
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Generated using few-shot learning",
            "confidence": 0.7
        }
    
    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Format schema context for few-shot examples."""
        if not schema_context or 'tables' not in schema_context:
            return "Schema information not available"
        
        schema_lines = ["Available Tables:"]
        for table_name, table_info in schema_context['tables'].items():
            columns = table_info.get('columns', {})
            column_details = []
            for col, info in columns.items():
                col_type = info.get('type', 'unknown')
                is_pk = info.get('primary_key', False)
                pk_marker = " (PK)" if is_pk else ""
                column_details.append(f"{col}:{col_type}{pk_marker}")
            
            schema_lines.append(f"  {table_name}({', '.join(column_details)})")
        
        return "\n".join(schema_lines)
    
    def _extract_sql_fallback(self, response: str) -> str:
        """Extract SQL from unstructured response."""
        # Look for SQL patterns
        import re
        sql_pattern = r'(SELECT\s+.*?(?:;|$))'
        matches = re.findall(sql_pattern, response, re.IGNORECASE | re.DOTALL)
        
        if matches:
            return matches[0].strip().rstrip(';')
        
        return "SELECT 1"  # Minimal fallback


class SelfConsistencyStrategy(PromptStrategy):
    """Enhanced self-consistency with voting mechanism."""
    
    def __init__(self, num_samples: int = 3, template_path: str = "prompts/templates/self_consistency.yaml"):
        self.num_samples = num_samples
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        """Load YAML template file."""
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Default template for self-consistency."""
        return {
            "system_prompt": "You are an expert SQL analyst. Generate multiple different approaches to solve the same SQL problem, then use voting and confidence scoring to select the best solution.",
            "voting_criteria": [
                "correctness: Does the SQL syntax and logic appear correct?",
                "completeness: Does it fully answer the original question?", 
                "efficiency: Is the query reasonably optimized?",
                "readability: Is the SQL clear and maintainable?",
                "business_logic: Does it make sense from a business perspective?"
            ]
        }
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate enhanced prompt for self-consistency sampling with voting mechanism."""
        schema_info = self._format_schema_context(schema_context)
        
        voting_criteria = "\n".join([f"- {criterion}" for criterion in self.template.get('voting_criteria', [])])
        
        return f"""
{self.template.get('system_prompt', 'Generate multiple approaches to solve this SQL problem.')}

Question: {question}

E-commerce Schema: 
{schema_info}

Generate exactly {self.num_samples} different approaches to solve this SQL problem:

Approach 1: Direct/Straightforward - Use the most obvious SQL pattern
Approach 2: Optimized/Efficient - Focus on performance and query optimization  
Approach 3: Comprehensive/Detailed - Include additional context and validation

For each approach, provide:
1. Your reasoning process and approach type
2. The complete SQL query
3. Confidence score (0-1) based on the voting criteria
4. Pros and cons of this approach

Voting Criteria for Confidence Scoring:
{voting_criteria}

Then analyze all approaches using the voting criteria and select the best one.

Format your response as:
{{
    "approaches": [
        {{
            "approach_type": "direct|optimized|comprehensive",
            "reasoning": "Detailed explanation of this approach",
            "sql": "SELECT ...",
            "confidence": 0.85,
            "pros": ["advantage1", "advantage2"],
            "cons": ["limitation1", "limitation2"]
        }}
    ],
    "voting_analysis": {{
        "criteria_scores": {{
            "correctness": [0.9, 0.8, 0.95],
            "completeness": [0.8, 0.9, 0.95], 
            "efficiency": [0.7, 0.95, 0.8],
            "readability": [0.9, 0.8, 0.85],
            "business_logic": [0.85, 0.9, 0.9]
        }},
        "weighted_scores": [0.83, 0.87, 0.89]
    }},
    "selected_approach": 2,
    "final_sql": "SELECT ...",
    "final_confidence": 0.89,
    "selection_reason": "Explanation of why this approach was selected"
}}
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse self-consistency response with voting logic."""
        try:
            # Try to parse structured response
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if 'final_sql' in result:
                    return {
                        "sql": result['final_sql'],
                        "confidence": result.get('final_confidence', 0.9),
                        "used_tables": [],
                        "used_columns": [],
                        "reason_short": "Selected via self-consistency voting",
                        "approaches": result.get('approaches', [])
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback - implement simple voting on SQL patterns
        return {
            "sql": self._extract_sql_fallback(response),
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Selected via self-consistency (fallback)",
            "confidence": 0.85
        }
    
    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Format schema for self-consistency prompting."""
        if not schema_context or 'tables' not in schema_context:
            return "No schema available"
        
        return str(schema_context.get('schema_graph_brief', 'Schema information available'))
    
    def _extract_sql_fallback(self, response: str) -> str:
        """Extract most common SQL pattern from response."""
        import re
        sql_patterns = re.findall(r'SELECT\s+.*?(?:;|\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        if sql_patterns:
            # Return the first valid-looking SQL
            return sql_patterns[0].strip().rstrip(';')
        
        return "SELECT 1"


class LeastToMostStrategy(PromptStrategy):
    """Enhanced least-to-most decomposition for complex queries."""
    
    def __init__(self, template_path: str = "prompts/templates/least_to_most.yaml"):
        self.template = self._load_template(template_path)
    
    def _load_template(self, path: str) -> Dict[str, Any]:
        """Load YAML template file."""
        template_file = Path(path)
        if template_file.exists():
            with open(template_file, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_template()
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Default template for least-to-most decomposition."""
        return {
            "system_prompt": "You are an expert SQL analyst specializing in breaking down complex database questions into simpler, manageable sub-problems.",
            "decomposition_framework": [
                "1. Question Analysis: Break down the main question into component parts",
                "2. Schema Mapping: Identify required tables, columns, and relationships", 
                "3. Operation Planning: Determine SQL operations needed (JOINs, aggregations, filters)",
                "4. Sub-query Design: Plan any necessary sub-queries or CTEs",
                "5. Integration: Combine sub-solutions into the final query",
                "6. Validation: Verify the final query addresses all question components"
            ]
        }
    
    def _analyze_question_complexity(self, question: str) -> str:
        """Analyze question to determine complexity type."""
        question_lower = question.lower()
        
        # Check for complex aggregation patterns
        if any(word in question_lower for word in ['average', 'more than', 'less than', 'compare', 'ratio']):
            return "complex_aggregation"
        
        # Check for multi-table analysis
        if any(word in question_lower for word in ['join', 'across', 'between', 'relationship', 'combined']):
            return "multi_table_analysis"
        
        # Check for temporal comparison
        if any(word in question_lower for word in ['monthly', 'yearly', 'trend', 'over time', 'period', 'growth']):
            return "temporal_comparison"
        
        # Check for conditional logic
        if any(word in question_lower for word in ['if', 'when', 'where', 'classify', 'categorize', 'based on']):
            return "conditional_logic"
        
        return "multi_table_analysis"  # Default
    
    def generate_prompt(self, question: str, schema_context: Dict[str, Any], **kwargs) -> str:
        """Generate enhanced decomposition prompt with structured framework."""
        schema_info = self._format_schema_context(schema_context)
        
        framework_steps = "\n".join(self.template.get('decomposition_framework', []))
        
        # Determine complexity type
        complexity_type = self._analyze_question_complexity(question)
        
        return f"""
{self.template.get('system_prompt', 'Break down complex SQL questions systematically.')}

Question: {question}

E-commerce Schema: 
{schema_info}

This appears to be a {complexity_type} type question. Follow this systematic decomposition framework:
{framework_steps}

Break down the question into specific sub-problems and solve each step:

Provide your response as:
{{
    "question_analysis": {{
        "main_question": "Restated version of the original question",
        "complexity_type": "{complexity_type}",
        "key_components": ["component1", "component2", "component3"]
    }},
    "decomposition": [
        {{
            "step": 1,
            "sub_question": "What tables and columns are needed?",
            "answer": "Detailed answer with specific table/column names",
            "sql_fragment": "Relevant SQL snippet if applicable"
        }},
        {{
            "step": 2, 
            "sub_question": "What joins are required?",
            "answer": "Explanation of join logic and conditions",
            "sql_fragment": "JOIN clauses"
        }},
        {{
            "step": 3,
            "sub_question": "What filters or conditions apply?",
            "answer": "Business logic and filter conditions",
            "sql_fragment": "WHERE clause components"
        }},
        {{
            "step": 4,
            "sub_question": "What aggregations or calculations are needed?",
            "answer": "Aggregation logic and grouping strategy", 
            "sql_fragment": "GROUP BY and aggregate functions"
        }},
        {{
            "step": 5,
            "sub_question": "How should results be ordered and limited?",
            "answer": "Sorting and result limiting logic",
            "sql_fragment": "ORDER BY and LIMIT clauses"
        }}
    ],
    "integration_strategy": "Explanation of how sub-solutions combine",
    "final_sql": "Complete SQL query combining all sub-solutions",
    "validation_checks": [
        "Does the query answer the original question completely?",
        "Are all business rules properly implemented?", 
        "Is the query syntactically correct?",
        "Are there any potential performance issues?"
    ],
    "confidence": 0.88
}}
"""
    
    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parse decomposition response."""
        try:
            # Try structured parsing
            start_idx = response.rfind("{")
            end_idx = response.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                if 'final_sql' in result:
                    return {
                        "sql": result['final_sql'],
                        "confidence": result.get('confidence', 0.85),
                        "used_tables": [],
                        "used_columns": [],
                        "reason_short": "Generated via least-to-most decomposition",
                        "decomposition": result.get('decomposition', [])
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback
        return {
            "sql": self._extract_sql_fallback(response),
            "used_tables": [],
            "used_columns": [],
            "reason_short": "Generated via decomposition (fallback)",
            "confidence": 0.75
        }
    
    def _format_schema_context(self, schema_context: Dict[str, Any]) -> str:
        """Format schema for decomposition."""
        return str(schema_context.get('schema_graph_brief', 'Schema available'))
    
    def _extract_sql_fallback(self, response: str) -> str:
        """Extract SQL from decomposition response."""
        import re
        sql_match = re.search(r'SELECT\s+.*?(?:;|\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        if sql_match:
            return sql_match.group(0).strip().rstrip(';')
        
        return "SELECT 1"


class PromptingStrategyEvaluator:
    """Evaluation framework for prompting strategies with performance comparison utilities."""
    
    def __init__(self):
        self.evaluation_metrics = {
            'prompt_quality': 0.3,      # How well-structured is the prompt
            'response_parsability': 0.2, # How easy to parse the response
            'business_relevance': 0.25,  # How relevant to e-commerce domain
            'complexity_handling': 0.25  # How well it handles question complexity
        }
    
    def evaluate_strategy_performance(self, strategy: PromptStrategy, question: str, 
                                    schema_context: Dict[str, Any], 
                                    expected_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate a single strategy's performance on a question."""
        try:
            # Generate prompt
            prompt = strategy.generate_prompt(question, schema_context)
            
            # Evaluate prompt quality
            prompt_score = self._evaluate_prompt_quality(prompt, question)
            
            # Evaluate business relevance
            business_score = self._evaluate_business_relevance(prompt, question)
            
            # Evaluate complexity handling
            complexity_score = self._evaluate_complexity_handling(prompt, question)
            
            # Calculate overall score
            overall_score = (
                prompt_score * self.evaluation_metrics['prompt_quality'] +
                business_score * self.evaluation_metrics['business_relevance'] +
                complexity_score * self.evaluation_metrics['complexity_handling'] +
                0.8 * self.evaluation_metrics['response_parsability']  # Default parsability score
            )
            
            return {
                'strategy_name': strategy.__class__.__name__,
                'overall_score': overall_score,
                'prompt_quality': prompt_score,
                'business_relevance': business_score,
                'complexity_handling': complexity_score,
                'prompt_length': len(prompt),
                'prompt_structure_score': self._evaluate_prompt_structure(prompt),
                'evaluation_timestamp': str(Path(__file__).stat().st_mtime)
            }
            
        except Exception as e:
            return {
                'strategy_name': strategy.__class__.__name__,
                'error': str(e),
                'overall_score': 0.0,
                'evaluation_timestamp': str(Path(__file__).stat().st_mtime)
            }
    
    def _evaluate_prompt_quality(self, prompt: str, question: str) -> float:
        """Evaluate the quality of the generated prompt."""
        score = 0.0
        
        # Check for clear instructions
        if 'JSON format' in prompt or 'json' in prompt.lower():
            score += 0.2
        
        # Check for examples or templates
        if 'example' in prompt.lower() or 'template' in prompt.lower():
            score += 0.2
        
        # Check for schema information inclusion
        if 'schema' in prompt.lower() and 'table' in prompt.lower():
            score += 0.2
        
        # Check for step-by-step guidance
        if any(word in prompt.lower() for word in ['step', 'process', 'framework']):
            score += 0.2
        
        # Check for business context
        if any(word in prompt.lower() for word in ['e-commerce', 'business', 'customer', 'order']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_business_relevance(self, prompt: str, question: str) -> float:
        """Evaluate how well the prompt addresses e-commerce business context."""
        score = 0.0
        
        # E-commerce domain terms
        ecommerce_terms = ['customer', 'order', 'product', 'payment', 'delivery', 'category', 'revenue']
        prompt_lower = prompt.lower()
        
        # Count relevant terms
        term_count = sum(1 for term in ecommerce_terms if term in prompt_lower)
        score += min(term_count * 0.15, 0.6)
        
        # Check for business logic considerations
        if any(word in prompt_lower for word in ['business', 'analysis', 'metrics', 'performance']):
            score += 0.2
        
        # Check for domain-specific examples
        if 'brazilian' in prompt_lower or 'olist' in prompt_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_complexity_handling(self, prompt: str, question: str) -> float:
        """Evaluate how well the prompt handles question complexity."""
        question_lower = question.lower()
        prompt_lower = prompt.lower()
        
        # Determine question complexity
        complexity_indicators = ['average', 'compare', 'multiple', 'join', 'across', 'trend', 'classify']
        complexity_level = sum(1 for indicator in complexity_indicators if indicator in question_lower)
        
        score = 0.5  # Base score
        
        # For complex questions, check if prompt provides appropriate guidance
        if complexity_level >= 2:
            if any(word in prompt_lower for word in ['decompose', 'break down', 'step by step']):
                score += 0.3
            if any(word in prompt_lower for word in ['sub-question', 'sub-problem', 'component']):
                score += 0.2
        
        # For simple questions, check if prompt is appropriately concise
        elif complexity_level <= 1:
            if len(prompt) < 1000:  # Reasonable length for simple questions
                score += 0.3
            if 'example' in prompt_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_prompt_structure(self, prompt: str) -> float:
        """Evaluate the structural quality of the prompt."""
        score = 0.0
        
        # Check for clear sections
        if prompt.count('\n\n') >= 2:  # Multiple sections
            score += 0.3
        
        # Check for formatting
        if '{' in prompt and '}' in prompt:  # JSON structure example
            score += 0.2
        
        # Check for clear question statement
        if 'Question:' in prompt:
            score += 0.2
        
        # Check for schema section
        if 'Schema' in prompt:
            score += 0.3
        
        return min(score, 1.0)
    
    def compare_strategies(self, strategies: Dict[str, PromptStrategy], 
                          questions: List[str], schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple strategies across multiple questions."""
        results = {}
        
        for strategy_name, strategy in strategies.items():
            strategy_results = []
            
            for question in questions:
                evaluation = self.evaluate_strategy_performance(strategy, question, schema_context)
                strategy_results.append(evaluation)
            
            # Calculate aggregate metrics
            avg_score = sum(r.get('overall_score', 0) for r in strategy_results) / len(strategy_results)
            avg_prompt_quality = sum(r.get('prompt_quality', 0) for r in strategy_results) / len(strategy_results)
            avg_business_relevance = sum(r.get('business_relevance', 0) for r in strategy_results) / len(strategy_results)
            avg_complexity_handling = sum(r.get('complexity_handling', 0) for r in strategy_results) / len(strategy_results)
            
            results[strategy_name] = {
                'individual_results': strategy_results,
                'aggregate_metrics': {
                    'average_overall_score': avg_score,
                    'average_prompt_quality': avg_prompt_quality,
                    'average_business_relevance': avg_business_relevance,
                    'average_complexity_handling': avg_complexity_handling,
                    'total_questions_evaluated': len(questions)
                }
            }
        
        # Rank strategies
        ranked_strategies = sorted(
            results.items(), 
            key=lambda x: x[1]['aggregate_metrics']['average_overall_score'], 
            reverse=True
        )
        
        return {
            'strategy_results': results,
            'ranking': [{'strategy': name, 'score': data['aggregate_metrics']['average_overall_score']} 
                       for name, data in ranked_strategies],
            'best_strategy': ranked_strategies[0][0] if ranked_strategies else None,
            'evaluation_summary': {
                'total_strategies': len(strategies),
                'total_questions': len(questions),
                'evaluation_completed': True
            }
        }


class PromptingEngine:
    """Enhanced prompting engine that manages and evaluates strategies."""
    
    def __init__(self):
        self.strategies = {
            "zero_shot": ZeroShotStrategy(),
            "chain_of_thought": ChainOfThoughtStrategy(),
            "few_shot": FewShotStrategy(),
            "self_consistency": SelfConsistencyStrategy(),
            "least_to_most": LeastToMostStrategy()
        }
        self.evaluator = PromptingStrategyEvaluator()
    
    def get_strategy(self, strategy_name: str) -> PromptStrategy:
        """Get a specific prompting strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return self.strategies[strategy_name]
    
    def evaluate_all_strategies(self, question: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all strategies on a single question."""
        results = {}
        for name, strategy in self.strategies.items():
            try:
                prompt = strategy.generate_prompt(question, schema_context)
                # In real implementation, would call model here
                # For now, just return the prompt for evaluation
                results[name] = {
                    "prompt": prompt,
                    "status": "ready",
                    "strategy_type": name
                }
            except Exception as e:
                results[name] = {
                    "error": str(e),
                    "status": "failed",
                    "strategy_type": name
                }
        return results
    
    def select_best_strategy(self, question: str, schema_context: Dict[str, Any]) -> str:
        """Select the most appropriate strategy for a given question using enhanced heuristics."""
        question_lower = question.lower()
        
        # Calculate complexity score
        complexity_indicators = [
            'average', 'more than', 'less than', 'compare', 'ratio', 'between',
            'join', 'across', 'relationship', 'combined', 'multiple tables',
            'monthly', 'yearly', 'trend', 'over time', 'period', 'growth',
            'classify', 'categorize', 'based on', 'if', 'when', 'where'
        ]
        
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in question_lower)
        
        # High complexity questions benefit from decomposition
        if complexity_score >= 3:
            return "least_to_most"
        
        # Questions with multiple conditions or logical operators
        if any(word in question_lower for word in ['and', 'or', 'both', 'either', 'multiple', 'various']):
            return "least_to_most"
        
        # Questions asking for detailed reasoning or explanation
        if any(word in question_lower for word in ['why', 'how', 'explain', 'reason', 'analyze', 'breakdown']):
            return "chain_of_thought"
        
        # Questions that might benefit from multiple approaches (reliability critical)
        if any(word in question_lower for word in ['best', 'optimal', 'most', 'top', 'highest', 'lowest']):
            return "self_consistency"
        
        # Simple, direct questions work well with few-shot learning
        if any(word in question_lower for word in ['show', 'list', 'find', 'get', 'what', 'which', 'who']):
            return "few_shot"
        
        # Default to chain-of-thought for systematic reasoning
        return "chain_of_thought"
    
    def evaluate_strategy_selection(self, question: str, schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all strategies and recommend the best one for the question."""
        # Get automatic selection
        auto_selected = self.select_best_strategy(question, schema_context)
        
        # Evaluate all strategies for this question
        evaluation_results = {}
        for name, strategy in self.strategies.items():
            evaluation = self.evaluator.evaluate_strategy_performance(strategy, question, schema_context)
            evaluation_results[name] = evaluation
        
        # Find best performing strategy
        best_strategy = max(evaluation_results.items(), key=lambda x: x[1].get('overall_score', 0))
        
        return {
            'question': question,
            'auto_selected_strategy': auto_selected,
            'best_evaluated_strategy': best_strategy[0],
            'strategy_evaluations': evaluation_results,
            'selection_agreement': auto_selected == best_strategy[0],
            'recommendation': {
                'strategy': best_strategy[0],
                'confidence': best_strategy[1].get('overall_score', 0),
                'reason': f"Scored {best_strategy[1].get('overall_score', 0):.3f} based on prompt quality, business relevance, and complexity handling"
            }
        }
    
    def benchmark_strategies(self, test_questions: List[str], schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive benchmark of all strategies across multiple questions."""
        return self.evaluator.compare_strategies(self.strategies, test_questions, schema_context)
    
    def get_strategy_recommendations(self, questions: List[str], schema_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get strategy recommendations for a set of questions."""
        recommendations = {}
        
        for question in questions:
            evaluation = self.evaluate_strategy_selection(question, schema_context)
            recommendations[question] = {
                'recommended_strategy': evaluation['recommendation']['strategy'],
                'confidence': evaluation['recommendation']['confidence'],
                'auto_vs_evaluated': evaluation['selection_agreement']
            }
        
        # Aggregate insights
        strategy_usage = {}
        for rec in recommendations.values():
            strategy = rec['recommended_strategy']
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        return {
            'individual_recommendations': recommendations,
            'strategy_usage_summary': strategy_usage,
            'most_recommended_strategy': max(strategy_usage.items(), key=lambda x: x[1])[0] if strategy_usage else None,
            'total_questions_analyzed': len(questions)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced prompting engine with evaluation capabilities
    engine = PromptingEngine()
    
    # Mock e-commerce schema
    mock_schema = {
        'tables': {
            'customers': {
                'columns': {
                    'customer_id': {'type': 'INTEGER', 'primary_key': True},
                    'customer_city': {'type': 'VARCHAR'},
                    'customer_state': {'type': 'VARCHAR'}
                }
            },
            'orders': {
                'columns': {
                    'order_id': {'type': 'INTEGER', 'primary_key': True},
                    'customer_id': {'type': 'INTEGER'},
                    'order_purchase_timestamp': {'type': 'DATETIME'}
                }
            },
            'order_payments': {
                'columns': {
                    'order_id': {'type': 'INTEGER'},
                    'payment_type': {'type': 'VARCHAR'},
                    'payment_value': {'type': 'DECIMAL'}
                }
            }
        }
    }
    
    # Test questions of varying complexity
    test_questions = [
        "Which city has the most customers?",
        "What is the average order value by payment method?",
        "Compare monthly revenue trends between 2017 and 2018",
        "Find customers who have made more than 5 orders and spent over $500"
    ]
    
    print("Testing Enhanced Prompting Strategies with Evaluation:")
    print("=" * 60)
    
    # Test strategy selection and evaluation
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)
        
        # Get strategy recommendation
        evaluation = engine.evaluate_strategy_selection(question, mock_schema)
        print(f"Auto-selected: {evaluation['auto_selected_strategy']}")
        print(f"Best evaluated: {evaluation['best_evaluated_strategy']}")
        print(f"Agreement: {evaluation['selection_agreement']}")
        print(f"Confidence: {evaluation['recommendation']['confidence']:.3f}")
    
    # Comprehensive benchmark
    print(f"\n{'='*60}")
    print("COMPREHENSIVE STRATEGY BENCHMARK")
    print("=" * 60)
    
    benchmark_results = engine.benchmark_strategies(test_questions, mock_schema)
    
    print("\nStrategy Rankings:")
    for i, ranking in enumerate(benchmark_results['ranking'], 1):
        print(f"{i}. {ranking['strategy']}: {ranking['score']:.3f}")
    
    print(f"\nBest Overall Strategy: {benchmark_results['best_strategy']}")
    
    # Strategy recommendations summary
    recommendations = engine.get_strategy_recommendations(test_questions, mock_schema)
    print(f"\nStrategy Usage Summary:")
    for strategy, count in recommendations['strategy_usage_summary'].items():
        print(f"  {strategy}: {count} questions")
    
    print(f"\nMost Recommended: {recommendations['most_recommended_strategy']}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE - Enhanced prompting strategies ready for use!")