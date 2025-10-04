"""
Gradio UI for NL-to-SQL Assistant
Interactive web interface for natural language to SQL conversion.
"""

import gradio as gr
import pandas as pd
import sys
import os
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.inference import NL2SQLInference
from app.sql_executor import SQLExecutor
from app.insights import InsightsGenerator
from app.charts import ChartGenerator
from app.metrics import ModelEvaluator


class NL2SQLApp:
    """
    Main application class for the NL-to-SQL Gradio interface.
    
    Coordinates between inference, execution, insights, and visualization
    components to provide a complete user experience.
    """
    
    def __init__(self):
        """Initialize the application with all components."""
        try:
            self.inference = NL2SQLInference()
            self.sql_executor = SQLExecutor()
            self.insights_generator = InsightsGenerator()
            self.chart_generator = ChartGenerator(style="plotly")
            self.evaluator = ModelEvaluator()
            
            # Get schema information for display
            self.schema_info = self.sql_executor.get_schema_info()
            
            print("✅ NL2SQL App initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing app: {e}")
            print("💡 Install missing dependencies with: pip install transformers torch duckdb loguru")
            raise
    
    def process_query(self, question: str, chart_type: str = "auto") -> Tuple[str, str, str, str, Any]:
        """
        Process a natural language question through the complete pipeline.
        
        Args:
            question: Natural language question
            chart_type: Type of chart to generate
            
        Returns:
            Tuple of (sql, results_html, insights_text, recommendations_text, chart)
        """
        if not question.strip():
            return "", "Please enter a question.", "", "", None
        
        try:
            # Step 1: Generate SQL
            schema_str = self._format_schema_for_model()
            inference_result = self.inference.generate_sql(question, schema_str)
            
            if 'error' in inference_result:
                return "", f"Error generating SQL: {inference_result['error']}", "", "", None
            
            generated_sql = inference_result['sql']
            confidence = inference_result.get('confidence', 0.0)
            
            # Step 2: Execute SQL
            execution_result = self.sql_executor.execute_query(generated_sql)
            
            if not execution_result['success']:
                return (
                    generated_sql,
                    f"SQL execution failed: {execution_result['error']}",
                    "",
                    "",
                    None
                )
            
            query_data = execution_result['data']
            row_count = execution_result['row_count']
            
            # Step 3: Generate insights
            insights = self.insights_generator.generate_insights(query_data, generated_sql)
            insights_text = self._format_insights(insights, confidence)
            
            # Step 4: Generate recommendations
            recommendations = self.insights_generator.generate_recommendations(query_data, generated_sql)
            recommendations_text = self._format_recommendations(recommendations)
            
            # Step 5: Create visualization
            chart_result = self.chart_generator.generate_chart(query_data, chart_type)
            chart = chart_result['chart'] if chart_result['success'] else None
            
            # Step 6: Format results table
            results_html = self._format_results_table(query_data, row_count)
            
            return generated_sql, results_html, insights_text, recommendations_text, chart
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            return "", error_msg, "", "", None
    
    def _format_schema_for_model(self) -> str:
        """Format schema information for the model."""
        schema_parts = []
        
        for table_name, table_info in self.schema_info['tables'].items():
            columns = [col['name'] for col in table_info['columns']]
            schema_parts.append(f"{table_name}({', '.join(columns)})")
        
        return "Tables: " + ", ".join(schema_parts)
    
    def _format_results_table(self, data: List[Dict[str, Any]], row_count: int) -> str:
        """Format query results as HTML table."""
        if not data:
            return f"<p>Query executed successfully but returned no results.</p>"
        
        # Convert to DataFrame for better formatting
        df = pd.DataFrame(data)
        
        # Limit display to first 100 rows
        display_df = df.head(100)
        
        html = f"<div><p><strong>Results ({row_count} rows):</strong></p>"
        
        if row_count > 100:
            html += f"<p><em>Showing first 100 rows of {row_count} total rows.</em></p>"
        
        # Convert DataFrame to HTML
        table_html = display_df.to_html(
            classes="table table-striped table-hover",
            table_id="results-table",
            escape=False,
            index=False
        )
        
        html += table_html + "</div>"
        return html
    
    def _format_insights(self, insights: Dict[str, Any], confidence: float) -> str:
        """Format insights as readable text."""
        text_parts = [
            f"**Query Analysis (Confidence: {confidence:.1%})**\n",
            f"• {insights['summary']}\n"
        ]
        
        if insights['insights']:
            text_parts.append("\n**Key Insights:**")
            for i, insight in enumerate(insights['insights'][:5], 1):
                text_parts.append(f"{i}. {insight}")
        
        if 'statistics' in insights and insights['statistics']:
            text_parts.append("\n**Statistics:**")
            
            # Numeric statistics
            if 'numeric' in insights['statistics']:
                text_parts.append("• Numeric columns found with statistical summaries")
            
            # Categorical statistics
            if 'categorical' in insights['statistics']:
                text_parts.append("• Categorical columns found with value distributions")
        
        return "\n".join(text_parts)
    
    def _format_recommendations(self, recommendations: List[str]) -> str:
        """Format recommendations as readable text."""
        if not recommendations:
            return "No specific recommendations at this time."
        
        text_parts = ["**Recommendations for further analysis:**"]
        
        for i, rec in enumerate(recommendations, 1):
            text_parts.append(f"{i}. {rec}")
        
        return "\n".join(text_parts)
    
    def get_schema_display(self) -> str:
        """Get formatted schema information for display."""
        if not self.schema_info['tables']:
            return "No schema information available."
        
        schema_text = "**Database Schema:**\n\n"
        
        for table_name, table_info in self.schema_info['tables'].items():
            schema_text += f"**{table_name}**\n"
            
            for col in table_info['columns']:
                col_info = f"• {col['name']} ({col['type']})"
                if col.get('primary_key'):
                    col_info += " [PK]"
                if not col.get('nullable', True):
                    col_info += " [NOT NULL]"
                schema_text += col_info + "\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def run_model_comparison(self) -> Tuple[str, str]:
        """Run model comparison and return results."""
        try:
            # Create dummy models for demonstration
            models = self.evaluator.create_dummy_models()
            test_data = self.evaluator.create_test_dataset()
            
            # Run evaluation
            results_df = self.evaluator.evaluate_models(test_data, models)
            
            # Format results as HTML table
            results_html = results_df.round(3).to_html(
                classes="table table-striped",
                escape=False,
                index=False
            )
            
            # Generate comparison report
            report = self.evaluator.generate_comparison_report(results_df)
            
            return results_html, report
            
        except Exception as e:
            error_msg = f"Error running model comparison: {str(e)}"
            return error_msg, error_msg
    

    def get_sample_questions(self) -> List[str]:
        """Get list of sample questions for the interface."""
        return [
            "Show me all customers",
            "What are the total sales by region?",
            "Find the top 5 products by price",
            "How many orders were placed?",
            "What is the average order value?",
            "Which customers have spent the most?",
            "Show me recent orders",
            "What products are in stock?"
        ]


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize the app
    app = NL2SQLApp()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .table {
        font-size: 12px;
    }
    #results-table {
        max-height: 400px;
        overflow-y: auto;
    }
    """
    
    with gr.Blocks(css=css, title="NL-to-SQL Assistant", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🚀 NL-to-SQL Assistant
        
        Convert natural language questions into SQL queries and get instant results with insights and visualizations.
        """)
        
        with gr.Tabs():
            
            # Main Query Tab
            with gr.TabItem("Query Assistant"):
                
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Ask a question about your data",
                            placeholder="e.g., Show me the top 5 customers by total spending",
                            lines=2
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("Generate SQL & Execute", variant="primary")
                            chart_type = gr.Dropdown(
                                choices=["auto", "bar", "line", "pie", "scatter", "histogram"],
                                value="auto",
                                label="Chart Type"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Sample Questions")
                        sample_questions = app.get_sample_questions()
                        
                        for question in sample_questions[:4]:
                            gr.Button(
                                question,
                                size="sm"
                            ).click(
                                lambda q=question: q,
                                outputs=question_input
                            )
                
                # Results Section
                with gr.Row():
                    with gr.Column():
                        sql_output = gr.Code(
                            label="Generated SQL",
                            language="sql",
                            lines=5
                        )
                        
                        results_output = gr.HTML(
                            label="Query Results"
                        )
                
                with gr.Row():
                    with gr.Column():
                        insights_output = gr.Markdown(
                            label="Insights & Analysis"
                        )
                    
                    with gr.Column():
                        recommendations_output = gr.Markdown(
                            label="Recommendations"
                        )
                
                # Chart Section
                with gr.Row():
                    chart_output = gr.Plot(
                        label="Data Visualization"
                    )
            
            # Schema Browser Tab
            with gr.TabItem("Database Schema"):
                schema_display = gr.Markdown(
                    value=app.get_schema_display(),
                    label="Database Schema"
                )
            
            # Model Comparison Tab
            with gr.TabItem("Model Comparison"):
                gr.Markdown("""
                ### Model Performance Comparison
                
                Compare different NL-to-SQL models on various metrics including execution correctness,
                exact match accuracy, schema compliance, and BLEU scores.
                """)
                
                compare_btn = gr.Button("Run Model Comparison", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        comparison_results = gr.HTML(
                            label="Comparison Results"
                        )
                    
                    with gr.Column():
                        comparison_report = gr.Textbox(
                            label="Detailed Report",
                            lines=15,
                            max_lines=20
                        )
        
        # Event handlers
        submit_btn.click(
            fn=app.process_query,
            inputs=[question_input, chart_type],
            outputs=[sql_output, results_output, insights_output, recommendations_output, chart_output]
        )
        
        # Allow Enter key to submit
        question_input.submit(
            fn=app.process_query,
            inputs=[question_input, chart_type],
            outputs=[sql_output, results_output, insights_output, recommendations_output, chart_output]
        )
        
        compare_btn.click(
            fn=app.run_model_comparison,
            outputs=[comparison_results, comparison_report]
        )
    
    return interface


def main():
    """Main function to launch the Gradio interface."""
    try:
        interface = create_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        print(f"Error launching interface: {e}")
        raise


if __name__ == "__main__":
    main()