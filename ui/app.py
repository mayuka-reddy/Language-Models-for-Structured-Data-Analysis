"""
Enhanced Gradio UI for NL-to-SQL Assistant.
Production-ready interface with automatic results loading and comprehensive visualizations.
"""

import gradio as gr
import json
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE_URL = "http://localhost:8000"
STRATEGIES = ["zero_shot", "chain_of_thought", "few_shot", "self_consistency", "least_to_most"]

# Load trained model results
def load_trained_model_results():
    """Load trained model results from multiple possible locations."""
    possible_paths = [
        Path('../results/models/trained_results.json'),
        Path('results/models/trained_results.json'),
        Path('../model_outputs/model_results.json'),
        Path('model_outputs/model_results.json'),
        Path('data/model_results.json')
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    results = json.load(f)
                    print(f"✅ Loaded training results from: {path}")
                    return results
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
    
    print("⚠️  No training results found. Run: python scripts/train_all_techniques.py")
    return None

# Load model results at startup
TRAINED_MODEL_RESULTS = load_trained_model_results()


class NL2SQLInterface:
    """Main interface class for the Gradio app."""
    
    def __init__(self):
        self.current_results = {}
        self.query_history = []
    
    def query_single_strategy(
        self, 
        question: str, 
        strategy: str, 
        use_rag: bool = True
    ) -> Tuple[str, str, str, str]:
        """Query using a single strategy."""
        if not question.strip():
            return "", "", "", "Please enter a question."
        
        try:
            # Prepare request
            payload = {
                "question": question,
                "context": {
                    "strategy": strategy,
                    "use_rag": use_rag
                }
            }
            
            # Call API
            response = requests.post(f"{API_BASE_URL}/parse", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                # Format results
                sql_query = result.get("sql", "")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reason_short", "")
                tables_used = ", ".join(result.get("used_tables", []))
                
                # Store result
                self.current_results[strategy] = result
                
                return sql_query, f"{confidence:.2f}", reasoning, tables_used
            else:
                error_msg = f"API Error: {response.status_code}"
                return "", "", "", error_msg
                
        except requests.exceptions.RequestException as e:
            return "", "", "", f"Connection Error: {str(e)}"
        except Exception as e:
            return "", "", "", f"Error: {str(e)}"
    
    def compare_all_strategies(self, question: str, use_rag: bool = True) -> str:
        """Compare all prompting strategies on the same question."""
        if not question.strip():
            return "Please enter a question."
        
        results = {}
        
        for strategy in STRATEGIES:
            try:
                payload = {
                    "question": question,
                    "context": {
                        "strategy": strategy,
                        "use_rag": use_rag
                    }
                }
                
                response = requests.post(f"{API_BASE_URL}/parse", json=payload, timeout=30)
                
                if response.status_code == 200:
                    results[strategy] = response.json()
                else:
                    results[strategy] = {"error": f"API Error: {response.status_code}"}
                    
            except Exception as e:
                results[strategy] = {"error": str(e)}
        
        # Format comparison table
        comparison_html = self._format_comparison_table(results)
        
        # Store for history
        self.query_history.append({
            "question": question,
            "results": results,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        return comparison_html
    
    def _format_comparison_table(self, results: Dict[str, Any]) -> str:
        """Format comparison results as HTML table."""
        html = """
        <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <thead>
            <tr style="background-color: #f0f0f0;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Strategy</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">SQL Query</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: center;">Confidence</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Reasoning</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Tables Used</th>
            </tr>
        </thead>
        <tbody>
        """
        
        for strategy, result in results.items():
            if "error" in result:
                html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">{strategy.replace('_', ' ').title()}</td>
                    <td style="border: 1px solid #ddd; padding: 12px; color: red;" colspan="4">Error: {result['error']}</td>
                </tr>
                """
            else:
                sql = result.get("sql", "")[:100] + "..." if len(result.get("sql", "")) > 100 else result.get("sql", "")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reason_short", "")
                tables = ", ".join(result.get("used_tables", []))
                
                # Color code confidence
                conf_color = "#28a745" if confidence > 0.8 else "#ffc107" if confidence > 0.5 else "#dc3545"
                
                html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">{strategy.replace('_', ' ').title()}</td>
                    <td style="border: 1px solid #ddd; padding: 12px; font-family: monospace; font-size: 12px;">{sql}</td>
                    <td style="border: 1px solid #ddd; padding: 12px; text-align: center; color: {conf_color}; font-weight: bold;">{confidence:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 12px;">{reasoning}</td>
                    <td style="border: 1px solid #ddd; padding: 12px;">{tables}</td>
                </tr>
                """
        
        html += """
        </tbody>
        </table>
        </div>
        """
        
        return html
    
    def execute_query(self, question: str, strategy: str = "chain_of_thought") -> Tuple[str, str]:
        """Execute the generated SQL query and return results."""
        if not question.strip():
            return "", "Please enter a question."
        
        try:
            payload = {
                "question": question,
                "context": {"strategy": strategy}
            }
            
            response = requests.post(f"{API_BASE_URL}/execute", json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                # Format SQL
                sql_query = result.get("sql", "")
                
                # Format results as table
                query_results = result.get("results", [])
                if query_results:
                    df = pd.DataFrame(query_results)
                    results_html = df.to_html(classes="table table-striped", table_id="results-table")
                else:
                    results_html = "<p>No results returned.</p>"
                
                return sql_query, results_html
            else:
                return "", f"Execution Error: {response.status_code}"
                
        except Exception as e:
            return "", f"Error: {str(e)}"
    
    def create_performance_chart(self) -> go.Figure:
        """Create a performance comparison chart."""
        if not self.query_history:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No data available. Run some queries first!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Aggregate performance data
        strategy_performance = {strategy: [] for strategy in STRATEGIES}
        
        for query in self.query_history:
            for strategy, result in query["results"].items():
                if "error" not in result:
                    confidence = result.get("confidence", 0.0)
                    strategy_performance[strategy].append(confidence)
        
        # Create chart
        fig = go.Figure()
        
        for strategy, confidences in strategy_performance.items():
            if confidences:
                fig.add_trace(go.Box(
                    y=confidences,
                    name=strategy.replace('_', ' ').title(),
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
        
        fig.update_layout(
            title="Strategy Performance Comparison",
            yaxis_title="Confidence Score",
            xaxis_title="Strategy",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def get_query_history(self) -> str:
        """Get formatted query history."""
        if not self.query_history:
            return "No queries in history."
        
        html = """
        <div style="max-height: 400px; overflow-y: auto;">
        <h4>Query History</h4>
        """
        
        for i, query in enumerate(reversed(self.query_history[-10:])):  # Last 10 queries
            html += f"""
            <div style="border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px;">
                <strong>Query {len(self.query_history) - i}:</strong> {query['question']}<br>
                <small style="color: #666;">Time: {query['timestamp']}</small><br>
                <small>Strategies run: {len([r for r in query['results'].values() if 'error' not in r])}/{len(query['results'])}</small>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _create_model_performance_tab(self):
        """Create enhanced model performance display tab with comprehensive visualizations."""
        if not TRAINED_MODEL_RESULTS:
            return gr.Markdown("""
            ## ⚠️ No Training Results Found
            
            Please run the training script first:
            ```bash
            python scripts/train_all_techniques.py
            ```
            
            This will generate all results and metrics automatically.
            """)
        
        # Extract performance data
        if 'model_performance' in TRAINED_MODEL_RESULTS:
            # Old format
            model_perf = TRAINED_MODEL_RESULTS['model_performance']
            best_strategy = model_perf['best_strategy']
            overall = model_perf['overall_summary']
            strategy_results = model_perf['strategy_results']
        else:
            # New format from train_all_techniques.py
            strategy_results = TRAINED_MODEL_RESULTS.get('strategy_results', {})
            if not strategy_results:
                return gr.Markdown("## ⚠️ Invalid results format")
            
            # Calculate best strategy
            best_strategy = max(strategy_results.items(),
                              key=lambda x: x[1].get('avg_bleu_score', 0))[0]
            
            # Calculate overall metrics
            overall = {
                'avg_success_rate': sum(s.get('success_rate', 0) for s in strategy_results.values()) / len(strategy_results),
                'avg_bleu_score': sum(s.get('avg_bleu_score', 0) for s in strategy_results.values()) / len(strategy_results),
                'avg_execution_accuracy': sum(s.get('execution_accuracy', 0) for s in strategy_results.values()) / len(strategy_results),
                'total_strategies': len(strategy_results),
                'total_questions': TRAINED_MODEL_RESULTS.get('metadata', {}).get('total_questions', 10)
            }
        
        # Performance Summary
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"""
                ## 🏆 Training Results Summary
                
                **Best Strategy:** {best_strategy.replace('_', ' ').title()}  
                **Overall Success Rate:** {overall['avg_success_rate']:.1%}  
                **Average BLEU Score:** {overall['avg_bleu_score']:.3f}  
                **Average Execution Accuracy:** {overall['avg_execution_accuracy']:.1%}  
                **Total Questions Tested:** {overall['total_questions']}  
                **Training Completed:** {TRAINED_MODEL_RESULTS['metadata']['timestamp'][:19]}
                """)
            
            with gr.Column():
                # Create strategy comparison chart
                strategies = list(strategy_results.keys())
                success_rates = [strategy_results[s]['success_rate'] for s in strategies]
                bleu_scores = [strategy_results[s]['avg_bleu_score'] for s in strategies]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name='Success Rate',
                    x=[s.replace('_', ' ').title() for s in strategies],
                    y=success_rates,
                    yaxis='y',
                    offsetgroup=1
                ))
                fig.add_trace(go.Bar(
                    name='BLEU Score',
                    x=[s.replace('_', ' ').title() for s in strategies],
                    y=bleu_scores,
                    yaxis='y2',
                    offsetgroup=2
                ))
                
                fig.update_layout(
                    title='Strategy Performance Comparison',
                    xaxis_title='Strategy',
                    yaxis=dict(title='Success Rate', side='left'),
                    yaxis2=dict(title='BLEU Score', side='right', overlaying='y'),
                    barmode='group',
                    height=400
                )
                
                gr.Plot(value=fig)
        
        # Detailed Results Table
        gr.Markdown("## 📊 Detailed Strategy Results")
        
        # Create detailed results table
        detailed_data = []
        for strategy, metrics in strategy_results.items():
            detailed_data.append({
                'Strategy': strategy.replace('_', ' ').title(),
                'Success Rate': f"{metrics['success_rate']:.1%}",
                'BLEU Score': f"{metrics['avg_bleu_score']:.3f}",
                'Execution Accuracy': f"{metrics['execution_accuracy']:.1%}",
                'Avg Confidence': f"{metrics['avg_confidence']:.3f}",
                'Avg Time (ms)': f"{metrics['avg_execution_time']*1000:.2f}"
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        gr.Dataframe(value=detailed_df, interactive=False)
        
        # Sample Questions and Results
        gr.Markdown("## 📝 Sample Training Questions & Results")
        
        sample_questions = TRAINED_MODEL_RESULTS['detailed_results']['question_by_question'][:3]
        
        for i, q_data in enumerate(sample_questions, 1):
            with gr.Accordion(f"Question {i}: {q_data['question']}", open=False):
                gr.Markdown(f"""
                **Category:** {q_data['category']}  
                **Complexity:** {q_data['complexity']}  
                **Reference SQL:** 
                ```sql
                {q_data['reference_sql']}
                ```
                """)
                
                # Show results from best strategy
                if best_strategy in q_data['strategy_results']:
                    best_result = q_data['strategy_results'][best_strategy]
                    gr.Markdown(f"""
                    **Best Strategy Result ({best_strategy.replace('_', ' ').title()}):**
                    ```sql
                    {best_result['generated_sql']}
                    ```
                    **Confidence:** {best_result['confidence']:.3f}
                    """)
        
        # RAG Performance
        if 'rag_performance' in TRAINED_MODEL_RESULTS:
            rag_perf = TRAINED_MODEL_RESULTS['rag_performance']
            gr.Markdown(f"""
            ## 🔍 RAG Pipeline Performance
            
            **Average Retrieval Time:** {rag_perf['avg_retrieval_time']:.4f}s  
            **Average Relevance Score:** {rag_perf['avg_relevance_score']:.3f}  
            **Schema Cards Indexed:** {rag_perf['total_schema_cards']}
            """)
        
        return None


def create_interface():
    """Create the Gradio interface."""
    interface = NL2SQLInterface()
    
    with gr.Blocks(title="NL-to-SQL Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚀 NL-to-SQL Assistant
        
        Convert natural language questions into SQL queries using advanced prompting strategies, 
        schema-aware RAG, and fine-tuned models.
        """)
        
        with gr.Tabs():
            # Model Performance Tab (NEW)
            with gr.TabItem("🏆 Model Performance"):
                if TRAINED_MODEL_RESULTS:
                    interface._create_model_performance_tab()
                else:
                    gr.Markdown("""
                    ## ⚠️ No Trained Model Results Found
                    
                    Please run the training pipeline first:
                    1. Open `ML_Model_Training_Pipeline.ipynb`
                    2. Run all cells to train the model
                    3. Run `python frontend_integration.py`
                    4. Restart this UI
                    """)
            
            # Single Query Tab
            with gr.TabItem("Single Query"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Natural Language Question",
                            placeholder="e.g., Which city has the highest number of customers?",
                            lines=2
                        )
                        
                        with gr.Row():
                            strategy_dropdown = gr.Dropdown(
                                choices=STRATEGIES,
                                value="chain_of_thought",
                                label="Prompting Strategy"
                            )
                            use_rag_checkbox = gr.Checkbox(
                                label="Use RAG",
                                value=True
                            )
                        
                        query_button = gr.Button("Generate SQL", variant="primary")
                    
                    with gr.Column(scale=3):
                        sql_output = gr.Code(
                            label="Generated SQL",
                            language="sql"
                        )
                        
                        with gr.Row():
                            confidence_output = gr.Textbox(
                                label="Confidence",
                                interactive=False
                            )
                            tables_output = gr.Textbox(
                                label="Tables Used",
                                interactive=False
                            )
                        
                        reasoning_output = gr.Textbox(
                            label="Reasoning",
                            lines=3,
                            interactive=False
                        )
                
                # Execute Query Section
                with gr.Row():
                    execute_button = gr.Button("Execute Query", variant="secondary")
                
                with gr.Row():
                    execution_results = gr.HTML(label="Query Results")
            
            # Strategy Comparison Tab
            with gr.TabItem("Strategy Comparison"):
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_question = gr.Textbox(
                            label="Question to Compare",
                            placeholder="Enter a question to test all strategies",
                            lines=2
                        )
                        compare_rag = gr.Checkbox(
                            label="Use RAG",
                            value=True
                        )
                        compare_button = gr.Button("Compare All Strategies", variant="primary")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Strategy Performance")
                        performance_chart = gr.Plot()
                
                comparison_results = gr.HTML(label="Comparison Results")
            
            # Query History Tab
            with gr.TabItem("History & Analytics"):
                with gr.Row():
                    refresh_history_button = gr.Button("Refresh History")
                
                with gr.Row():
                    with gr.Column():
                        history_display = gr.HTML()
                    
                    with gr.Column():
                        gr.Markdown("### Performance Analytics")
                        analytics_chart = gr.Plot()
        
        # Event handlers
        query_button.click(
            fn=interface.query_single_strategy,
            inputs=[question_input, strategy_dropdown, use_rag_checkbox],
            outputs=[sql_output, confidence_output, reasoning_output, tables_output]
        )
        
        execute_button.click(
            fn=interface.execute_query,
            inputs=[question_input, strategy_dropdown],
            outputs=[sql_output, execution_results]
        )
        
        compare_button.click(
            fn=interface.compare_all_strategies,
            inputs=[compare_question, compare_rag],
            outputs=[comparison_results]
        )
        
        compare_button.click(
            fn=interface.create_performance_chart,
            outputs=[performance_chart]
        )
        
        refresh_history_button.click(
            fn=interface.get_query_history,
            outputs=[history_display]
        )
        
        refresh_history_button.click(
            fn=interface.create_performance_chart,
            outputs=[analytics_chart]
        )
    
    return app


def main():
    """Main function to launch the interface."""
    app = create_interface()
    
    # Launch with configuration
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()