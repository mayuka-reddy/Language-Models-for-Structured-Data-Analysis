#!/usr/bin/env python3
"""
Perfect Frontend Launcher for NL-to-SQL Assistant
Automatically loads training results and launches enhanced Gradio UI.

Usage:
    python launch_frontend.py
    
Features:
    - Auto-loads training results from scripts/train_all_techniques.py
    - Beautiful visualizations of all 5 prompting techniques
    - Interactive strategy comparison
    - RAG performance metrics
    - Real-time query testing
"""

import sys
from pathlib import Path
import json
import pickle

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_training_results():
    """Load training results from train_all_techniques.py output."""
    possible_paths = [
        Path('results/models/trained_results.json'),
        Path('results/models/trained_results.pkl'),
        Path('model_outputs/model_results.json'),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                if path.suffix == '.pkl':
                    with open(path, 'rb') as f:
                        results = pickle.load(f)
                else:
                    with open(path, 'r') as f:
                        results = json.load(f)
                print(f"✅ Loaded training results from: {path}")
                return results
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
    
    print("\n⚠️  No training results found!")
    print("📋 Please run the training script first:")
    print("   python scripts/train_all_techniques.py\n")
    return None


def create_performance_comparison_chart(results):
    """Create comprehensive performance comparison chart."""
    if not results or 'strategy_results' not in results:
        return None
    
    strategy_results = results['strategy_results']
    strategies = list(strategy_results.keys())
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('BLEU Scores', 'Success Rates', 
                       'Execution Times', 'Schema Compliance'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    
    # BLEU scores
    bleu_scores = [strategy_results[s]['avg_bleu_score'] for s in strategies]
    fig.add_trace(
        go.Bar(x=[s.replace('_', ' ').title() for s in strategies], 
               y=bleu_scores, 
               marker_color=colors,
               name='BLEU',
               showlegend=False),
        row=1, col=1
    )
    
    # Success rates
    success_rates = [strategy_results[s]['success_rate'] for s in strategies]
    fig.add_trace(
        go.Bar(x=[s.replace('_', ' ').title() for s in strategies], 
               y=success_rates,
               marker_color=colors,
               name='Success',
               showlegend=False),
        row=1, col=2
    )
    
    # Execution times
    exec_times = [strategy_results[s]['avg_execution_time'] * 1000 for s in strategies]  # Convert to ms
    fig.add_trace(
        go.Bar(x=[s.replace('_', ' ').title() for s in strategies], 
               y=exec_times,
               marker_color=colors,
               name='Time',
               showlegend=False),
        row=2, col=1
    )
    
    # Schema compliance
    compliance = [strategy_results[s]['schema_compliance_rate'] for s in strategies]
    fig.add_trace(
        go.Bar(x=[s.replace('_', ' ').title() for s in strategies], 
               y=compliance,
               marker_color=colors,
               name='Compliance',
               showlegend=False),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_yaxes(title_text="Rate", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=2)
    
    fig.update_layout(
        title_text="Comprehensive Strategy Performance Comparison",
        height=700,
        showlegend=False
    )
    
    return fig


def create_rag_performance_chart(results):
    """Create RAG performance visualization."""
    if not results or 'rag_performance' not in results:
        return None
    
    rag_perf = results['rag_performance']
    
    fig = go.Figure()
    
    # Gauge chart for relevance score
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=rag_perf['avg_relevance_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "RAG Relevance Score", 'font': {'size': 24}},
        delta={'reference': 0.75, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#ffcccc'},
                {'range': [0.5, 0.75], 'color': '#ffffcc'},
                {'range': [0.75, 1], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.75
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'size': 16}
    )
    
    return fig


def create_metrics_table(results):
    """Create comprehensive metrics table."""
    if not results or 'strategy_results' not in results:
        return pd.DataFrame()
    
    strategy_results = results['strategy_results']
    
    data = []
    for strategy, metrics in strategy_results.items():
        data.append({
            'Strategy': strategy.replace('_', ' ').title(),
            'BLEU Score': f"{metrics['avg_bleu_score']:.3f}",
            'Success Rate': f"{metrics['success_rate']:.1%}",
            'Execution Accuracy': f"{metrics['execution_accuracy']:.1%}",
            'Schema Compliance': f"{metrics['schema_compliance_rate']:.1%}",
            'Avg Confidence': f"{metrics['avg_confidence']:.3f}",
            'Avg Time (ms)': f"{metrics['avg_execution_time']*1000:.2f}"
        })
    
    return pd.DataFrame(data)


def create_ui(results):
    """Create the perfect Gradio UI."""
    
    with gr.Blocks(title="NL-to-SQL Assistant - Training Results", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚀 NL-to-SQL Assistant - Training Results Dashboard
        
        **Comprehensive evaluation of 5 prompting techniques with RAG pipeline**
        
        Dataset: Brazilian E-Commerce (Olist) | Techniques: Zero-Shot, Few-Shot, CoT, Self-Consistency, Least-to-Most
        """)
        
        if not results:
            gr.Markdown("""
            ## ⚠️ No Training Results Available
            
            Please run the training script first:
            ```bash
            python scripts/train_all_techniques.py
            ```
            
            This will:
            1. Train all 5 prompting techniques
            2. Evaluate RAG pipeline
            3. Generate comprehensive metrics
            4. Create visualizations
            5. Save results for this dashboard
            
            Then restart this frontend to see the results.
            """)
            return app
        
        with gr.Tabs():
            # Tab 1: Overview
            with gr.TabItem("📊 Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        metadata = results.get('metadata', {})
                        strategy_results = results.get('strategy_results', {})
                        
                        # Calculate best strategy
                        best_strategy = max(strategy_results.items(), 
                                          key=lambda x: x[1].get('avg_bleu_score', 0))[0]
                        best_metrics = strategy_results[best_strategy]
                        
                        gr.Markdown(f"""
                        ## 🏆 Training Summary
                        
                        **Dataset:** {metadata.get('dataset', 'Brazilian E-Commerce (Olist)')}  
                        **Training Date:** {metadata.get('timestamp', 'N/A')[:19]}  
                        **Total Questions:** {metadata.get('total_questions', len(strategy_results))}  
                        **Strategies Evaluated:** {metadata.get('total_strategies', len(strategy_results))}
                        
                        ### Best Performing Strategy
                        **{best_strategy.replace('_', ' ').title()}**
                        - BLEU Score: **{best_metrics['avg_bleu_score']:.3f}**
                        - Success Rate: **{best_metrics['success_rate']:.1%}**
                        - Execution Accuracy: **{best_metrics['execution_accuracy']:.1%}**
                        - Schema Compliance: **{best_metrics['schema_compliance_rate']:.1%}**
                        """)
                    
                    with gr.Column(scale=2):
                        # Performance comparison chart
                        perf_chart = create_performance_comparison_chart(results)
                        if perf_chart:
                            gr.Plot(value=perf_chart)
            
            # Tab 2: Detailed Metrics
            with gr.TabItem("📈 Detailed Metrics"):
                gr.Markdown("## Comprehensive Performance Metrics")
                
                metrics_df = create_metrics_table(results)
                if not metrics_df.empty:
                    gr.Dataframe(value=metrics_df, interactive=False)
                
                # RAG Performance
                if 'rag_performance' in results:
                    rag_perf = results['rag_performance']
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(f"""
                            ## 🔍 RAG Pipeline Performance
                            
                            - **Average Relevance Score:** {rag_perf['avg_relevance_score']:.3f}
                            - **Target (0.75+):** {'✅ ACHIEVED' if rag_perf.get('target_achieved', False) else '❌ NOT MET'}
                            - **Min Relevance:** {rag_perf.get('min_relevance_score', 0):.3f}
                            - **Max Relevance:** {rag_perf.get('max_relevance_score', 0):.3f}
                            - **Avg Retrieval Time:** {rag_perf['avg_retrieval_time']:.4f}s
                            - **Schema Cards Indexed:** {rag_perf['total_schema_cards']}
                            """)
                        
                        with gr.Column():
                            rag_chart = create_rag_performance_chart(results)
                            if rag_chart:
                                gr.Plot(value=rag_chart)
            
            # Tab 3: Sample Questions
            with gr.TabItem("📝 Sample Questions"):
                gr.Markdown("## Training Questions & Results")
                
                if 'training_questions' in results:
                    questions = results['training_questions'][:5]  # Show first 5
                    
                    for i, q in enumerate(questions, 1):
                        with gr.Accordion(f"Question {i}: {q['question']}", open=False):
                            gr.Markdown(f"""
                            **Category:** {q['category']}  
                            **Complexity:** {q['complexity']}  
                            **Requires JOIN:** {'Yes' if q.get('requires_join', False) else 'No'}
                            
                            **Reference SQL:**
                            ```sql
                            {q['sql']}
                            ```
                            """)
            
            # Tab 4: Export & Integration
            with gr.TabItem("💾 Export & Integration"):
                gr.Markdown("""
                ## Export Results & Integration Guide
                
                ### Available Files
                
                All results are automatically saved in the `results/` directory:
                
                - **Metrics Tables:** `results/metrics/overall_performance.csv`
                - **Visualizations:** `results/visualizations/*.png`
                - **Model Artifacts:** `results/models/trained_results.pkl`
                - **Reports:** `results/reports/training_summary.md`
                
                ### Load Results in Python
                
                ```python
                import pickle
                
                # Load complete results
                with open('results/models/trained_results.pkl', 'rb') as f:
                    results = pickle.load(f)
                
                # Access metrics
                print(f"Best strategy: {results['metadata']['best_strategy']}")
                print(f"RAG relevance: {results['rag_performance']['avg_relevance_score']:.3f}")
                ```
                
                ### Integration with Your Application
                
                ```python
                from models.techniques.prompting_strategies import PromptingEngine
                from models.techniques.rag_pipeline import EnhancedRAGPipeline
                
                # Initialize components
                engine = PromptingEngine()
                rag = EnhancedRAGPipeline(schema)
                
                # Use best strategy
                best_strategy = engine.get_strategy('chain_of_thought')
                prompt = best_strategy.generate_prompt(question, schema)
                ```
                """)
        
        gr.Markdown("""
        ---
        **🎯 Perfect Training Results Dashboard** | Generated from `scripts/train_all_techniques.py`
        """)
    
    return app


def main():
    """Main execution function."""
    print("="*70)
    print("🚀 LAUNCHING NL-TO-SQL FRONTEND")
    print("="*70)
    
    # Load training results
    print("\n📊 Loading training results...")
    results = load_training_results()
    
    if results:
        print(f"✅ Results loaded successfully!")
        print(f"   - Strategies: {len(results.get('strategy_results', {}))}")
        print(f"   - Questions: {results.get('metadata', {}).get('total_questions', 'N/A')}")
        if 'rag_performance' in results:
            print(f"   - RAG Relevance: {results['rag_performance']['avg_relevance_score']:.3f}")
    
    # Create and launch UI
    print("\n🌐 Creating Gradio interface...")
    app = create_ui(results)
    
    print("\n✨ Launching frontend...")
    print("="*70)
    print("📊 Dashboard URL: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop")
    print("="*70)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()