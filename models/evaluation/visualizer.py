"""
Performance visualization components for ML model evaluation.
Creates training curves, comparative performance charts, and interactive dashboards.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PerformanceVisualizer:
    """Creates comprehensive performance visualizations for ML model evaluation."""
    
    def __init__(self, output_dir: str = "models/evaluation/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, training_history: Dict[str, List[float]], save_path: Optional[str] = None) -> str:
        """
        Create training curve plots showing loss and metrics over epochs.
        
        Args:
            training_history: Dictionary with 'train_loss', 'eval_loss', etc.
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Plot training and validation loss
        if 'train_loss' in training_history and 'eval_loss' in training_history:
            epochs = range(1, len(training_history['train_loss']) + 1)
            axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
            axes[0, 0].plot(epochs, training_history['eval_loss'], 'r-', label='Validation Loss', linewidth=2)
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot learning rate if available
        if 'learning_rate' in training_history:
            epochs = range(1, len(training_history['learning_rate']) + 1)
            axes[0, 1].plot(epochs, training_history['learning_rate'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot gradient norm if available
        if 'gradient_norm' in training_history:
            epochs = range(1, len(training_history['gradient_norm']) + 1)
            axes[1, 0].plot(epochs, training_history['gradient_norm'], 'purple', linewidth=2)
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot accuracy metrics if available
        if 'train_accuracy' in training_history:
            epochs = range(1, len(training_history['train_accuracy']) + 1)
            axes[1, 1].plot(epochs, training_history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
            if 'eval_accuracy' in training_history:
                axes[1, 1].plot(epochs, training_history['eval_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
            axes[1, 1].set_title('Accuracy Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "training_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comparative_performance_charts(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Create comparative performance charts for multiple models/techniques.
        
        Args:
            results_df: DataFrame with model performance metrics
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Metrics to plot
        metrics = [
            ('execution_accuracy', 'Execution Accuracy'),
            ('exact_match_accuracy', 'Exact Match Accuracy'),
            ('schema_compliance_rate', 'Schema Compliance'),
            ('avg_bleu_score', 'BLEU Score'),
            ('avg_response_time', 'Response Time (s)'),
            ('error_rate', 'Error Rate')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            row, col = idx // 3, idx % 3
            
            if metric in results_df.columns:
                # Bar plot for the metric
                bars = axes[row, col].bar(results_df['model'], results_df[metric], 
                                        color=sns.color_palette("husl", len(results_df)))
                axes[row, col].set_title(title)
                axes[row, col].set_xlabel('Model')
                axes[row, col].set_ylabel(title)
                axes[row, col].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                      f'{height:.3f}', ha='center', va='bottom')
                
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "comparative_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_confusion_matrices(self, predictions: List[bool], ground_truth: List[bool], 
                                technique_names: List[str], save_path: Optional[str] = None) -> str:
        """
        Create confusion matrices for classification metrics.
        
        Args:
            predictions: List of prediction results for each technique
            ground_truth: List of ground truth values
            technique_names: Names of techniques being compared
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn not available, skipping confusion matrices")
            return ""
        
        n_techniques = len(technique_names)
        fig, axes = plt.subplots(1, n_techniques, figsize=(5*n_techniques, 4))
        
        if n_techniques == 1:
            axes = [axes]
        
        fig.suptitle('Confusion Matrices by Technique', fontsize=16, fontweight='bold')
        
        for idx, (preds, name) in enumerate(zip(predictions, technique_names)):
            # Create confusion matrix
            cm = confusion_matrix(ground_truth, preds)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Incorrect', 'Correct'],
                       yticklabels=['Incorrect', 'Correct'],
                       ax=axes[idx])
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confusion_matrices.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def build_interactive_performance_dashboard(self, results_data: Dict[str, Any], 
                                             save_path: Optional[str] = None) -> str:
        """
        Build enhanced interactive performance dashboard using Plotly.
        
        Args:
            results_data: Dictionary containing performance data
            save_path: Optional path to save the HTML dashboard
            
        Returns:
            Path to saved dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available, creating static dashboard")
            return self._create_static_dashboard(results_data, save_path)
        
        # Create subplots with enhanced layout
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'Model Accuracy Comparison', 'Response Time Distribution', 'BLEU Score Analysis',
                'Error Rate by Model', 'Confidence Calibration', 'Query Complexity Performance',
                'Training Progress', 'Schema Compliance', 'Performance Correlation',
                'Technique Radar Chart', 'Statistical Significance', 'Performance Trends'
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatterpolar"}, {"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Extract data
        models = results_data.get('models', [])
        metrics = results_data.get('metrics', {})
        evaluation_results = results_data.get('evaluation_results', [])
        
        if models and metrics:
            # 1. Model accuracy comparison
            execution_accuracy = metrics.get('execution_accuracy', [])
            exact_match = metrics.get('exact_match_accuracy', [])
            
            fig.add_trace(
                go.Bar(x=models, y=execution_accuracy, name='Execution Accuracy', 
                      marker_color='lightblue', text=[f'{x:.1%}' for x in execution_accuracy],
                      textposition='auto'),
                row=1, col=1
            )
            
            # 2. Response time box plot
            response_times = metrics.get('response_times', {})
            for i, model in enumerate(models):
                if model in response_times:
                    fig.add_trace(
                        go.Box(y=response_times[model], name=model, 
                              boxpoints='outliers', jitter=0.3),
                        row=1, col=2
                    )
            
            # 3. BLEU score histogram
            bleu_scores = metrics.get('bleu_scores', [])
            if bleu_scores:
                fig.add_trace(
                    go.Histogram(x=bleu_scores, name='BLEU Scores', 
                               marker_color='lightgreen', nbinsx=20),
                    row=1, col=3
                )
            
            # 4. Error rate by model
            error_rates = metrics.get('error_rates', [])
            if error_rates:
                fig.add_trace(
                    go.Bar(x=models, y=error_rates, name='Error Rate', 
                          marker_color='lightcoral', text=[f'{x:.1%}' for x in error_rates],
                          textposition='auto'),
                    row=2, col=1
                )
            
            # 5. Confidence calibration scatter
            if evaluation_results:
                confidences = [r.get('confidence_score', 0) for r in evaluation_results]
                accuracies = [1 if r.get('execution_correct', False) else 0 for r in evaluation_results]
                
                fig.add_trace(
                    go.Scatter(x=confidences, y=accuracies, mode='markers',
                             name='Confidence vs Accuracy', marker_color='purple',
                             opacity=0.6),
                    row=2, col=2
                )
            
            # 6. Query complexity performance
            complexity_data = metrics.get('complexity_analysis', {})
            if complexity_data:
                complexity_types = list(complexity_data.keys())
                complexity_accuracies = [complexity_data[ct].get('accuracy', 0) for ct in complexity_types]
                
                fig.add_trace(
                    go.Bar(x=complexity_types, y=complexity_accuracies, 
                          name='Complexity Performance', marker_color='orange'),
                    row=2, col=3
                )
            
            # 7. Training progress (if available)
            training_data = results_data.get('training_history', {})
            if training_data:
                epochs = list(range(1, len(training_data.get('train_loss', [])) + 1))
                train_loss = training_data.get('train_loss', [])
                val_loss = training_data.get('eval_loss', [])
                
                if train_loss:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=train_loss, mode='lines', 
                                 name='Training Loss', line_color='blue'),
                        row=3, col=1
                    )
                if val_loss:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=val_loss, mode='lines', 
                                 name='Validation Loss', line_color='red'),
                        row=3, col=1
                    )
            
            # 8. Schema compliance
            schema_compliance = metrics.get('schema_compliance_rate', [])
            if schema_compliance:
                fig.add_trace(
                    go.Bar(x=models, y=schema_compliance, name='Schema Compliance',
                          marker_color='lightseagreen', text=[f'{x:.1%}' for x in schema_compliance],
                          textposition='auto'),
                    row=3, col=2
                )
            
            # 9. Performance correlation heatmap
            correlation_data = results_data.get('correlation_matrix', {})
            if correlation_data:
                metrics_names = list(correlation_data.keys())
                correlation_values = [[correlation_data[m1].get(m2, 0) for m2 in metrics_names] 
                                    for m1 in metrics_names]
                
                fig.add_trace(
                    go.Heatmap(z=correlation_values, x=metrics_names, y=metrics_names,
                             colorscale='RdBu', zmid=0),
                    row=3, col=3
                )
        
        # Update layout with enhanced styling
        fig.update_layout(
            title_text="Enhanced ML Model Performance Dashboard",
            title_x=0.5,
            title_font_size=20,
            height=1600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="Response Time (s)", row=1, col=2)
        
        fig.update_xaxes(title_text="BLEU Score", row=1, col=3)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        
        # Add annotations for key insights
        if models and metrics:
            best_model_idx = np.argmax(execution_accuracy) if execution_accuracy else 0
            best_model = models[best_model_idx] if models else "Unknown"
            
            fig.add_annotation(
                text=f"Best Model: {best_model}",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=14, color="green"),
                bgcolor="lightgreen",
                bordercolor="green",
                borderwidth=1
            )
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "enhanced_interactive_dashboard.html"
        
        fig.write_html(save_path, include_plotlyjs=True)
        
        return str(save_path)
    
    def _create_static_dashboard(self, results_data: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Create static dashboard when Plotly is not available."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('ML Model Performance Dashboard', fontsize=20, fontweight='bold')
        
        # Extract data
        models = results_data.get('models', ['Model A', 'Model B', 'Model C'])
        metrics = results_data.get('metrics', {})
        
        # Model accuracy comparison
        accuracy = metrics.get('execution_accuracy', [0.8, 0.85, 0.9])
        axes[0, 0].bar(models, accuracy, color='lightblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Execution Accuracy')
        
        # Response time analysis
        response_times = metrics.get('avg_response_time', [1.2, 0.8, 1.5])
        axes[0, 1].bar(models, response_times, color='lightcoral')
        axes[0, 1].set_title('Average Response Time')
        axes[0, 1].set_ylabel('Time (seconds)')
        
        # BLEU score distribution
        bleu_scores = metrics.get('bleu_scores', np.random.normal(0.7, 0.1, 100))
        axes[1, 0].hist(bleu_scores, bins=20, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('BLEU Score Distribution')
        axes[1, 0].set_xlabel('BLEU Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # Error rate by model
        error_rates = metrics.get('error_rates', [0.1, 0.05, 0.08])
        axes[1, 1].bar(models, error_rates, color='orange')
        axes[1, 1].set_title('Error Rate by Model')
        axes[1, 1].set_ylabel('Error Rate')
        
        # Training progress (mock data)
        epochs = range(1, 21)
        train_loss = [2.5 - 0.1*i + 0.05*np.random.randn() for i in epochs]
        val_loss = [2.3 - 0.08*i + 0.08*np.random.randn() for i in epochs]
        axes[2, 0].plot(epochs, train_loss, 'b-', label='Training Loss')
        axes[2, 0].plot(epochs, val_loss, 'r-', label='Validation Loss')
        axes[2, 0].set_title('Training Progress')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].legend()
        
        # Confidence vs Accuracy scatter
        confidence = metrics.get('confidence_scores', np.random.uniform(0.5, 1.0, 50))
        accuracy_scatter = metrics.get('accuracy_scatter', np.random.uniform(0.6, 1.0, 50))
        axes[2, 1].scatter(confidence, accuracy_scatter, alpha=0.6)
        axes[2, 1].set_title('Confidence vs Accuracy')
        axes[2, 1].set_xlabel('Confidence Score')
        axes[2, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "static_dashboard.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_technique_comparison_radar(self, technique_metrics: Dict[str, Dict[str, float]], 
                                        save_path: Optional[str] = None) -> str:
        """
        Create radar chart comparing different techniques across multiple metrics.
        
        Args:
            technique_metrics: Dict of technique names to their metric scores
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not technique_metrics:
            return ""
        
        # Prepare data
        techniques = list(technique_metrics.keys())
        metrics = list(next(iter(technique_metrics.values())).keys())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of metrics
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each technique
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
        
        for technique, color in zip(techniques, colors):
            values = [technique_metrics[technique][metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=technique, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Technique Performance Comparison', size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "technique_radar_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_response_time_distribution(self, response_times_by_technique: Dict[str, List[float]], 
                                      save_path: Optional[str] = None) -> str:
        """
        Create response time distribution plots for different techniques.
        
        Args:
            response_times_by_technique: Dict mapping technique names to response time lists
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Response Time Analysis by Technique', fontsize=16, fontweight='bold')
        
        techniques = list(response_times_by_technique.keys())
        colors = sns.color_palette("husl", len(techniques))
        
        # Box plot comparison
        ax1 = axes[0, 0]
        data_for_box = []
        labels_for_box = []
        
        for technique, times in response_times_by_technique.items():
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                data_for_box.append(valid_times)
                labels_for_box.append(technique)
        
        if data_for_box:
            ax1.boxplot(data_for_box, labels=labels_for_box)
            ax1.set_title('Response Time Distribution (Box Plot)')
            ax1.set_ylabel('Response Time (seconds)')
            ax1.tick_params(axis='x', rotation=45)
        
        # Histogram overlay
        ax2 = axes[0, 1]
        for i, (technique, times) in enumerate(response_times_by_technique.items()):
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                ax2.hist(valid_times, alpha=0.6, label=technique, color=colors[i], bins=20)
        
        ax2.set_title('Response Time Histograms')
        ax2.set_xlabel('Response Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Cumulative distribution
        ax3 = axes[1, 0]
        for i, (technique, times) in enumerate(response_times_by_technique.items()):
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                sorted_times = np.sort(valid_times)
                y = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
                ax3.plot(sorted_times, y, label=technique, color=colors[i], linewidth=2)
        
        ax3.set_title('Cumulative Distribution Function')
        ax3.set_xlabel('Response Time (seconds)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Performance percentiles
        ax4 = axes[1, 1]
        percentiles = [50, 75, 90, 95, 99]
        technique_names = []
        percentile_data = {p: [] for p in percentiles}
        
        for technique, times in response_times_by_technique.items():
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                technique_names.append(technique)
                for p in percentiles:
                    percentile_data[p].append(np.percentile(valid_times, p))
        
        x = np.arange(len(technique_names))
        width = 0.15
        
        for i, p in enumerate(percentiles):
            ax4.bar(x + i * width, percentile_data[p], width, 
                   label=f'P{p}', alpha=0.8)
        
        ax4.set_title('Response Time Percentiles')
        ax4.set_xlabel('Technique')
        ax4.set_ylabel('Response Time (seconds)')
        ax4.set_xticks(x + width * 2)
        ax4.set_xticklabels(technique_names, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "response_time_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_accuracy_confidence_scatter(self, evaluation_results: List, 
                                         save_path: Optional[str] = None) -> str:
        """
        Create scatter plot showing relationship between confidence and accuracy.
        
        Args:
            evaluation_results: List of evaluation results with confidence and accuracy
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Confidence vs Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        confidences = []
        accuracies = []
        bleu_scores = []
        
        for result in evaluation_results:
            if hasattr(result, 'confidence_score') and hasattr(result, 'execution_correct'):
                confidences.append(result.confidence_score)
                accuracies.append(1.0 if result.execution_correct else 0.0)
                if hasattr(result, 'bleu_score'):
                    bleu_scores.append(result.bleu_score)
        
        if not confidences:
            # Create empty plot with message
            axes[0].text(0.5, 0.5, 'No confidence data available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'No confidence data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        else:
            # Confidence vs Binary Accuracy
            axes[0].scatter(confidences, accuracies, alpha=0.6, s=50)
            axes[0].set_xlabel('Confidence Score')
            axes[0].set_ylabel('Execution Correct (0/1)')
            axes[0].set_title('Confidence vs Binary Accuracy')
            axes[0].grid(True, alpha=0.3)
            
            # Add trend line
            if len(confidences) > 1:
                z = np.polyfit(confidences, accuracies, 1)
                p = np.poly1d(z)
                axes[0].plot(sorted(confidences), p(sorted(confidences)), "r--", alpha=0.8)
            
            # Confidence vs BLEU Score (if available)
            if bleu_scores:
                scatter = axes[1].scatter(confidences, bleu_scores, alpha=0.6, s=50, 
                                        c=accuracies, cmap='RdYlGn', vmin=0, vmax=1)
                axes[1].set_xlabel('Confidence Score')
                axes[1].set_ylabel('BLEU Score')
                axes[1].set_title('Confidence vs BLEU Score (colored by accuracy)')
                axes[1].grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=axes[1])
                cbar.set_label('Execution Correct')
                
                # Add trend line
                if len(confidences) > 1:
                    z = np.polyfit(confidences, bleu_scores, 1)
                    p = np.poly1d(z)
                    axes[1].plot(sorted(confidences), p(sorted(confidences)), "r--", alpha=0.8)
            else:
                axes[1].text(0.5, 0.5, 'No BLEU score data available', 
                           ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confidence_accuracy_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_error_analysis_dashboard(self, evaluation_results: List, 
                                      save_path: Optional[str] = None) -> str:
        """
        Create comprehensive error analysis dashboard.
        
        Args:
            evaluation_results: List of evaluation results
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # Extract error data
        error_types = []
        schema_compliance = []
        execution_correct = []
        bleu_scores = []
        query_lengths = []
        
        for result in evaluation_results:
            if hasattr(result, 'error_message') and result.error_message:
                error_types.append('Error')
            else:
                error_types.append('Success')
            
            if hasattr(result, 'schema_compliant'):
                schema_compliance.append(result.schema_compliant)
            if hasattr(result, 'execution_correct'):
                execution_correct.append(result.execution_correct)
            if hasattr(result, 'bleu_score'):
                bleu_scores.append(result.bleu_score)
            if hasattr(result, 'predicted_sql'):
                query_lengths.append(len(result.predicted_sql))
        
        # Error rate pie chart
        if error_types:
            error_counts = pd.Series(error_types).value_counts()
            axes[0, 0].pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Success vs Error Rate')
        
        # Schema compliance
        if schema_compliance:
            compliance_counts = pd.Series(schema_compliance).value_counts()
            axes[0, 1].bar(['Non-Compliant', 'Compliant'], 
                          [compliance_counts.get(False, 0), compliance_counts.get(True, 0)])
            axes[0, 1].set_title('Schema Compliance')
            axes[0, 1].set_ylabel('Count')
        
        # Execution correctness by BLEU score
        if execution_correct and bleu_scores:
            correct_bleu = [bleu for correct, bleu in zip(execution_correct, bleu_scores) if correct]
            incorrect_bleu = [bleu for correct, bleu in zip(execution_correct, bleu_scores) if not correct]
            
            axes[0, 2].hist([correct_bleu, incorrect_bleu], bins=20, alpha=0.7, 
                           label=['Correct', 'Incorrect'], color=['green', 'red'])
            axes[0, 2].set_title('BLEU Score Distribution by Correctness')
            axes[0, 2].set_xlabel('BLEU Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
        
        # Query length vs accuracy
        if query_lengths and execution_correct:
            correct_lengths = [length for length, correct in zip(query_lengths, execution_correct) if correct]
            incorrect_lengths = [length for length, correct in zip(query_lengths, execution_correct) if not correct]
            
            axes[1, 0].hist([correct_lengths, incorrect_lengths], bins=20, alpha=0.7,
                           label=['Correct', 'Incorrect'], color=['green', 'red'])
            axes[1, 0].set_title('Query Length Distribution by Correctness')
            axes[1, 0].set_xlabel('Query Length (characters)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Performance correlation matrix
        if all([execution_correct, bleu_scores, schema_compliance]):
            corr_data = pd.DataFrame({
                'Execution_Correct': [1 if x else 0 for x in execution_correct],
                'BLEU_Score': bleu_scores,
                'Schema_Compliant': [1 if x else 0 for x in schema_compliance]
            })
            
            correlation_matrix = corr_data.corr()
            im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_xticks(range(len(correlation_matrix.columns)))
            axes[1, 1].set_yticks(range(len(correlation_matrix.columns)))
            axes[1, 1].set_xticklabels(correlation_matrix.columns, rotation=45)
            axes[1, 1].set_yticklabels(correlation_matrix.columns)
            axes[1, 1].set_title('Metric Correlation Matrix')
            
            # Add correlation values
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                   ha='center', va='center', color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')
        
        # Error rate by query complexity (if we can determine complexity)
        if query_lengths and execution_correct:
            # Simple complexity based on length
            complexity_bins = ['Short (<50)', 'Medium (50-150)', 'Long (>150)']
            complexity_errors = [0, 0, 0]
            complexity_totals = [0, 0, 0]
            
            for length, correct in zip(query_lengths, execution_correct):
                if length < 50:
                    idx = 0
                elif length < 150:
                    idx = 1
                else:
                    idx = 2
                
                complexity_totals[idx] += 1
                if not correct:
                    complexity_errors[idx] += 1
            
            error_rates = [errors/total if total > 0 else 0 for errors, total in zip(complexity_errors, complexity_totals)]
            
            axes[1, 2].bar(complexity_bins, error_rates, color='orange', alpha=0.7)
            axes[1, 2].set_title('Error Rate by Query Complexity')
            axes[1, 2].set_ylabel('Error Rate')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "error_analysis_dashboard.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def create_comprehensive_performance_summary(self, results_data: Dict[str, Any], 
                                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive performance summary with multiple visualization types.
        
        Args:
            results_data: Dictionary containing all performance data
            save_path: Optional path to save the summary plot
            
        Returns:
            Path to saved summary plot
        """
        fig = plt.figure(figsize=(20, 24))
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Comprehensive ML Model Performance Summary', fontsize=24, fontweight='bold', y=0.98)
        
        # Extract data
        models = results_data.get('models', [])
        metrics = results_data.get('metrics', {})
        evaluation_results = results_data.get('evaluation_results', [])
        
        # 1. Overall Performance Comparison (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        if models and metrics:
            execution_acc = metrics.get('execution_accuracy', [])
            exact_match = metrics.get('exact_match_accuracy', [])
            bleu_scores = metrics.get('avg_bleu_score', [])
            
            x = np.arange(len(models))
            width = 0.25
            
            ax1.bar(x - width, execution_acc, width, label='Execution Accuracy', alpha=0.8)
            ax1.bar(x, exact_match, width, label='Exact Match', alpha=0.8)
            ax1.bar(x + width, bleu_scores, width, label='BLEU Score', alpha=0.8)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Overall Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Response Time Analysis
        ax2 = fig.add_subplot(gs[0, 2:])
        response_times = metrics.get('response_times', {})
        if response_times:
            data_for_box = []
            labels_for_box = []
            for model, times in response_times.items():
                valid_times = [t for t in times if t != float('inf')]
                if valid_times:
                    data_for_box.append(valid_times)
                    labels_for_box.append(model)
            
            if data_for_box:
                ax2.boxplot(data_for_box, labels=labels_for_box)
                ax2.set_title('Response Time Distribution')
                ax2.set_ylabel('Time (seconds)')
                ax2.tick_params(axis='x', rotation=45)
        
        # 3. Error Analysis (second row)
        ax3 = fig.add_subplot(gs[1, :2])
        if evaluation_results:
            error_counts = {'Success': 0, 'Error': 0}
            for result in evaluation_results:
                if result.get('error_message'):
                    error_counts['Error'] += 1
                else:
                    error_counts['Success'] += 1
            
            ax3.pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
            ax3.set_title('Success vs Error Rate')
        
        # 4. Schema Compliance
        ax4 = fig.add_subplot(gs[1, 2:])
        schema_compliance = metrics.get('schema_compliance_rate', [])
        if schema_compliance and models:
            bars = ax4.bar(models, schema_compliance, color='lightseagreen', alpha=0.8)
            ax4.set_title('Schema Compliance by Model')
            ax4.set_ylabel('Compliance Rate')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, schema_compliance):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # 5. Confidence vs Accuracy (third row)
        ax5 = fig.add_subplot(gs[2, :2])
        if evaluation_results:
            confidences = [r.get('confidence_score', 0) for r in evaluation_results]
            accuracies = [1 if r.get('execution_correct', False) else 0 for r in evaluation_results]
            
            if confidences and accuracies:
                ax5.scatter(confidences, accuracies, alpha=0.6, s=30)
                ax5.set_xlabel('Confidence Score')
                ax5.set_ylabel('Execution Correct (0/1)')
                ax5.set_title('Confidence vs Accuracy')
                ax5.grid(True, alpha=0.3)
                
                # Add trend line
                if len(confidences) > 1:
                    z = np.polyfit(confidences, accuracies, 1)
                    p = np.poly1d(z)
                    ax5.plot(sorted(confidences), p(sorted(confidences)), "r--", alpha=0.8)
        
        # 6. Query Complexity Analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        complexity_data = metrics.get('complexity_analysis', {})
        if complexity_data:
            complexity_types = list(complexity_data.keys())
            complexity_accuracies = [complexity_data[ct].get('accuracy', 0) for ct in complexity_types]
            
            bars = ax6.bar(complexity_types, complexity_accuracies, color='orange', alpha=0.8)
            ax6.set_title('Performance by Query Complexity')
            ax6.set_ylabel('Accuracy')
            ax6.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, complexity_accuracies):
                ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.1%}', ha='center', va='bottom')
        
        # 7. Performance Statistics Table (fourth row)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        if models and metrics:
            # Create performance statistics table
            table_data = []
            for i, model in enumerate(models):
                row = [
                    model,
                    f"{execution_acc[i]:.1%}" if i < len(execution_acc) else "N/A",
                    f"{exact_match[i]:.1%}" if i < len(exact_match) else "N/A",
                    f"{bleu_scores[i]:.3f}" if i < len(bleu_scores) else "N/A",
                    f"{schema_compliance[i]:.1%}" if i < len(schema_compliance) else "N/A",
                    f"{np.mean(response_times.get(model, [0])):.3f}s" if model in response_times else "N/A"
                ]
                table_data.append(row)
            
            table = ax7.table(cellText=table_data,
                            colLabels=['Model', 'Exec Acc', 'Exact Match', 'BLEU', 'Schema Comp', 'Avg Time'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax7.set_title('Performance Statistics Summary', pad=20)
        
        # 8. Training Progress (if available) (fifth row)
        ax8 = fig.add_subplot(gs[4, :2])
        training_data = results_data.get('training_history', {})
        if training_data:
            epochs = list(range(1, len(training_data.get('train_loss', [])) + 1))
            train_loss = training_data.get('train_loss', [])
            val_loss = training_data.get('eval_loss', [])
            
            if train_loss:
                ax8.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
            if val_loss:
                ax8.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
            
            ax8.set_title('Training Progress')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Loss')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        # 9. Performance Trends
        ax9 = fig.add_subplot(gs[4, 2:])
        if models and metrics:
            # Create a simple trend analysis
            metric_names = ['Execution Acc', 'Exact Match', 'BLEU Score', 'Schema Comp']
            metric_values = [
                np.mean(execution_acc) if execution_acc else 0,
                np.mean(exact_match) if exact_match else 0,
                np.mean(bleu_scores) if bleu_scores else 0,
                np.mean(schema_compliance) if schema_compliance else 0
            ]
            
            bars = ax9.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
            ax9.set_title('Average Performance Across All Models')
            ax9.set_ylabel('Score')
            ax9.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, metric_values):
                ax9.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 10. Key Insights Text (bottom row)
        ax10 = fig.add_subplot(gs[5, :])
        ax10.axis('off')
        
        # Generate key insights
        insights = []
        if models and metrics:
            best_model_idx = np.argmax(execution_acc) if execution_acc else 0
            best_model = models[best_model_idx] if models else "Unknown"
            best_accuracy = execution_acc[best_model_idx] if execution_acc else 0
            
            insights.append(f"• Best performing model: {best_model} ({best_accuracy:.1%} execution accuracy)")
            
            if response_times:
                fastest_model = min(response_times.keys(), 
                                  key=lambda x: np.mean([t for t in response_times[x] if t != float('inf')]))
                avg_time = np.mean([t for t in response_times[fastest_model] if t != float('inf')])
                insights.append(f"• Fastest model: {fastest_model} ({avg_time:.3f}s average response time)")
            
            if schema_compliance:
                avg_compliance = np.mean(schema_compliance)
                insights.append(f"• Average schema compliance: {avg_compliance:.1%}")
            
            if evaluation_results:
                total_queries = len(evaluation_results)
                successful_queries = sum(1 for r in evaluation_results if not r.get('error_message'))
                insights.append(f"• Overall success rate: {successful_queries/total_queries:.1%} ({successful_queries}/{total_queries} queries)")
        
        insights_text = "\n".join(insights) if insights else "No insights available"
        ax10.text(0.05, 0.8, "Key Insights:", fontsize=16, fontweight='bold', transform=ax10.transAxes)
        ax10.text(0.05, 0.2, insights_text, fontsize=12, transform=ax10.transAxes, verticalalignment='top')
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "comprehensive_performance_summary.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(save_path)
    
    def save_visualization_summary(self, plot_paths: List[str]) -> str:
        """
        Create a summary document of all generated visualizations.
        
        Args:
            plot_paths: List of paths to generated plots
            
        Returns:
            Path to summary document
        """
        summary_path = self.output_dir / "visualization_summary.json"
        
        summary = {
            "generated_plots": plot_paths,
            "total_plots": len(plot_paths),
            "output_directory": str(self.output_dir),
            "plot_descriptions": {
                "training_curves.png": "Training progress showing loss and metrics over epochs",
                "comparative_performance.png": "Bar charts comparing model performance across metrics",
                "confusion_matrices.png": "Confusion matrices for classification accuracy",
                "interactive_dashboard.html": "Interactive dashboard with multiple performance views",
                "enhanced_interactive_dashboard.html": "Enhanced interactive dashboard with comprehensive analysis",
                "technique_radar_comparison.png": "Radar chart comparing techniques across metrics",
                "response_time_analysis.png": "Comprehensive response time distribution analysis",
                "confidence_accuracy_analysis.png": "Confidence vs accuracy correlation analysis",
                "error_analysis_dashboard.png": "Comprehensive error analysis and debugging dashboard",
                "comprehensive_performance_summary.png": "Complete performance summary with all key metrics"
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)


# Example usage and testing
if __name__ == "__main__":
    # Test the visualizer
    visualizer = PerformanceVisualizer()
    
    print("Testing Performance Visualizer:")
    print("=" * 50)
    
    # Mock training history
    training_history = {
        'train_loss': [2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 0.9, 0.8],
        'eval_loss': [2.3, 2.0, 1.9, 1.6, 1.4, 1.2, 1.0, 0.9],
        'learning_rate': [5e-5, 4e-5, 3e-5, 2e-5, 1e-5, 8e-6, 6e-6, 4e-6]
    }
    
    # Create training curves
    training_plot = visualizer.plot_training_curves(training_history)
    print(f"Training curves saved to: {training_plot}")
    
    # Mock performance data
    results_df = pd.DataFrame({
        'model': ['CoT', 'Few-Shot', 'Self-Consistency', 'RAG'],
        'execution_accuracy': [0.85, 0.78, 0.92, 0.88],
        'exact_match_accuracy': [0.72, 0.65, 0.85, 0.80],
        'schema_compliance_rate': [0.95, 0.90, 0.98, 0.96],
        'avg_bleu_score': [0.68, 0.62, 0.75, 0.71],
        'avg_response_time': [1.2, 0.8, 2.1, 1.5],
        'error_rate': [0.08, 0.12, 0.05, 0.07]
    })
    
    # Create comparative charts
    comparison_plot = visualizer.create_comparative_performance_charts(results_df)
    print(f"Comparative performance charts saved to: {comparison_plot}")
    
    # Create interactive dashboard
    dashboard_data = {
        'models': results_df['model'].tolist(),
        'metrics': {
            'execution_accuracy': results_df['execution_accuracy'].tolist(),
            'avg_response_time': results_df['avg_response_time'].tolist(),
            'error_rates': results_df['error_rate'].tolist()
        }
    }
    
    dashboard_path = visualizer.build_interactive_performance_dashboard(dashboard_data)
    print(f"Interactive dashboard saved to: {dashboard_path}")
    
    print("\nVisualization testing completed successfully!")