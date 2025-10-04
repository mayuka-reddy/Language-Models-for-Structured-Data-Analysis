"""
Chart Generator Module
Creates visualizations from query results using matplotlib and plotly.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from loguru import logger
import io
import base64


class ChartGenerator:
    """
    Generates various types of charts and visualizations from query results.
    
    Supports both matplotlib (static) and plotly (interactive) charts
    with automatic chart type selection based on data characteristics.
    """
    
    def __init__(self, style: str = "plotly"):
        """
        Initialize chart generator.
        
        Args:
            style: Chart library to use ('plotly' or 'matplotlib')
        """
        self.style = style.lower()
        
        # Set matplotlib style
        if self.style == "matplotlib":
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        
        logger.info(f"Chart Generator initialized with {style} style")
    
    def generate_chart(self, data: List[Dict[str, Any]], chart_type: str = "auto") -> Dict[str, Any]:
        """
        Generate appropriate chart from data.
        
        Args:
            data: List of dictionaries representing query results
            chart_type: Type of chart to generate ('auto', 'bar', 'line', 'pie', 'scatter')
            
        Returns:
            Dictionary containing chart data and metadata
        """
        if not data:
            return {
                'success': False,
                'error': 'No data provided for chart generation',
                'chart': None
            }
        
        try:
            df = pd.DataFrame(data)
            
            # Auto-select chart type if not specified
            if chart_type == "auto":
                chart_type = self._select_chart_type(df)
            
            # Generate chart based on type
            if chart_type == "bar":
                chart_result = self._create_bar_chart(df)
            elif chart_type == "line":
                chart_result = self._create_line_chart(df)
            elif chart_type == "pie":
                chart_result = self._create_pie_chart(df)
            elif chart_type == "scatter":
                chart_result = self._create_scatter_chart(df)
            elif chart_type == "histogram":
                chart_result = self._create_histogram(df)
            else:
                # Default to bar chart
                chart_result = self._create_bar_chart(df)
            
            return {
                'success': True,
                'chart_type': chart_type,
                'chart': chart_result,
                'data_summary': self._get_data_summary(df)
            }
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return {
                'success': False,
                'error': str(e),
                'chart': None
            }
    
    def _select_chart_type(self, df: pd.DataFrame) -> str:
        """
        Automatically select appropriate chart type based on data characteristics.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            Recommended chart type
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        
        row_count = len(df)
        
        # Decision logic for chart type
        if len(numeric_cols) >= 2:
            # Multiple numeric columns - scatter plot
            return "scatter"
        elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
            if row_count <= 10:
                # Few categories - pie chart
                return "pie"
            else:
                # Many categories - bar chart
                return "bar"
        elif len(numeric_cols) == 1:
            # Single numeric column - histogram
            return "histogram"
        elif len(categorical_cols) >= 1:
            # Categorical data - bar chart of counts
            return "bar"
        else:
            # Default fallback
            return "bar"
    
    def _create_bar_chart(self, df: pd.DataFrame) -> Any:
        """Create a bar chart from the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        
        if self.style == "plotly":
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Standard bar chart with categories
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                fig = px.bar(
                    df, 
                    x=x_col, 
                    y=y_col,
                    title=f"{y_col} by {x_col}",
                    labels={x_col: x_col.replace('_', ' ').title(), 
                           y_col: y_col.replace('_', ' ').title()}
                )
                fig.update_layout(xaxis_tickangle=-45)
                return fig
            
            elif len(categorical_cols) > 0:
                # Count of categorical values
                x_col = categorical_cols[0]
                value_counts = df[x_col].value_counts()
                
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Count of {x_col}",
                    labels={'x': x_col.replace('_', ' ').title(), 'y': 'Count'}
                )
                return fig
            
            else:
                # Numeric columns only - show first few columns
                cols_to_plot = numeric_cols[:3]  # Limit to 3 columns
                fig = go.Figure()
                
                for col in cols_to_plot:
                    fig.add_trace(go.Bar(
                        x=df.index,
                        y=df[col],
                        name=col.replace('_', ' ').title()
                    ))
                
                fig.update_layout(title="Numeric Values by Row")
                return fig
        
        else:  # matplotlib
            plt.figure(figsize=(10, 6))
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                plt.bar(df[x_col], df[y_col])
                plt.xlabel(x_col.replace('_', ' ').title())
                plt.ylabel(y_col.replace('_', ' ').title())
                plt.title(f"{y_col} by {x_col}")
                plt.xticks(rotation=45)
            
            elif len(categorical_cols) > 0:
                x_col = categorical_cols[0]
                value_counts = df[x_col].value_counts()
                
                plt.bar(value_counts.index, value_counts.values)
                plt.xlabel(x_col.replace('_', ' ').title())
                plt.ylabel('Count')
                plt.title(f"Count of {x_col}")
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            return self._matplotlib_to_base64()
    
    def _create_line_chart(self, df: pd.DataFrame) -> Any:
        """Create a line chart from the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.style == "plotly":
            if len(numeric_cols) >= 2:
                # Use first column as x, others as y
                x_col = numeric_cols[0]
                y_cols = numeric_cols[1:4]  # Limit to 3 lines
                
                fig = go.Figure()
                for y_col in y_cols:
                    fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='lines+markers',
                        name=y_col.replace('_', ' ').title()
                    ))
                
                fig.update_layout(
                    title="Line Chart",
                    xaxis_title=x_col.replace('_', ' ').title(),
                    yaxis_title="Values"
                )
                return fig
            
            else:
                # Single numeric column - use index as x
                y_col = numeric_cols[0]
                fig = px.line(
                    df,
                    x=df.index,
                    y=y_col,
                    title=f"{y_col} Over Rows",
                    labels={'x': 'Row Index', 'y': y_col.replace('_', ' ').title()}
                )
                return fig
        
        else:  # matplotlib
            plt.figure(figsize=(10, 6))
            
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_cols = numeric_cols[1:4]
                
                for y_col in y_cols:
                    plt.plot(df[x_col], df[y_col], marker='o', label=y_col.replace('_', ' ').title())
                
                plt.xlabel(x_col.replace('_', ' ').title())
                plt.ylabel('Values')
                plt.title('Line Chart')
                plt.legend()
            
            else:
                y_col = numeric_cols[0]
                plt.plot(df.index, df[y_col], marker='o')
                plt.xlabel('Row Index')
                plt.ylabel(y_col.replace('_', ' ').title())
                plt.title(f"{y_col} Over Rows")
            
            plt.tight_layout()
            return self._matplotlib_to_base64()
    
    def _create_pie_chart(self, df: pd.DataFrame) -> Any:
        """Create a pie chart from the data."""
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if self.style == "plotly":
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Pie chart with values
                labels_col = categorical_cols[0]
                values_col = numeric_cols[0]
                
                fig = px.pie(
                    df,
                    names=labels_col,
                    values=values_col,
                    title=f"{values_col} by {labels_col}"
                )
                return fig
            
            elif len(categorical_cols) > 0:
                # Pie chart of counts
                labels_col = categorical_cols[0]
                value_counts = df[labels_col].value_counts()
                
                fig = px.pie(
                    names=value_counts.index,
                    values=value_counts.values,
                    title=f"Distribution of {labels_col}"
                )
                return fig
            
            else:
                # Numeric columns - show distribution of first column
                col = numeric_cols[0]
                # Create bins for numeric data
                bins = pd.cut(df[col], bins=5)
                bin_counts = bins.value_counts()
                
                fig = px.pie(
                    names=[str(x) for x in bin_counts.index],
                    values=bin_counts.values,
                    title=f"Distribution of {col}"
                )
                return fig
        
        else:  # matplotlib
            plt.figure(figsize=(8, 8))
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                labels_col = categorical_cols[0]
                values_col = numeric_cols[0]
                
                plt.pie(df[values_col], labels=df[labels_col], autopct='%1.1f%%')
                plt.title(f"{values_col} by {labels_col}")
            
            elif len(categorical_cols) > 0:
                labels_col = categorical_cols[0]
                value_counts = df[labels_col].value_counts()
                
                plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                plt.title(f"Distribution of {labels_col}")
            
            plt.tight_layout()
            return self._matplotlib_to_base64()
    
    def _create_scatter_chart(self, df: pd.DataFrame) -> Any:
        """Create a scatter plot from the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            # Not enough numeric columns for scatter plot
            return self._create_bar_chart(df)
        
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]
        
        if self.style == "plotly":
            # Check if there's a categorical column for color coding
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns
            color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
            
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col}",
                labels={
                    x_col: x_col.replace('_', ' ').title(),
                    y_col: y_col.replace('_', ' ').title()
                }
            )
            return fig
        
        else:  # matplotlib
            plt.figure(figsize=(10, 6))
            
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns
            if len(categorical_cols) > 0:
                # Color by category
                categories = df[categorical_cols[0]].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                
                for i, category in enumerate(categories):
                    mask = df[categorical_cols[0]] == category
                    plt.scatter(df[mask][x_col], df[mask][y_col], 
                              c=[colors[i]], label=category, alpha=0.7)
                
                plt.legend()
            else:
                plt.scatter(df[x_col], df[y_col], alpha=0.7)
            
            plt.xlabel(x_col.replace('_', ' ').title())
            plt.ylabel(y_col.replace('_', ' ').title())
            plt.title(f"{y_col} vs {x_col}")
            plt.tight_layout()
            return self._matplotlib_to_base64()
    
    def _create_histogram(self, df: pd.DataFrame) -> Any:
        """Create a histogram from the data."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return self._create_bar_chart(df)
        
        col = numeric_cols[0]
        
        if self.style == "plotly":
            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                labels={col: col.replace('_', ' ').title()}
            )
            return fig
        
        else:  # matplotlib
            plt.figure(figsize=(10, 6))
            plt.hist(df[col], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel(col.replace('_', ' ').title())
            plt.ylabel('Frequency')
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            return self._matplotlib_to_base64()
    
    def _matplotlib_to_base64(self) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary information about the data used for charting."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'string']).columns),
            'column_names': list(df.columns)
        }
    
    def get_available_chart_types(self, data: List[Dict[str, Any]]) -> List[str]:
        """
        Get list of suitable chart types for the given data.
        
        Args:
            data: Query results data
            
        Returns:
            List of recommended chart types
        """
        if not data:
            return []
        
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        
        chart_types = []
        
        # Always available
        chart_types.append("bar")
        
        # Line chart for numeric data
        if len(numeric_cols) >= 1:
            chart_types.append("line")
            chart_types.append("histogram")
        
        # Pie chart for categorical data
        if len(categorical_cols) >= 1:
            chart_types.append("pie")
        
        # Scatter plot for multiple numeric columns
        if len(numeric_cols) >= 2:
            chart_types.append("scatter")
        
        return chart_types


# Example usage and testing
if __name__ == "__main__":
    # Test the chart generator
    generator = ChartGenerator(style="plotly")
    
    # Sample data
    sample_data = [
        {'region': 'North', 'sales': 1500, 'customers': 25, 'avg_order': 60.0},
        {'region': 'South', 'sales': 1200, 'customers': 20, 'avg_order': 60.0},
        {'region': 'East', 'sales': 1800, 'customers': 30, 'avg_order': 60.0},
        {'region': 'West', 'sales': 1000, 'customers': 15, 'avg_order': 66.7}
    ]
    
    print("Testing Chart Generator:")
    print("=" * 50)
    
    # Test different chart types
    chart_types = ['auto', 'bar', 'pie', 'scatter']
    
    for chart_type in chart_types:
        print(f"\nTesting {chart_type} chart:")
        result = generator.generate_chart(sample_data, chart_type)
        
        if result['success']:
            print(f"✓ {chart_type} chart generated successfully")
            print(f"  Actual chart type: {result['chart_type']}")
            print(f"  Data summary: {result['data_summary']}")
        else:
            print(f"✗ Failed to generate {chart_type} chart: {result['error']}")
    
    # Test available chart types
    available_types = generator.get_available_chart_types(sample_data)
    print(f"\nAvailable chart types for this data: {available_types}")