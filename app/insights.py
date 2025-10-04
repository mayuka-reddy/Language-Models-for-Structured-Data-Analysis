"""
Insights Generator Module
Generates basic insights and summaries from query results.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from loguru import logger


class InsightsGenerator:
    """
    Generates basic insights and summaries from SQL query results.
    
    Provides heuristic-based analysis including statistical summaries,
    trends, and key findings from tabular data.
    """
    
    def __init__(self):
        """Initialize insights generator."""
        logger.info("Insights Generator initialized")
    
    def generate_insights(self, data: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        """
        Generate insights from query results.
        
        Args:
            data: List of dictionaries representing query results
            query: Original SQL query (optional, for context)
            
        Returns:
            Dictionary containing various insights and summaries
        """
        if not data:
            return {
                'summary': 'No data returned from query',
                'row_count': 0,
                'insights': [],
                'statistics': {}
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(data)
        
        insights = {
            'summary': self._generate_summary(df, query),
            'row_count': len(df),
            'column_count': len(df.columns),
            'insights': self._generate_key_insights(df),
            'statistics': self._generate_statistics(df),
            'data_types': self._analyze_data_types(df)
        }
        
        return insights
    
    def _generate_summary(self, df: pd.DataFrame, query: str = "") -> str:
        """Generate a high-level summary of the results."""
        row_count = len(df)
        col_count = len(df.columns)
        
        # Basic summary
        summary_parts = [
            f"Query returned {row_count} rows and {col_count} columns."
        ]
        
        # Add query-specific insights
        if query.upper().strip().startswith('SELECT'):
            if 'GROUP BY' in query.upper():
                summary_parts.append("This appears to be an aggregation query.")
            if 'JOIN' in query.upper():
                summary_parts.append("This query involves multiple tables.")
            if 'ORDER BY' in query.upper():
                summary_parts.append("Results are sorted.")
        
        # Add data insights
        if row_count > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_parts.append(f"Found {len(numeric_cols)} numeric columns for analysis.")
        
        return " ".join(summary_parts)
    
    def _generate_key_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate key insights from the data."""
        insights = []
        
        if len(df) == 0:
            return ["No data to analyze"]
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_insights = self._analyze_numeric_column(df[col], col)
            insights.extend(col_insights)
        
        # Analyze categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_cols:
            col_insights = self._analyze_categorical_column(df[col], col)
            insights.extend(col_insights)
        
        # Overall data insights
        insights.extend(self._analyze_overall_patterns(df))
        
        return insights[:10]  # Limit to top 10 insights
    
    def _analyze_numeric_column(self, series: pd.Series, col_name: str) -> List[str]:
        """Analyze a numeric column and generate insights."""
        insights = []
        
        if series.empty or series.isna().all():
            return [f"{col_name}: No valid numeric data"]
        
        # Basic statistics
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std()
        min_val = series.min()
        max_val = series.max()
        
        # Generate insights
        if not pd.isna(mean_val):
            insights.append(f"{col_name}: Average is {mean_val:.2f}")
        
        if not pd.isna(min_val) and not pd.isna(max_val):
            range_val = max_val - min_val
            insights.append(f"{col_name}: Range from {min_val:.2f} to {max_val:.2f}")
        
        # Check for outliers (simple method)
        if not pd.isna(std_val) and std_val > 0:
            outliers = series[abs(series - mean_val) > 2 * std_val]
            if len(outliers) > 0:
                insights.append(f"{col_name}: Found {len(outliers)} potential outliers")
        
        # Check distribution
        if abs(mean_val - median_val) > 0.1 * std_val:
            if mean_val > median_val:
                insights.append(f"{col_name}: Data appears right-skewed")
            else:
                insights.append(f"{col_name}: Data appears left-skewed")
        
        return insights
    
    def _analyze_categorical_column(self, series: pd.Series, col_name: str) -> List[str]:
        """Analyze a categorical column and generate insights."""
        insights = []
        
        if series.empty:
            return [f"{col_name}: No data"]
        
        # Value counts
        value_counts = series.value_counts()
        unique_count = len(value_counts)
        total_count = len(series)
        
        insights.append(f"{col_name}: {unique_count} unique values")
        
        if unique_count > 0:
            # Most common value
            most_common = value_counts.index[0]
            most_common_count = value_counts.iloc[0]
            percentage = (most_common_count / total_count) * 100
            
            insights.append(f"{col_name}: Most common value is '{most_common}' ({percentage:.1f}%)")
            
            # Check for dominance
            if percentage > 80:
                insights.append(f"{col_name}: Highly dominated by one value")
            elif percentage < 20 and unique_count > 5:
                insights.append(f"{col_name}: Values are well distributed")
        
        return insights
    
    def _analyze_overall_patterns(self, df: pd.DataFrame) -> List[str]:
        """Analyze overall patterns in the dataset."""
        insights = []
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            insights.append(f"Dataset has {total_missing} missing values across all columns")
            
            # Identify columns with most missing values
            cols_with_missing = missing_counts[missing_counts > 0]
            if len(cols_with_missing) > 0:
                worst_col = cols_with_missing.index[0]
                worst_count = cols_with_missing.iloc[0]
                insights.append(f"Column '{worst_col}' has the most missing values ({worst_count})")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"Found {duplicate_count} duplicate rows")
        
        # Data size insights
        row_count = len(df)
        if row_count == 1:
            insights.append("Single row result - likely an aggregation or specific lookup")
        elif row_count < 10:
            insights.append("Small result set - good for detailed analysis")
        elif row_count > 1000:
            insights.append("Large result set - consider filtering for focused analysis")
        
        return insights
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary of the data."""
        stats = {}
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe()
            stats['numeric'] = numeric_stats.to_dict()
        
        # Categorical columns statistics
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(categorical_cols) > 0:
            categorical_stats = {}
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                categorical_stats[col] = {
                    'unique_count': len(value_counts),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0
                }
            stats['categorical'] = categorical_stats
        
        return stats
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze and return data types of columns."""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    def generate_recommendations(self, data: List[Dict[str, Any]], query: str = "") -> List[str]:
        """
        Generate recommendations for further analysis.
        
        Args:
            data: Query results
            query: Original SQL query
            
        Returns:
            List of recommendation strings
        """
        if not data:
            return ["No data to analyze - check your query conditions"]
        
        df = pd.DataFrame(data)
        recommendations = []
        
        # Query-based recommendations
        query_upper = query.upper()
        
        if 'LIMIT' not in query_upper and len(df) > 100:
            recommendations.append("Consider adding LIMIT clause for large result sets")
        
        if 'ORDER BY' not in query_upper and len(df) > 10:
            recommendations.append("Consider adding ORDER BY for consistent results")
        
        # Data-based recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            recommendations.append("Multiple numeric columns found - consider correlation analysis")
        
        if len(numeric_cols) > 0:
            recommendations.append("Numeric data available - consider aggregations (SUM, AVG, MAX)")
        
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns
        if len(categorical_cols) > 0:
            recommendations.append("Categorical data found - consider GROUP BY analysis")
        
        # Missing value recommendations
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            recommendations.append("Missing values detected - consider data cleaning")
        
        return recommendations[:5]  # Limit to top 5 recommendations


# Example usage and testing
if __name__ == "__main__":
    # Test the insights generator
    generator = InsightsGenerator()
    
    # Sample data
    sample_data = [
        {'customer_id': 1, 'name': 'John Doe', 'region': 'North', 'total_orders': 5, 'total_spent': 1500.50},
        {'customer_id': 2, 'name': 'Jane Smith', 'region': 'South', 'total_orders': 3, 'total_spent': 890.25},
        {'customer_id': 3, 'name': 'Bob Johnson', 'region': 'North', 'total_orders': 8, 'total_spent': 2100.75},
        {'customer_id': 4, 'name': 'Alice Brown', 'region': 'West', 'total_orders': 2, 'total_spent': 450.00},
        {'customer_id': 5, 'name': 'Charlie Wilson', 'region': 'North', 'total_orders': 6, 'total_spent': 1800.30}
    ]
    
    sample_query = "SELECT customer_id, name, region, COUNT(*) as total_orders, SUM(amount) as total_spent FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.customer_id, c.name, c.region"
    
    print("Testing Insights Generator:")
    print("=" * 50)
    
    # Generate insights
    insights = generator.generate_insights(sample_data, sample_query)
    
    print("Summary:")
    print(insights['summary'])
    print()
    
    print("Key Insights:")
    for i, insight in enumerate(insights['insights'], 1):
        print(f"{i}. {insight}")
    print()
    
    print("Recommendations:")
    recommendations = generator.generate_recommendations(sample_data, sample_query)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print()
    
    print("Statistics:")
    if 'numeric' in insights['statistics']:
        print("Numeric columns:")
        for col, stats in insights['statistics']['numeric'].items():
            print(f"  {col}: mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}")
    
    if 'categorical' in insights['statistics']:
        print("Categorical columns:")
        for col, stats in insights['statistics']['categorical'].items():
            print(f"  {col}: {stats['unique_count']} unique values, most common: {stats['most_common']}")