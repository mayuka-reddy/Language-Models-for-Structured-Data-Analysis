#!/usr/bin/env python3
"""
Direct script to run model metrics evaluation
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app.metrics import ModelEvaluator
import pandas as pd

def main():
    print("🚀 NL-to-SQL Model Metrics Evaluation")
    print("=" * 50)
    
    # Initialize evaluator
    print("📊 Initializing evaluator...")
    evaluator = ModelEvaluator()
    
    # Create dummy models for comparison
    print("🤖 Creating test models...")
    models = evaluator.create_dummy_models()
    print(f"   Created {len(models)} models: {list(models.keys())}")
    
    # Create test dataset
    print("📝 Creating test dataset...")
    test_data = evaluator.create_test_dataset()
    print(f"   Created {len(test_data)} test cases")
    
    # Run evaluation
    print("🔄 Running evaluation...")
    results_df = evaluator.evaluate_models(test_data, models)
    
    # Display results
    print("\n📈 EVALUATION RESULTS:")
    print("=" * 60)
    print(results_df.round(3).to_string(index=False))
    
    # Generate detailed report
    print("\n📋 DETAILED REPORT:")
    print("=" * 40)
    report = evaluator.generate_comparison_report(results_df)
    print(report)
    
    # Save results
    output_file = "model_metrics_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print("\n✅ Metrics evaluation completed!")

if __name__ == "__main__":
    main()