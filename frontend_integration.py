#!/usr/bin/env python3
"""
Frontend Integration Script
Loads trained model results and integrates with Gradio UI frontend.
"""

import pickle
import json
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import os

class ModelResultsIntegrator:
    """Integrates trained model results with Gradio frontend."""
    
    def __init__(self):
        self.results_loaded = False
        self.model_results = None
        self.load_model_results()
    
    def load_model_results(self):
        """Load trained model results from pickle file."""
        pickle_file = Path('model_outputs/model_results.pkl')
        json_file = Path('model_outputs/model_results.json')
        
        print("📦 Loading trained model results...")
        
        # Try to load pickle file first
        if pickle_file.exists():
            try:
                with open(pickle_file, 'rb') as f:
                    self.model_results = pickle.load(f)
                print(f"✅ Loaded results from {pickle_file}")
                self.results_loaded = True
            except Exception as e:
                print(f"❌ Error loading pickle file: {e}")
        
        # Fallback to JSON file
        elif json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    self.model_results = json.load(f)
                print(f"✅ Loaded results from {json_file}")
                self.results_loaded = True
            except Exception as e:
                print(f"❌ Error loading JSON file: {e}")
        
        else:
            print("❌ No trained model results found!")
            print("🔧 Please run the ML_Model_Training_Pipeline.ipynb notebook first")
            return False
        
        return True
    
    def display_console_results(self):
        """Display key results in console."""
        if not self.results_loaded:
            print("❌ No results to display")
            return
        
        print("\n" + "="*60)
        print("🎯 TRAINED MODEL PERFORMANCE RESULTS")
        print("="*60)
        
        # Model Performance
        model_perf = self.model_results['model_performance']
        best_strategy = model_perf['best_strategy']
        overall = model_perf['overall_summary']
        
        print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
        best_metrics = model_perf['strategy_results'][best_strategy]
        print(f"   📊 Success Rate: {best_metrics['success_rate']:.1%}")
        print(f"   🎯 BLEU Score: {best_metrics['avg_bleu_score']:.3f}")
        print(f"   ⚡ Execution Accuracy: {best_metrics['execution_accuracy']:.1%}")
        print(f"   🕒 Response Time: {best_metrics['avg_execution_time']:.4f}s")
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   🎯 Avg Success Rate: {overall['avg_success_rate']:.1%}")
        print(f"   📊 Avg BLEU Score: {overall['avg_bleu_score']:.3f}")
        print(f"   ⚡ Avg Execution Accuracy: {overall['avg_execution_accuracy']:.1%}")
        print(f"   🔢 Total Strategies: {overall['total_strategies']}")
        print(f"   📝 Total Questions: {overall['total_questions']}")
        
        # RAG Performance
        rag_perf = self.model_results['rag_performance']
        print(f"\n🔍 RAG PIPELINE PERFORMANCE:")
        print(f"   ⚡ Avg Retrieval Time: {rag_perf['avg_retrieval_time']:.4f}s")
        print(f"   🎯 Avg Relevance Score: {rag_perf['avg_relevance_score']:.3f}")
        print(f"   📋 Schema Cards: {rag_perf['total_schema_cards']}")
        
        # Training Info
        metadata = self.model_results['metadata']
        print(f"\n📅 TRAINING INFO:")
        print(f"   🕒 Completed: {metadata['timestamp']}")
        print(f"   ⏱️ Duration: {metadata['training_duration']:.2f}s")
        print(f"   📊 Version: {metadata['version']}")
        
        # Sample Questions
        questions = self.model_results['detailed_results']['question_by_question'][:3]
        print(f"\n📝 SAMPLE RESULTS:")
        for i, q in enumerate(questions, 1):
            print(f"   {i}. {q['question']}")
            best_result = q['strategy_results'].get(best_strategy, {})
            if best_result:
                print(f"      SQL: {best_result['generated_sql'][:60]}...")
                print(f"      Confidence: {best_result['confidence']:.3f}")
        
        print("\n" + "="*60)
    
    def create_frontend_data(self):
        """Create optimized data for frontend display."""
        if not self.results_loaded:
            return None
        
        # Prepare frontend-optimized data
        frontend_data = {
            'summary': {
                'best_strategy': self.model_results['model_performance']['best_strategy'],
                'total_questions': self.model_results['metadata']['total_questions'],
                'avg_accuracy': self.model_results['model_performance']['overall_summary']['avg_execution_accuracy'],
                'avg_bleu_score': self.model_results['model_performance']['overall_summary']['avg_bleu_score'],
                'training_time': self.model_results['metadata']['training_duration']
            },
            'strategy_performance': self.model_results['model_performance']['strategy_results'],
            'rag_performance': self.model_results['rag_performance'],
            'question_results': self.model_results['detailed_results']['question_by_question'],
            'timestamp': datetime.now().isoformat()
        }
        
        return frontend_data
    
    def save_frontend_data(self):
        """Save frontend-ready data."""
        frontend_data = self.create_frontend_data()
        if not frontend_data:
            return False
        
        # Save to demo_results directory for existing frontend
        output_dir = Path('demo_results')
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / 'demo_results.json'
        with open(output_file, 'w') as f:
            json.dump(frontend_data, f, indent=2, default=str)
        
        print(f"✅ Frontend data saved to: {output_file}")
        return True
    
    def prepare_ui_integration(self):
        """Prepare model results for Gradio UI integration."""
        if not self.results_loaded:
            return False
        
        # Create UI data directory
        ui_data_dir = Path('ui/data')
        ui_data_dir.mkdir(exist_ok=True)
        
        # Save model results for UI access
        ui_results_file = ui_data_dir / 'model_results.json'
        with open(ui_results_file, 'w') as f:
            json.dump(self.model_results, f, indent=2, default=str)
        
        print(f"✅ Model results prepared for UI: {ui_results_file}")
        return True
    
    def start_gradio_ui(self):
        """Start the Gradio UI frontend."""
        if not self.prepare_ui_integration():
            print("❌ Failed to prepare UI integration")
            return
        
        print("\n🌐 STARTING GRADIO UI FRONTEND")
        print("📊 Interface: http://localhost:7860")
        print("🎯 Features: Single Query, Strategy Comparison, Analytics")
        print("🛑 Press Ctrl+C to stop")
        
        try:
            # Change to UI directory and run the app
            ui_dir = Path('ui')
            if ui_dir.exists() and (ui_dir / 'app.py').exists():
                print(f"\n🚀 Launching Gradio app from {ui_dir}/app.py...")
                
                # Run the UI app
                result = subprocess.run([
                    sys.executable, 'app.py'
                ], cwd=ui_dir, check=True)
                
            else:
                print("❌ UI directory or app.py not found!")
                print("📁 Expected: ui/app.py")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error starting Gradio UI: {e}")
        except KeyboardInterrupt:
            print("\n👋 Gradio UI stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """Main execution function."""
    print("🚀 ML Model Results - Gradio UI Integration")
    print("="*50)
    
    # Initialize integrator
    integrator = ModelResultsIntegrator()
    
    if not integrator.results_loaded:
        print("\n❌ No trained model results found!")
        print("📋 Please run the training pipeline first:")
        print("   1. Open ML_Model_Training_Pipeline.ipynb")
        print("   2. Run all cells to train the model")
        print("   3. Then run this script again")
        return
    
    # Display console results
    integrator.display_console_results()
    
    # Ask user what to do
    print("\n🎯 Choose an option:")
    print("   1. Start Gradio UI (recommended)")
    print("   2. Just display console results")
    print("   3. Prepare UI data only")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🌐 Starting Gradio UI...")
            integrator.start_gradio_ui()
        elif choice == "2":
            print("\n✅ Console results displayed above")
        elif choice == "3":
            integrator.prepare_ui_integration()
            print("✅ UI data prepared")
        else:
            print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()