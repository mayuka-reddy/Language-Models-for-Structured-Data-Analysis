"""
Supervised Fine-Tuning (SFT) script for NL-to-SQL models.
Owner: Mayuka Kothuru

Implements full parameter fine-tuning using HuggingFace transformers
with support for T5, CodeT5, and other seq2seq models.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import wandb
from loguru import logger


@dataclass
class SFTConfig:
    """Configuration for supervised fine-tuning."""
    
    # Model configuration
    model_name: str = "t5-small"
    tokenizer_name: Optional[str] = None
    max_input_length: int = 512
    max_target_length: int = 256
    
    # Training configuration
    output_dir: str = "models/training/model_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Data configuration
    train_data_path: str = "models/training/training_data/train_pairs.json"
    eval_data_path: str = "models/training/training_data/eval_pairs.json"
    
    # Logging and evaluation
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Experiment tracking
    wandb_project: str = "nl2sql-sft"
    wandb_run_name: Optional[str] = None
    
    # Hardware
    fp16: bool = True
    dataloader_num_workers: int = 4


class NL2SQLDataset(Dataset):
    """Dataset class for NL-to-SQL training pairs."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: AutoTokenizer, 
        max_input_length: int = 512,
        max_target_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} training examples from {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Construct input text
        input_text = self._format_input(item)
        target_text = item['sql']
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }
    
    def _format_input(self, item: Dict[str, Any]) -> str:
        """Format input text with question and schema context."""
        input_parts = [
            "Generate SQL for the following question:",
            f"Question: {item['question']}"
        ]
        
        # Add schema context if available
        if 'schema_context' in item:
            schema_info = item['schema_context']
            if 'relevant_tables' in schema_info:
                tables_str = ", ".join(schema_info['relevant_tables'])
                input_parts.append(f"Tables: {tables_str}")
            
            if 'relevant_columns' in schema_info:
                columns_str = ", ".join(schema_info['relevant_columns'])
                input_parts.append(f"Columns: {columns_str}")
        
        return " | ".join(input_parts)


class EnhancedSFTTrainer:
    """Enhanced supervised fine-tuning trainer."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        
        # Initialize tokenizer and model
        tokenizer_name = config.tokenizer_name or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize datasets
        self.train_dataset = None
        self.eval_dataset = None
        self._load_datasets()
        
        # Initialize trainer
        self.trainer = None
        self._setup_trainer()
    
    def _load_datasets(self) -> None:
        """Load training and evaluation datasets."""
        if os.path.exists(self.config.train_data_path):
            self.train_dataset = NL2SQLDataset(
                self.config.train_data_path,
                self.tokenizer,
                self.config.max_input_length,
                self.config.max_target_length
            )
        else:
            logger.warning(f"Training data not found at {self.config.train_data_path}")
        
        if os.path.exists(self.config.eval_data_path):
            self.eval_dataset = NL2SQLDataset(
                self.config.eval_data_path,
                self.tokenizer,
                self.config.max_input_length,
                self.config.max_target_length
            )
        else:
            logger.warning(f"Evaluation data not found at {self.config.eval_data_path}")
    
    def _setup_trainer(self) -> None:
        """Setup HuggingFace trainer."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to="wandb" if wandb.api.api_key else None,
            run_name=self.config.wandb_run_name
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
    
    def prepare_training_data(self) -> None:
        """Prepare and validate training data."""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create sample data if files don't exist
        if not os.path.exists(self.config.train_data_path):
            self._create_sample_data(self.config.train_data_path)
        if not os.path.exists(self.config.eval_data_path):
            self._create_sample_data(self.config.eval_data_path)
    
    def train_model(self) -> Dict[str, Any]:
        """Run training and return results."""
        logger.info("Starting supervised fine-tuning...")
        
        # Prepare data
        self.prepare_training_data()
        
        # Initialize wandb if available
        if wandb.api.api_key:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Train model
        train_result = self.trainer.train()
        
        # Save final model
        self.save_model(self.config.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.config.output_dir}")
        
        return {
            'train_loss': train_result.training_loss,
            'train_runtime': train_result.metrics.get('train_runtime', 0),
            'train_samples_per_second': train_result.metrics.get('train_samples_per_second', 0)
        }
    
    def save_model(self, path: str) -> bool:
        """Save model and tokenizer."""
        try:
            self.trainer.save_model(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Model saved successfully to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return {}
        
        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def plot_training_curves(self) -> None:
        """Plot training curves (placeholder for visualization)."""
        # This would implement actual plotting using matplotlib/wandb
        logger.info("Training curves would be plotted here")
    
    def _create_sample_data(self, output_path: str) -> None:
        """Create sample training data for testing."""
        sample_data = [
            {
                "question": "Which city has the highest number of customers?",
                "sql": "SELECT city, COUNT(*) as customer_count FROM customer GROUP BY city ORDER BY customer_count DESC LIMIT 1",
                "schema_context": {
                    "relevant_tables": ["customer"],
                    "relevant_columns": ["city", "customer_id"]
                }
            },
            {
                "question": "What was the total sales in August 2025?",
                "sql": "SELECT SUM(total_amount) as total_sales FROM orders WHERE order_date >= '2025-08-01' AND order_date < '2025-09-01'",
                "schema_context": {
                    "relevant_tables": ["orders"],
                    "relevant_columns": ["total_amount", "order_date"]
                }
            }
        ]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Sample data created at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for NL-to-SQL")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Model name")
    parser.add_argument("--output_dir", type=str, default="models/training/model_checkpoints", help="Output directory")
    parser.add_argument("--create_sample_data", action="store_true", help="Create sample training data")
    
    args = parser.parse_args()
    
    # Initialize config
    config = SFTConfig(
        model_name=args.model_name,
        output_dir=args.output_dir
    )
    
    # Create trainer and run training
    trainer = EnhancedSFTTrainer(config)
    
    if args.create_sample_data:
        trainer.prepare_training_data()
        return
    
    # Run training
    train_results = trainer.train_model()
    
    # Run evaluation
    eval_results = trainer.evaluate_model()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()