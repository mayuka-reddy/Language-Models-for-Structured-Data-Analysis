"""
Centralized configuration management for ML model training and evaluation.
Handles loading, validation, and management of configuration files.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import os

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model settings."""
    name: str = "t5-small"
    tokenizer_name: Optional[str] = None
    max_input_length: int = 512
    max_target_length: int = 256
    cache_dir: str = "models/cache"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    output_dir: str = "models/training/model_checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    fp16: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_data_path: str = "models/training/training_data/train_pairs.json"
    eval_data_path: str = "models/training/training_data/eval_pairs.json"
    test_data_path: str = "models/training/training_data/test_pairs.json"
    augmentation_factor: float = 1.5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    metrics_to_calculate: list = None
    db_path: Optional[str] = "data/sample_retail.db"
    output_dir: str = "models/evaluation/results"
    visualization_output: str = "models/evaluation/plots"
    
    def __post_init__(self):
        if self.metrics_to_calculate is None:
            self.metrics_to_calculate = [
                "execution_accuracy",
                "exact_match_accuracy", 
                "schema_compliance_rate",
                "bleu_score",
                "response_time"
            ]


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    wandb_project: str = "nl2sql-performance-demo"
    wandb_run_name: Optional[str] = None
    experiment_name: str = "ml-performance-demo"
    track_metrics: bool = True
    save_predictions: bool = True
    save_models: bool = True


@dataclass
class TechniqueConfig:
    """Configuration for different ML techniques."""
    enabled_techniques: list = None
    prompting_strategies: Dict[str, Any] = None
    rag_config: Dict[str, Any] = None
    sft_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.enabled_techniques is None:
            self.enabled_techniques = [
                "chain_of_thought",
                "few_shot", 
                "self_consistency",
                "least_to_most",
                "rag_pipeline",
                "supervised_fine_tuning"
            ]
        
        if self.prompting_strategies is None:
            self.prompting_strategies = {
                "chain_of_thought": {
                    "template_path": "prompts/templates/chain_of_thought.yaml",
                    "max_reasoning_steps": 5
                },
                "few_shot": {
                    "template_path": "prompts/templates/few_shot.yaml",
                    "num_examples": 3
                },
                "self_consistency": {
                    "num_samples": 3,
                    "voting_strategy": "majority"
                },
                "least_to_most": {
                    "max_decomposition_depth": 3
                }
            }
        
        if self.rag_config is None:
            self.rag_config = {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "top_k_retrieval": 5,
                "hybrid_alpha": 0.5,
                "index_cache_dir": "models/techniques/rag_indices"
            }
        
        if self.sft_config is None:
            self.sft_config = {
                "base_model": "t5-small",
                "fine_tune_layers": "all",
                "checkpoint_dir": "models/training/model_checkpoints"
            }


@dataclass
class MLPerformanceDemoConfig:
    """Main configuration class combining all sub-configurations."""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    evaluation: EvaluationConfig = None
    experiment: ExperimentConfig = None
    techniques: TechniqueConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()
        if self.techniques is None:
            self.techniques = TechniqueConfig()


class ConfigManager:
    """
    Centralized configuration manager for the ML performance demo system.
    
    Handles loading, validation, merging, and saving of configuration files
    across different components of the system.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration file paths
        self.default_config_path = self.config_dir / "ml_demo_config.yaml"
        self.user_config_path = self.config_dir / "user_config.yaml"
        
        # Environment variable prefix
        self.env_prefix = "ML_DEMO_"
    
    def load_config(self, config_path: Optional[str] = None) -> MLPerformanceDemoConfig:
        """
        Load configuration from file with environment variable overrides.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            Complete configuration object
        """
        # Start with default configuration
        config = MLPerformanceDemoConfig()
        
        # Load from file if specified or if default exists
        file_config = {}
        
        if config_path:
            file_config = self._load_config_file(config_path)
        elif self.default_config_path.exists():
            file_config = self._load_config_file(str(self.default_config_path))
        
        # Merge file configuration
        if file_config:
            config = self._merge_config(config, file_config)
        
        # Apply environment variable overrides
        config = self._apply_env_overrides(config)
        
        # Validate configuration
        self._validate_config(config)
        
        logger.info("Configuration loaded successfully")
        return config
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        
        file_path = Path(config_path)
        
        if not file_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    logger.error(f"Unsupported config file format: {file_path.suffix}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}
    
    def _merge_config(self, base_config: MLPerformanceDemoConfig, file_config: Dict[str, Any]) -> MLPerformanceDemoConfig:
        """Merge file configuration into base configuration."""
        
        # Convert base config to dict for easier merging
        config_dict = asdict(base_config)
        
        # Deep merge file config
        merged_dict = self._deep_merge_dicts(config_dict, file_config)
        
        # Convert back to config object
        try:
            return MLPerformanceDemoConfig(
                model=ModelConfig(**merged_dict.get('model', {})),
                training=TrainingConfig(**merged_dict.get('training', {})),
                data=DataConfig(**merged_dict.get('data', {})),
                evaluation=EvaluationConfig(**merged_dict.get('evaluation', {})),
                experiment=ExperimentConfig(**merged_dict.get('experiment', {})),
                techniques=TechniqueConfig(**merged_dict.get('techniques', {}))
            )
        except Exception as e:
            logger.error(f"Error merging configuration: {e}")
            return base_config
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: MLPerformanceDemoConfig) -> MLPerformanceDemoConfig:
        """Apply environment variable overrides to configuration."""
        
        # Define environment variable mappings
        env_mappings = {
            f"{self.env_prefix}MODEL_NAME": ("model", "name"),
            f"{self.env_prefix}LEARNING_RATE": ("training", "learning_rate"),
            f"{self.env_prefix}BATCH_SIZE": ("training", "per_device_train_batch_size"),
            f"{self.env_prefix}NUM_EPOCHS": ("training", "num_train_epochs"),
            f"{self.env_prefix}OUTPUT_DIR": ("training", "output_dir"),
            f"{self.env_prefix}WANDB_PROJECT": ("experiment", "wandb_project"),
            f"{self.env_prefix}DB_PATH": ("evaluation", "db_path"),
        }
        
        config_dict = asdict(config)
        
        for env_var, (section, key) in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if key in ["learning_rate"]:
                    value = float(value)
                elif key in ["per_device_train_batch_size", "num_train_epochs"]:
                    value = int(value)
                elif key in ["fp16", "track_metrics"]:
                    value = value.lower() in ['true', '1', 'yes']
                
                # Apply override
                if section in config_dict:
                    config_dict[section][key] = value
                    logger.info(f"Applied environment override: {env_var} = {value}")
        
        # Reconstruct config object
        try:
            return MLPerformanceDemoConfig(
                model=ModelConfig(**config_dict['model']),
                training=TrainingConfig(**config_dict['training']),
                data=DataConfig(**config_dict['data']),
                evaluation=EvaluationConfig(**config_dict['evaluation']),
                experiment=ExperimentConfig(**config_dict['experiment']),
                techniques=TechniqueConfig(**config_dict['techniques'])
            )
        except Exception as e:
            logger.error(f"Error applying environment overrides: {e}")
            return config
    
    def _validate_config(self, config: MLPerformanceDemoConfig) -> None:
        """Validate configuration values."""
        
        validation_errors = []
        
        # Validate model config
        if not config.model.name:
            validation_errors.append("Model name cannot be empty")
        
        if config.model.max_input_length <= 0:
            validation_errors.append("Max input length must be positive")
        
        # Validate training config
        if config.training.learning_rate <= 0:
            validation_errors.append("Learning rate must be positive")
        
        if config.training.num_train_epochs <= 0:
            validation_errors.append("Number of epochs must be positive")
        
        # Validate data config
        if not (0 < config.data.train_ratio < 1):
            validation_errors.append("Train ratio must be between 0 and 1")
        
        total_ratio = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            validation_errors.append("Data split ratios must sum to 1.0")
        
        # Validate technique config
        if not config.techniques.enabled_techniques:
            validation_errors.append("At least one technique must be enabled")
        
        # Log validation errors
        if validation_errors:
            for error in validation_errors:
                logger.error(f"Configuration validation error: {error}")
            raise ValueError(f"Configuration validation failed: {validation_errors}")
        
        logger.info("Configuration validation passed")
    
    def save_config(self, config: MLPerformanceDemoConfig, output_path: Optional[str] = None) -> str:
        """
        Save configuration to file.
        
        Args:
            config: Configuration object to save
            output_path: Optional output path (defaults to default config path)
            
        Returns:
            Path where configuration was saved
        """
        
        if output_path is None:
            output_path = str(self.default_config_path)
        
        # Convert config to dictionary
        config_dict = asdict(config)
        
        # Save as YAML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {output_path}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def create_default_config(self) -> str:
        """Create and save default configuration file."""
        
        default_config = MLPerformanceDemoConfig()
        return self.save_config(default_config, str(self.default_config_path))
    
    def get_technique_config(self, config: MLPerformanceDemoConfig, technique_name: str) -> Dict[str, Any]:
        """Get configuration for a specific technique."""
        
        technique_configs = {
            "chain_of_thought": config.techniques.prompting_strategies.get("chain_of_thought", {}),
            "few_shot": config.techniques.prompting_strategies.get("few_shot", {}),
            "self_consistency": config.techniques.prompting_strategies.get("self_consistency", {}),
            "least_to_most": config.techniques.prompting_strategies.get("least_to_most", {}),
            "rag_pipeline": config.techniques.rag_config,
            "supervised_fine_tuning": config.techniques.sft_config
        }
        
        return technique_configs.get(technique_name, {})
    
    def update_config_section(
        self,
        config: MLPerformanceDemoConfig,
        section: str,
        updates: Dict[str, Any]
    ) -> MLPerformanceDemoConfig:
        """Update a specific section of the configuration."""
        
        config_dict = asdict(config)
        
        if section in config_dict:
            config_dict[section].update(updates)
            logger.info(f"Updated {section} configuration: {updates}")
        else:
            logger.warning(f"Configuration section '{section}' not found")
        
        # Reconstruct and validate
        try:
            updated_config = MLPerformanceDemoConfig(
                model=ModelConfig(**config_dict['model']),
                training=TrainingConfig(**config_dict['training']),
                data=DataConfig(**config_dict['data']),
                evaluation=EvaluationConfig(**config_dict['evaluation']),
                experiment=ExperimentConfig(**config_dict['experiment']),
                techniques=TechniqueConfig(**config_dict['techniques'])
            )
            
            self._validate_config(updated_config)
            return updated_config
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return config


# Example usage and testing
if __name__ == "__main__":
    # Test the configuration manager
    config_manager = ConfigManager()
    
    print("Testing Configuration Manager:")
    print("=" * 50)
    
    # Create default configuration
    default_config_path = config_manager.create_default_config()
    print(f"Default configuration created at: {default_config_path}")
    
    # Load configuration
    config = config_manager.load_config()
    print(f"Loaded configuration with model: {config.model.name}")
    
    # Test technique-specific config
    cot_config = config_manager.get_technique_config(config, "chain_of_thought")
    print(f"Chain-of-Thought config: {cot_config}")
    
    # Test configuration update
    updated_config = config_manager.update_config_section(
        config, 
        "training", 
        {"learning_rate": 1e-4, "num_train_epochs": 5}
    )
    print(f"Updated learning rate: {updated_config.training.learning_rate}")
    
    print("\nConfiguration manager testing completed successfully!")