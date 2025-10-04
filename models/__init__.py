"""
Models package for NL-to-SQL Assistant
Handles model loading and management utilities.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def get_model_cache_dir() -> str:
    """Get the model cache directory path."""
    cache_dir = os.environ.get('MODEL_CACHE_DIR', './models/cache')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_model_config(config_path: str = "configs/model_config.yaml") -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing model configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        # Return default configuration
        return {
            'model': {
                'name': 't5-small',
                'max_input_length': 512,
                'max_output_length': 256,
                'num_beams': 4,
                'temperature': 0.7
            }
        }


def get_available_models() -> Dict[str, str]:
    """
    Get list of available models.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        't5-small': 'T5 Small - Fast inference, good for prototyping',
        't5-base': 'T5 Base - Better accuracy, slower inference',
        'google/flan-t5-small': 'FLAN-T5 Small - Instruction-tuned T5',
        'google/flan-t5-base': 'FLAN-T5 Base - Larger instruction-tuned model'
    }


def validate_model_name(model_name: str) -> bool:
    """
    Validate if model name is supported.
    
    Args:
        model_name: Name of the model to validate
        
    Returns:
        True if model is supported, False otherwise
    """
    available_models = get_available_models()
    return model_name in available_models


__all__ = [
    'get_model_cache_dir',
    'load_model_config', 
    'get_available_models',
    'validate_model_name'
]