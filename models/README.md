# Models Directory

This directory contains model-related files and utilities for the NL-to-SQL Assistant.

## Model Storage

Models are automatically downloaded and cached in the `cache/` subdirectory when first used. The cache location can be configured via the `MODEL_CACHE_DIR` environment variable.

## Supported Models

### Primary Models
- **t5-small**: Fast, lightweight model good for prototyping
- **t5-base**: Larger model with better accuracy
- **google/flan-t5-small**: Instruction-tuned T5 model
- **google/flan-t5-base**: Larger instruction-tuned model

### Model Selection Guidelines

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| t5-small | ~60MB | Fast | Good | Development, testing |
| t5-base | ~220MB | Medium | Better | Production (small scale) |
| flan-t5-small | ~80MB | Fast | Good | Instruction following |
| flan-t5-base | ~250MB | Medium | Better | Production (better quality) |

## Configuration

Model settings are configured in `configs/model_config.yaml`:

```yaml
model:
  name: "t5-small"
  max_input_length: 512
  max_output_length: 256
  num_beams: 4
  temperature: 0.7
```

## Custom Models

To use a custom fine-tuned model:

1. Place your model files in the `cache/` directory
2. Update the model name in the configuration
3. Ensure the model follows the T5 format

## Model Download

Models are downloaded automatically on first use. To pre-download models:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# This will download and cache the model
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
```

## Memory Requirements

| Model | RAM (Inference) | VRAM (GPU) |
|-------|----------------|------------|
| t5-small | ~1GB | ~1GB |
| t5-base | ~2GB | ~2GB |
| flan-t5-small | ~1GB | ~1GB |
| flan-t5-base | ~2GB | ~2GB |

## Performance Tips

1. **Use GPU**: Enable CUDA for faster inference
2. **Batch Processing**: Process multiple queries together
3. **Model Caching**: Keep models loaded in memory for repeated use
4. **Quantization**: Use 8-bit or 16-bit precision for memory efficiency

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller model or reduce batch size
2. **Slow Downloads**: Check internet connection, models are large
3. **CUDA Errors**: Ensure compatible PyTorch and CUDA versions
4. **Permission Errors**: Check write permissions for cache directory

### Environment Variables

- `MODEL_CACHE_DIR`: Custom cache directory path
- `TRANSFORMERS_CACHE`: HuggingFace cache directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection