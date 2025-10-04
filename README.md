# 🚀 NL-to-SQL Assistant

Convert natural language questions into SQL queries with instant execution, insights, and visualizations.

## ✨ Features

- **Natural Language to SQL**: Convert questions like "Show me top customers" to SQL queries using T5 model
- **Instant Execution**: Run queries against a sample retail database (SQLite/DuckDB)
- **Smart Insights**: Get automatic analysis and recommendations from results
- **Interactive Charts**: Visualize data with auto-generated charts (bar, line, pie, scatter)
- **Model Comparison**: Evaluate different NL-to-SQL models with comprehensive metrics
- **Web Interface**: Clean, intuitive Gradio-based UI

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│  T5 Inference    │───▶│  SQL Executor   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Chart Generator │    │ Insights Engine  │    │ Sample Database │
│                 │    │                  │    │ (SQLite/DuckDB) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- Internet connection (for model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/nl2sql-assistant.git
cd nl2sql-assistant

# Install dependencies
pip install -r requirements.txt
pip install -r missing_deps.txt

# Start the web interface
python ui/gradio_app.py
```

Open http://localhost:7860 in your browser!

### Using Makefile

```bash
# Complete setup
make setup

# Start UI
make run-ui
```

## 💡 Usage Examples

### Basic Queries
- "Show me all customers"
- "What are the total sales?"
- "Find the top 5 products by price"

### Advanced Queries
- "Which region has the highest average order value?"
- "Show me customers who have placed more than 3 orders"
- "What's the monthly sales trend?"

### Sample Database Schema

The system includes a sample retail database with:

- **customers**: customer_id, name, email, region, signup_date
- **products**: product_id, name, category, price, stock_quantity  
- **orders**: order_id, customer_id, order_date, total_amount, status
- **order_items**: order_item_id, order_id, product_id, quantity, unit_price

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test modules
make test-inference
make test-executor
make test-metrics

# Generate coverage report
make coverage
```

## 📊 Model Evaluation

The system includes comprehensive evaluation metrics:

- **Execution Correctness**: Does the SQL produce correct results?
- **Exact Match**: Does the SQL exactly match the expected query?
- **Schema Compliance**: Does the SQL follow database schema rules?
- **BLEU Score**: How similar is the generated SQL to expected SQL?

Access model comparison through the "Model Comparison" tab in the UI.

## 🛠️ Development

### Code Structure

```
nl2sql-assistant/
├── app/                    # Core backend logic
│   ├── inference.py       # T5 model inference
│   ├── sql_executor.py    # Database query execution
│   ├── insights.py        # Result analysis
│   ├── charts.py          # Visualization generation
│   └── metrics.py         # Model evaluation
├── ui/                    # Gradio web interface
│   └── gradio_app.py      # Main UI application
├── configs/               # Configuration files
├── tests/                 # Unit tests
├── models/                # Model storage and utilities
└── notebooks/             # Jupyter demos (optional)
```

### Development Workflow

```bash
# Install development dependencies
make dev-install

# Format code
make format

# Lint code
make lint

# Run development checks
make dev
```

### Configuration

Edit `configs/model_config.yaml` to customize:
- Model selection (t5-small, t5-base, flan-t5-small, etc.)
- Generation parameters (beam size, temperature)
- Performance settings

Edit `configs/ui_config.yaml` to customize:
- UI appearance and behavior
- Chart settings
- Sample questions

## 🔧 Customization

### Adding New Models

1. Update `configs/model_config.yaml`:
```yaml
model:
  name: "your-custom-model"
  max_input_length: 512
  max_output_length: 256
```

2. Ensure model follows T5 format or implement custom adapter

### Custom Database

Replace the sample database by modifying `SQLExecutor` initialization:

```python
executor = SQLExecutor("path/to/your/database.db", "sqlite")
```

### Custom Insights

Extend `InsightsGenerator` class to add domain-specific analysis:

```python
class CustomInsightsGenerator(InsightsGenerator):
    def _analyze_business_metrics(self, df):
        # Your custom analysis logic
        pass
```

## 📈 Performance

### Model Performance
- **t5-small**: ~1s per query, good accuracy
- **t5-base**: ~2s per query, better accuracy
- **flan-t5-small**: ~1s per query, instruction-tuned

### System Requirements
- **Minimum**: 4GB RAM, CPU-only
- **Recommended**: 8GB RAM, GPU with 2GB VRAM
- **Storage**: ~2GB for models and data

### Optimization Tips
1. Use GPU for faster inference
2. Keep models loaded in memory
3. Use smaller models for development
4. Batch multiple queries together

## 🐛 Troubleshooting

### Common Issues

**"Model download failed"**
- Check internet connection
- Verify disk space (models are ~200MB each)
- Try different model in config

**"Database connection error"**
- Run `make create-data` to recreate sample database
- Check file permissions in data directory

**"Out of memory"**
- Use smaller model (t5-small instead of t5-base)
- Reduce batch size in config
- Close other applications

**"Slow inference"**
- Enable GPU if available
- Use smaller model for development
- Check system resources

### Getting Help

1. Check the logs in the terminal
2. Run `make test` to verify installation
3. Try with sample questions first
4. Check configuration files for typos

## 🚧 Roadmap

### Phase 1 (Current)
- ✅ Basic NL-to-SQL conversion
- ✅ Web interface
- ✅ Sample database
- ✅ Basic insights and charts
- ✅ Model evaluation metrics

### Phase 2 (Future)
- 🔄 Schema-aware RAG integration
- 🔄 Fine-tuned model training
- 🔄 Advanced prompting strategies
- 🔄 Multi-database support
- 🔄 Query optimization suggestions

### Phase 3 (Future)
- 🔄 Production deployment
- 🔄 API endpoints
- 🔄 User authentication
- 🔄 Query history and favorites
- 🔄 Advanced analytics dashboard


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Made with ❤️ for the data community**
