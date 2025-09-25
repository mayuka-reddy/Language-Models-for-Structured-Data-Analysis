# Contributing to Language Models for Structured Data Analysis

Thank you for your interest in contributing to our project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/your-username/Language-Models-for-Structured-Data-Analysis.git
cd Language-Models-for-Structured-Data-Analysis
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions small and focused

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_prompting.py

# Run with coverage
pytest --cov=src tests/
```

### Code Formatting
```bash
# Format code with black
black src/ tests/

# Check code style
flake8 src/ tests/

# Type checking
mypy src/
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Additional prompting strategies
- [ ] Performance optimizations
- [ ] More evaluation metrics
- [ ] Documentation improvements

### Medium Priority
- [ ] Support for more database types
- [ ] Web interface enhancements
- [ ] Additional fine-tuning methods
- [ ] Error handling improvements

### Low Priority
- [ ] Additional language support
- [ ] Mobile interface
- [ ] Advanced visualization features

## 📝 Pull Request Process

1. **Update Documentation**: Update README.md and other docs if needed
2. **Add Tests**: Add tests for new functionality
3. **Update CHANGELOG**: Document your changes
4. **Submit PR**: Create a pull request with a clear description

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## 🐛 Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

## 📚 Documentation

- Update docstrings for any new functions/classes
- Add examples for new features
- Update README.md if adding new major features
- Keep API documentation current

## 🏷️ Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Update documentation

## 📞 Getting Help

- Open an issue for questions
- Join our discussions for general questions
- Contact team members for specific guidance

Thank you for contributing! 🎉
