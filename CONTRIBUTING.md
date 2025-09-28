# Contributing to Value Analysis Tool

Thank you for your interest in contributing to the Value Analysis Tool! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/Value_Analysis.git
   cd Value_Analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 How to Contribute

### Reporting Issues

Before creating an issue, please:
1. Check if the issue already exists
2. Use the issue templates provided
3. Include as much detail as possible:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

### Suggesting Features

Feature requests are welcome! Please:
1. Check existing feature requests first
2. Describe the feature clearly
3. Explain the use case and benefits
4. Consider implementation complexity

### Code Contributions

#### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   # Run tests
   pytest
   
   # Run linting
   black .
   isort .
   flake8
   
   # Type checking
   mypy .
   ```

4. **Commit your changes**
   ```bash
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

#### Commit Message Format

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Maintenance tasks

Examples:
```
feat: add support for new LLM providers
fix: resolve timeout issue with Ollama models
docs: update installation instructions
```

## 🎯 Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write descriptive docstrings
- Keep functions focused and small
- Use meaningful variable names

### Testing

- Write unit tests for new functionality
- Maintain test coverage above 80%
- Test both success and failure cases
- Use descriptive test names

### Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update type hints for API changes
- Include examples for new features

### Security

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Validate all user inputs
- Follow secure coding practices

## 🏗️ Project Structure

```
Value_Analysis/
├── Value_Analysis.py          # Main analysis script
├── Value_AnalysisGUI.py       # GUI interface
├── llm_config.py             # LLM configuration
├── config_manager.py         # Configuration management
├── enhanced_pdf_extractor.py # PDF processing
├── perplexity_*.py           # Perplexity integration
├── requirements*.txt         # Dependencies
├── setup.py                  # Setup script
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # This file
├── LICENSE                   # MIT License
└── .github/                  # GitHub workflows
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_llm_config.py

# Run with verbose output
pytest -v
```

### Test Structure

- Unit tests in `tests/` directory
- Integration tests for major features
- Mock external API calls
- Test both GUI and CLI interfaces

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document complex algorithms
- Provide usage examples

### User Documentation

- Keep README.md up to date
- Include installation instructions
- Provide troubleshooting guides
- Add screenshots for GUI changes

## 🔧 Development Tools

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:
- Black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

### IDE Configuration

Recommended VS Code settings:
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true
}
```

## 🐛 Debugging

### Common Issues

1. **Import errors**: Check virtual environment activation
2. **API errors**: Verify API keys in .env file
3. **Memory issues**: Use smaller models or reduce batch sizes
4. **Timeout errors**: Check network connectivity and API limits

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📞 Getting Help

- Create an issue for bugs or questions
- Check existing documentation first
- Join discussions in GitHub Discussions
- Review closed issues for similar problems

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Value Analysis Tool! 🚀
