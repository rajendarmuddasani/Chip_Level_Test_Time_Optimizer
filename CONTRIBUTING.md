# Contributing to Chip_Level_Test_Time_Optimizer

Thank you for your interest in contributing to this chip-level test-time optimization project.

## How to Contribute

### Reporting Issues
- Use the issue tracker to report bugs
- Provide detailed reproduction steps
- Include system information and dependencies

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Clearly describe the proposed feature
- Explain the use case and benefits

### Code Contributions

#### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/<your-fork>.git
cd <your-fork>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the tested development lock
python -m pip install -r requirements.txt
python -m pip check
```

#### Code Style
- Follow the existing repository style.
- Run the enforced Ruff gate with `make lint` or the equivalent command in `Makefile`.
- Avoid unrelated formatting churn in legacy utility modules.

#### Testing
```bash
# Full tests and coverage floor
python -m pytest tests -q --cov=benchmark --cov=deployment --cov=models --cov=evaluation --cov=preprocessing --cov-fail-under=60

# Canonical split, artifact, prediction, and claim replay
python scripts/validate_evidence.py

# Runtime dependency audit
python -m pip_audit --requirement requirements-runtime.txt
```

#### Pull Request Process
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request with a clear description

### Documentation
- Update README.md for user-facing changes
- Add docstrings to all functions and classes
- Include examples in docstrings

## Development Guidelines

### Commit Messages
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Example:
```
feat: Add MMD drift detection to ensemble

Implements Maximum Mean Discrepancy (MMD) calculation for detecting
distribution drift in production data. Triggers retraining when drift
exceeds threshold.

Closes #42
```

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No sensitive data or credentials
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

## Security

- Never commit credentials, API keys, or sensitive data
- Use environment variables for configuration
- Follow OWASP guidelines for data handling
- Report security issues privately to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue with the "question" label or contact the maintainers.

---

Thank you for contributing.
