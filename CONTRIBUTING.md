# Contributing to Sigray Machine Learning Platform

We welcome contributions to the Sigray Machine Learning Platform! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-platform.git
   cd ml-platform
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src --cov-report=html

   # Run specific test categories
   pytest tests/test_training_*.py
   pytest tests/test_inference_*.py
   ```

4. **Check Code Quality**
   ```bash
   # Format code
   black src/ tests/

   # Check linting
   flake8 src/ tests/

   # Type checking
   mypy src/
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python Style
- Follow PEP 8 style guide
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints for all public functions
- Write docstrings for all public classes and functions

### Docstring Format
```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
        RuntimeError: When operation fails
        
    Example:
        ```python
        result = example_function("test", 20)
        assert result is True
        ```
    """
    pass
```

### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn

# Local imports
from ..core.data_models import SliceStack
from ..core.config import TrainingConfig
from .utils import helper_function
```

## Testing Guidelines

### Test Structure
- Place tests in the `tests/` directory
- Mirror the source structure: `tests/test_module_name.py`
- Use descriptive test names: `test_function_name_with_condition`

### Test Categories
1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

### Writing Tests
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.module import YourClass


class TestYourClass:
    """Test suite for YourClass."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.instance = YourClass()
    
    def test_method_with_valid_input(self):
        """Test method with valid input."""
        result = self.instance.method("valid_input")
        assert result == expected_value
    
    def test_method_with_invalid_input(self):
        """Test method raises error with invalid input."""
        with pytest.raises(ValueError, match="Expected error message"):
            self.instance.method("invalid_input")
    
    @patch('src.module.external_dependency')
    def test_method_with_mock(self, mock_dependency):
        """Test method with mocked dependency."""
        mock_dependency.return_value = "mocked_result"
        result = self.instance.method_using_dependency()
        assert result == "expected_result"
        mock_dependency.assert_called_once()
```

## Documentation Guidelines

### Code Documentation
- Write clear, concise docstrings
- Include examples for complex functions
- Document all parameters and return values
- Explain any side effects or important behavior

### README Updates
- Update README.md for new features
- Add examples for new functionality
- Update installation instructions if needed
- Keep the feature list current

### API Documentation
- Document all public APIs
- Include usage examples
- Explain configuration options
- Document error conditions

## Commit Message Guidelines

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```
feat(training): add early stopping with configurable patience

Add early stopping mechanism to prevent overfitting during training.
The patience parameter can be configured in TrainingConfig.

Closes #123

fix(inference): handle GPU out of memory errors gracefully

When GPU runs out of memory during inference, automatically retry
with smaller batch size or fall back to CPU processing.

test(core): add comprehensive tests for SliceStack class

Add tests covering edge cases, error conditions, and performance
characteristics of the SliceStack data structure.
```

## Pull Request Guidelines

### Before Submitting
- [ ] All tests pass
- [ ] Code is properly formatted (Black)
- [ ] No linting errors (Flake8)
- [ ] Type checking passes (MyPy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Description Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
- [ ] All tests pass
```

## Issue Guidelines

### Bug Reports
Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Minimal code example if possible

### Feature Requests
Include:
- Clear description of the feature
- Use case and motivation
- Proposed API or interface
- Alternative solutions considered
- Implementation suggestions (if any)

## Code Review Process

### For Contributors
- Respond to review comments promptly
- Make requested changes in separate commits
- Ask for clarification if feedback is unclear
- Be open to suggestions and improvements

### For Reviewers
- Be constructive and specific in feedback
- Focus on code quality, correctness, and maintainability
- Suggest improvements rather than just pointing out problems
- Approve when ready, request changes when needed

## Release Process

### Version Numbering
We follow Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in setup.py
- [ ] Git tag created
- [ ] Release notes prepared

## Getting Help

### Communication Channels
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and discussions
- Email: support@sigray.com for private inquiries

### Resources
- [Project Documentation](https://github.com/sigray/ml-platform/wiki)
- [API Reference](https://github.com/sigray/ml-platform/wiki/API-Reference)
- [Examples](https://github.com/sigray/ml-platform/tree/main/examples)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the Sigray Machine Learning Platform!