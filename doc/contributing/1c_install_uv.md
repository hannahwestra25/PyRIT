# Setting up PyRIT Development Environment with uv (Windows)

This guide covers setting up a PyRIT development environment using [uv](https://github.com/astral-sh/uv), a fast Python package installer and resolver, on Windows.

## Why uv?

- **Much faster** than pip (10-100x faster dependency resolution)
- **Simpler** than conda/mamba for pure Python projects
- **Native Windows support** - no WSL required
- **Automatic virtual environment management**
- **Compatible with existing pyproject.toml**

## Prerequisites

1. **Install uv**: Download from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) or use:
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Python 3.12**: uv will automatically download and use the correct Python version based on `.python-version`

## Setup Steps

### 1. Clone the Repository

```powershell
git clone https://github.com/Azure/PyRIT.git
cd PyRIT
```

### 2. Initialize uv Environment

The repository includes a `.python-version` file that pins Python 3.12. Run:

```powershell
uv sync --extra dev
```

This command will:
- Create a `.venv` directory with a virtual environment
- Install Python 3.12 if not already available
- Install PyRIT in editable mode
- Install all dependencies including dev tools (pytest, black, ruff, etc.)
- Create a `uv.lock` file for reproducible builds

### 3. Verify Installation

```powershell
uv run python -c "import pyrit; pyrit.show_versions()"
```

You should see output showing PyRIT version 0.10.0.dev0 and your Python dependencies.

## Usage

### Running Python Scripts

Use `uv run` to execute Python with the virtual environment:

```powershell
uv run python your_script.py
```

### Running Tests

```powershell
uv run pytest tests/
```

### Running Specific Test Files

```powershell
uv run pytest tests/unit/test_something.py
```

### Using PyRIT CLI Tools

```powershell
uv run pyrit_scan --help
uv run pyrit_shell
```

### Running Jupyter Notebooks

```powershell
uv run jupyter lab
```

### Installing Additional Extras

PyRIT has several optional dependency groups. Install them as needed:

```powershell
# For Hugging Face models
uv sync --extra huggingface

# For all extras
uv sync --extra all

# Multiple extras
uv sync --extra dev --extra playwright --extra gcg
```

## Development Workflow

### Adding New Dependencies

Edit `pyproject.toml` to add dependencies, then run:

```powershell
uv sync
```

### Updating Dependencies

```powershell
uv lock --upgrade
uv sync
```

### Running Code Formatters

```powershell
uv run black .
uv run ruff check --fix .
```

### Running Type Checker

```powershell
uv run mypy pyrit/
```

### Pre-commit Hooks

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```

## VS Code Integration

VS Code should automatically detect the `.venv` virtual environment. If not:

1. Press `Ctrl+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `.venv\Scripts\python.exe`

## Troubleshooting

### uv command not found

Make sure uv is in your PATH. Restart PowerShell after installation.

### Import errors

Ensure you're using `uv run python` or have activated the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Dependency conflicts

Try regenerating the lock file:

```powershell
Remove-Item uv.lock
uv sync --extra dev
```

### Module not found errors

PyRIT is installed in editable mode, so changes to the source code are immediately reflected. If you see import errors:

```powershell
uv sync --reinstall-package pyrit
```

## Advantages over Other Methods

| Feature | uv | conda/mamba | pip + venv | Docker/DevContainer |
|---------|----|--------------|-----------|--------------------|
| Setup time | ~2 min | ~10-15 min | ~15-20 min | ~20-30 min |
| Disk space | ~1 GB | ~3-5 GB | ~1.5 GB | ~5-10 GB |
| Windows native | ✅ | ✅ | ✅ | ❌ (needs WSL2) |
| Speed | ⚡⚡⚡ | ⚡ | ⚡⚡ | ⚡ |
| Lock file | ✅ | ✅ | ❌ | ✅ |
| Isolation | ✅ | ✅ | ✅ | ✅✅ |

## Additional Resources

- [uv Documentation](https://github.com/astral-sh/uv)
- [PyRIT Contributing Guide](README.md)
- [Running Tests Guide](5_running_tests.md)
