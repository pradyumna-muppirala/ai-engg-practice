# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Tasks

### Running Tests
- The repository includes a test suite for `math_operations.py` located at `tests/test_math_operations.py`.
- Run all tests with: `pytest tests/test_math_operations.py -v`
- To run a specific test class or method, use pytest selectors, e.g., `pytest tests/test_math_operations.py::TestAdd::test_positive_numbers -v`

### Executing Practice Scripts
- Most files in the repository are standalone Python exercises/scripts.
- Run a script from the repository root with: `python <path/to/script.py>`
- Many scripts include example usage or demo code in their `if __name__ == "__main__":` block.

### Linting & Syntax Checking
- No formal linter is configured in the repository.
- To check Python syntax for any file: `python -m py_compile <path/to/script.py>`
- For basic style checks, you may use `pyflakes` or `pylint` if installed in your environment.

### Working with Virtual Environments
- The repository contains `.venv` and `.venv-1` directories (likely created during practice).
- To activate an existing virtual environment: `source .venv/Scripts/activate` (PowerShell: `.venv\Scripts\Activate.ps1`)
- To create a fresh virtual environment: `python -m venv .venv` followed by activation and `pip install -r requirements.txt` if a requirements file is added.

## High-Level Code Architecture & Structure

### Repository Organization
This repository is a collection of independent AI engineering practice exercises, organized by topic into the following directory structure:

```
ai-engg-practice/
├── math/                          # Core math utilities and practice
│   ├── math_operations.py         # Basic arithmetic, number theory, factorial, gcd, etc.
│   └── math_practice/             # Additional math exercises
│       ├── Math-practice.py
│       └── Calculus-Practice.py
├── data_analysis/                 # Data manipulation and analysis
│   ├── pandas/                    # Pandas exercises
│   ├── numpy/                    # NumPy exercises
│   └── data_prep/                # Data preparation and EDA
├── visualization/                # Matplotlib and Seaborn plotting exercises
├── machine_learning/             # ML algorithms and exercises
│   ├── Linear_Regression.py
│   ├── LogisticRegression-exercises.py
│   ├── k-NN-Exercise.py
│   ├── HD_ML_Exercise1.py
│   ├── Heart-Disease_Prediction-DA-Modelling.py
│   ├── SL_mini_project.py
│   └── gradient_descent_example.py
├── utils/                        # Reusable utility modules
│   ├── StringOps.py              # String manipulation functions
│   └── RegEx-Samples.py          # Regex-based text processing
├── skills/                       # Claude skill documentation
│   ├── StringOps.md
│   └── RegEx-Samples.md
├── experiments/                  # Standalone practice scripts and exercises
│   ├── HelloWorld.py
│   ├── ai_agent.py
│   ├── Probability-*.py
│   ├── Statistics_*.py
│   ├── Python-lang-practice/
│   └── ...
├── tests/                        # Unit tests
│   └── test_math_operations.py
└── data/                         # Data files (CSV, etc.) used by practice scripts
    ├── sales_data.csv
    ├── titanic.csv
    ├── Heart_Disease_Prediction.csv
    └── employee*.csv
```

### Reusable Utility Modules
Two directories contain utility modules designed to be exposed as Claude skills:
1. **utils/StringOps.py** – Provides string manipulation functions (`reverse_string`, `to_uppercase`, `to_lowercase`, `count_vowels`, `is_palindrome`, `concat_strings`). See `skills/StringOps.md` for usage instructions.
2. **utils/RegEx-Samples.py** – Provides regex-based text processing functions (`clean_the_text`, `replace_emails`). See `skills/RegEx-Samples.md` for usage instructions.

To use these utilities in your own code or Claude skills, add the repository root to `sys.path` and import from the module (as shown in each SKILL.md).

### Data Files
- All CSV data files are stored in the `data/` directory.
- Scripts reference data files using paths like `data/filename.csv` (e.g., `pd.read_csv("data/sales_data.csv")`).
- The `.gitignore` excludes `*.csv` files to avoid committing generated data.

### Agentic Loop Example
- `experiments/ai_agent.py` demonstrates an agentic loop using an external LLM (via OpenRouter) with a local Bash tool (`run_bash`) for executing commands.
- It initializes an Anthropic client pointed at OpenRouter and runs a loop that can reason, call tools, and integrate results.
- Example usage: `python experiments/ai_agent.py` (after setting `OPENROUTER_API_KEY` in environment).

### Databricks Integration
- `databricks.yml` defines a Databricks Asset Bundle for this repository.
- Enables deployment and execution of the practice code within Databricks workspaces.
- See the file for bundle name (`ai-engg-practice`) and development target configuration.

### Key Files to Explore
- `tests/test_math_operations.py` – Example of how unit tests are structured for utility modules.
- `math/math_operations.py` – Core utility with arithmetic, number theory, and helper functions (used by the test suite).
- `experiments/HelloWorld.py` – Simple entry point demonstrating sorting algorithms and matrix operations.
- `utils/StringOps.py` / `utils/RegEx-Samples.py` – Implementation of the skill-exposed utilities.

## Notes for Claude Code
- When asked to modify or extend a utility, check for existing tests (like `tests/test_math_operations.py`) and consider adding test cases for new functionality.
- The repository encourages experimentation; most scripts are safe to run independently.
- If you encounter missing dependencies (e.g., `pandas`, `matplotlib`, `scikit-learn`), they may need to be installed in your active environment.
- The `.gitignore` file excludes compiled Python files, text/data outputs, and Databricks local artifacts to keep the repository clean.
- All subdirectories contain `__init__.py` files to support Python package imports.
