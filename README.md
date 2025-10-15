# Move Analyzer

A Rust-based Move language analyzer with Python bindings for parsing and analyzing Move smart contracts.

## Features

- Parse Move source code and extract AST information
- Symbol extraction (functions, structs, modules)
- Project-wide analysis with dependency tracking
- Fuzzy search for symbols across modules
- Source code location tracking

## Installation

```bash
# Development build (recommended for development)
maturin develop

# Production build and install
maturin build --release
pip install target/wheels/*.whl
```

## Usage

### Python API

```python
import py_move_analyer

# Parse Move source code
result = py_move_analyer.parse(move_code)

# Extract symbols from source
symbols = py_move_analyer.extract_symbols(move_code)

# Analyze entire project
analyzer = py_move_analyer.analyze_project_advanced("/path/to/move/project")

# Search for symbols
results = py_move_analyer.symbol_finder(analyzer, "function_name")
```

### Basic Analysis

```python
# Create analyzer from directory
analyzer = py_move_analyer.analyze_project("/path/to/project")

# Get module information
modules = analyzer.get_module_names()
functions = analyzer.find_function("transfer")
structs = analyzer.find_struct("Coin")
```

## API Reference

### Core Functions

- `parse(content)` - Parse Move source code
- `extract_symbols(content)` - Extract symbols from source
- `analyze_project(dir_path)` - Basic project analysis
- `analyze_project_advanced(dir_path)` - Advanced analysis with fuzzy search
- `symbol_finder(analyzer, query)` - Search symbols with exact/fuzzy matching

### Data Types

- `FunctionInfo` - Function metadata and source code
- `StructInfo` - Struct definition and fields
- `ModuleInfo` - Module information with dependencies
- `SymbolExtractor` - Collection of extracted symbols

## Requirements

- Rust 1.88+
- Python 3.7+ (for Python bindings)
- Move compiler dependencies

## License

[Add your license here]