use pyo3::prelude::*;

// Module declarations
mod analyzer;
mod parser;
mod types;
mod utils;

// Import types for Python bindings
use analyzer::{AdvancedProjectAnalyzer, ProjectAnalyzer};
use parser::{extract_symbols_from_content, parse_move_content};
use types::{FunctionInfo, ModuleInfo, StructInfo, SymbolExtractor};

/// Parse Move source code and return AST as string representation
#[pyfunction]
fn parse(content: &str) -> PyResult<String> {
    parse_move_content(content)
}

/// Extract symbols (functions, structs, modules) from Move source code
#[pyfunction]
fn extract_symbols(content: &str) -> PyResult<SymbolExtractor> {
    extract_symbols_from_content(content)
}

/// Create advanced project analyzer by scanning a directory for Move files
#[pyfunction]
fn analyze_project_advanced(dir_path: &str) -> PyResult<AdvancedProjectAnalyzer> {
    let mut analyzer = AdvancedProjectAnalyzer::new();
    analyzer.add_directory(dir_path)?;
    Ok(analyzer)
}

/// Create advanced project analyzer from specific Move files
#[pyfunction]
fn analyze_files_advanced(file_paths: Vec<String>) -> PyResult<AdvancedProjectAnalyzer> {
    let mut analyzer = AdvancedProjectAnalyzer::new();

    for file_path in file_paths {
        analyzer.add_file(&file_path)?;
    }

    analyzer.build_dependency_graph();
    Ok(analyzer)
}

/// Search for symbols with exact and fuzzy matching
/// Supports queries like: "function_name" or "module::function_name"
/// Returns source code and file paths for matching symbols
#[pyfunction]
fn symbol_finder(analyzer: &AdvancedProjectAnalyzer, query: &str) -> PyResult<Vec<String>> {
    analyzer.search_symbols(query)
}

/// Create basic project analyzer by scanning a directory for Move files
#[pyfunction]
fn analyze_project(dir_path: &str) -> PyResult<ProjectAnalyzer> {
    let mut analyzer = ProjectAnalyzer::new();
    analyzer.add_directory(dir_path)?;
    Ok(analyzer)
}

/// Create basic project analyzer from specific Move files
#[pyfunction]
fn analyze_files(file_paths: Vec<String>) -> PyResult<ProjectAnalyzer> {
    let mut analyzer = ProjectAnalyzer::new();

    for file_path in file_paths {
        analyzer.add_file(&file_path)?;
    }

    analyzer.build_dependency_graph();
    Ok(analyzer)
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_move_analyer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Basic functions
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(extract_symbols, m)?)?;

    // Project analysis functions
    m.add_function(wrap_pyfunction!(analyze_project, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_files, m)?)?;

    // Advanced analysis functions
    m.add_function(wrap_pyfunction!(analyze_project_advanced, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_files_advanced, m)?)?;

    // Query functions
    m.add_function(wrap_pyfunction!(symbol_finder, m)?)?;

    // Data classes
    m.add_class::<FunctionInfo>()?;
    m.add_class::<StructInfo>()?;
    m.add_class::<SymbolExtractor>()?;
    m.add_class::<ModuleInfo>()?;
    m.add_class::<ProjectAnalyzer>()?;
    m.add_class::<AdvancedProjectAnalyzer>()?;

    Ok(())
}
