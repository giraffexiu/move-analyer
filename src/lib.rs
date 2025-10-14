use pyo3::prelude::*;

// Import all modules
mod types;
mod utils;
mod parser;
mod analyzer;

// Re-export types for Python bindings
use types::{FunctionInfo, StructInfo, ModuleInfo, SymbolExtractor};
use analyzer::{ProjectAnalyzer, AdvancedProjectAnalyzer};
use parser::{parse_move_content, extract_symbols_from_content};

/// Parses Move source code and returns the result as a string.
#[pyfunction]
fn parse(content: &str) -> PyResult<String> {
    parse_move_content(content)
}

/// Extract symbols from Move source code content
#[pyfunction]
fn extract_symbols(content: &str) -> PyResult<SymbolExtractor> {
    extract_symbols_from_content(content)
}

/// Create an advanced project analyzer from a directory
#[pyfunction]
fn analyze_project_advanced(dir_path: &str) -> PyResult<AdvancedProjectAnalyzer> {
    let mut analyzer = AdvancedProjectAnalyzer::new();
    analyzer.add_directory(dir_path)?;
    Ok(analyzer)
}

/// Create an advanced project analyzer from multiple files
#[pyfunction]
fn analyze_files_advanced(file_paths: Vec<String>) -> PyResult<AdvancedProjectAnalyzer> {
    let mut analyzer = AdvancedProjectAnalyzer::new();
    
    for file_path in file_paths {
        analyzer.add_file(&file_path)?;
    }
    
    analyzer.build_dependency_graph();
    Ok(analyzer)
}

/// Universal symbol finder for functions and structs
/// Supports: symbol_name, module::symbol_name
/// Automatically tries exact match first, then fuzzy match if no exact results found
#[pyfunction]
fn symbol_finder(
    analyzer: &AdvancedProjectAnalyzer,
    query: &str
) -> PyResult<Vec<String>> {
    analyzer.search_symbols(query)
}

/// Create a basic project analyzer from a directory
#[pyfunction]
fn analyze_project(dir_path: &str) -> PyResult<ProjectAnalyzer> {
    let mut analyzer = ProjectAnalyzer::new();
    analyzer.add_directory(dir_path)?;
    Ok(analyzer)
}

/// Create a project analyzer from multiple files
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