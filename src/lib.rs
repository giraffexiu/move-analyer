use pyo3::prelude::*;
use move_compiler::parser::syntax::parse_file_string;
use move_compiler::shared::{CompilationEnv, Flags, PackageConfig};
use move_compiler::diagnostics::warning_filters::WarningFiltersBuilder;
use move_compiler::editions::{Flavor, Edition};
use move_command_line_common::files::FileHash;


/// Parses Move source code and returns the result as a string.
#[pyfunction]
fn parse(content: &str) -> PyResult<String> {
    let file_hash = FileHash::new(content);
    let mut env = CompilationEnv::new(
        Flags::testing(),
        Default::default(),
        Default::default(),
        None,
        Default::default(),
        Some(PackageConfig {
            is_dependency: false,
            warning_filter: WarningFiltersBuilder::new_for_source(),
            flavor: Flavor::default(),
            edition: Edition::E2024_BETA,
        }),
        None,
    );
    let defs = parse_file_string(&mut env, file_hash, content, None);
    
    // Convert the result to a string representation
    match defs {
        Ok(parsed) => Ok(format!("{:?}", parsed)),
        Err(err) => Ok(format!("Parse error: {:?}", err)),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn py_move_analyer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    Ok(())
}