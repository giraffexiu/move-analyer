use pyo3::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};

/// Discover Move files in a directory recursively
pub fn discover_move_files(dir_path: &Path) -> PyResult<Vec<PathBuf>> {
    let mut move_files = Vec::new();

    if !dir_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("Directory not found: {}", dir_path.display()),
        ));
    }

    fn visit_dir(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
        if dir.is_dir() {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    // Skip hidden directories and common non-source directories
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if !name.starts_with('.')
                            && !["target", "build", "node_modules", "deps"].contains(&name)
                        {
                            visit_dir(&path, files)?;
                        }
                    }
                } else if path.extension().and_then(|s| s.to_str()) == Some("move") {
                    files.push(path);
                }
            }
        }
        Ok(())
    }

    visit_dir(dir_path, &mut move_files)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    Ok(move_files)
}

/// Helper function for recursive dependency collection
pub fn collect_dependencies_recursive(
    dependency_graph: &std::collections::HashMap<String, Vec<String>>,
    module_name: &str,
    visited: &mut std::collections::HashSet<String>,
    chain: &mut Vec<String>,
) {
    if visited.contains(module_name) {
        return; // Avoid cycles
    }

    visited.insert(module_name.to_string());

    if let Some(deps) = dependency_graph.get(module_name) {
        for dep in deps {
            if !chain.contains(dep) {
                chain.push(dep.clone());
                collect_dependencies_recursive(dependency_graph, dep, visited, chain);
            }
        }
    }
}
