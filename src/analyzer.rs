use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::parser::parse_move_file;
use crate::types::{FunctionInfo, ModuleInfo, StructInfo};
use crate::utils::{collect_dependencies_recursive, discover_move_files};

/// Fuzzy string matching utilities for symbol search
pub struct FuzzyMatcher;

impl FuzzyMatcher {
    /// Calculate Damerau-Levenshtein distance between two strings
    /// Includes transposition operations for better typo handling
    pub fn damerau_levenshtein_distance(s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };

                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1, // deletion
                        matrix[i][j - 1] + 1, // insertion
                    ),
                    matrix[i - 1][j - 1] + cost, // substitution
                );

                // Transposition (Damerau extension)
                if i > 1
                    && j > 1
                    && s1_chars[i - 1] == s2_chars[j - 2]
                    && s1_chars[i - 2] == s2_chars[j - 1]
                {
                    matrix[i][j] = std::cmp::min(matrix[i][j], matrix[i - 2][j - 2] + cost);
                }
            }
        }

        matrix[len1][len2]
    }

    /// Calculate normalized similarity score (0.0 to 1.0, higher means more similar)
    /// Uses character count for proper Unicode support
    pub fn similarity_score(s1: &str, s2: &str) -> f64 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        let max_len = std::cmp::max(len1, len2);

        if max_len == 0 {
            return 1.0;
        }

        let distance = Self::damerau_levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Find fuzzy matches above the specified similarity threshold
    /// Returns candidates sorted by similarity score (highest first)
    pub fn find_fuzzy_matches(
        query: &str,
        candidates: &[String],
        threshold: f64,
    ) -> Vec<(String, f64)> {
        let mut matches = Vec::new();
        let query_lower = query.to_lowercase();

        for candidate in candidates {
            let candidate_lower = candidate.to_lowercase();
            let score = Self::similarity_score(&query_lower, &candidate_lower);

            if score >= threshold {
                matches.push((candidate.clone(), score));
            }
        }

        // Sort by similarity score (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
}

/// Advanced Move project analyzer with fuzzy search and symbol indexing
#[pyclass]
pub struct AdvancedProjectAnalyzer {
    #[pyo3(get)]
    pub modules: Vec<ModuleInfo>,
    #[pyo3(get)]
    pub global_functions: HashMap<String, Vec<FunctionInfo>>,
    #[pyo3(get)]
    pub global_structs: HashMap<String, Vec<StructInfo>>,
    #[pyo3(get)]
    pub dependency_graph: HashMap<String, Vec<String>>,
}

#[pymethods]
impl AdvancedProjectAnalyzer {
    #[new]
    pub fn new() -> Self {
        AdvancedProjectAnalyzer {
            modules: Vec::new(),
            global_functions: HashMap::new(),
            global_structs: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Parse and add a single Move file to the project
    pub fn add_file(&mut self, file_path: &str) -> PyResult<bool> {
        let path = Path::new(file_path);
        match parse_move_file(path)? {
            Some(module_info) => {
                self.add_module(module_info);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Recursively scan directory and parse all Move files
    pub fn add_directory(&mut self, dir_path: &str) -> PyResult<usize> {
        let path = Path::new(dir_path);
        let move_files = discover_move_files(path)?;
        let mut added_count = 0;

        for file_path in move_files {
            if let Some(module_info) = parse_move_file(&file_path)? {
                self.add_module(module_info);
                added_count += 1;
            }
        }

        self.build_dependency_graph();

        Ok(added_count)
    }

    /// Add module to project and update symbol indices
    pub fn add_module(&mut self, module_info: ModuleInfo) {
        // Update global indices
        for func in &module_info.functions {
            // Add to global functions map
            self.global_functions
                .entry(func.name.clone())
                .or_insert_with(Vec::new)
                .push(func.clone());
        }

        for struct_info in &module_info.structs {
            // Add to global structs map
            self.global_structs
                .entry(struct_info.name.clone())
                .or_insert_with(Vec::new)
                .push(struct_info.clone());
        }

        self.modules.push(module_info);
    }

    /// Build module dependency graph from parsed modules
    pub fn build_dependency_graph(&mut self) {
        self.dependency_graph.clear();
        for module in &self.modules {
            self.dependency_graph
                .insert(module.name.clone(), module.dependencies.clone());
        }
    }

    /// Search functions using fuzzy string matching
    pub fn fuzzy_search_functions(&self, query: &str, threshold: f64) -> Vec<(FunctionInfo, f64)> {
        let all_names: Vec<String> = self.global_functions.keys().cloned().collect();
        let matches = FuzzyMatcher::find_fuzzy_matches(query, &all_names, threshold);

        let mut results = Vec::new();
        for (name, score) in matches {
            if let Some(funcs) = self.global_functions.get(&name) {
                for func in funcs {
                    results.push((func.clone(), score));
                }
            }
        }

        results
    }

    /// Search structs using fuzzy string matching
    pub fn fuzzy_search_structs(&self, query: &str, threshold: f64) -> Vec<(StructInfo, f64)> {
        let all_names: Vec<String> = self.global_structs.keys().cloned().collect();
        let matches = FuzzyMatcher::find_fuzzy_matches(query, &all_names, threshold);

        let mut results = Vec::new();
        for (name, score) in matches {
            if let Some(structs) = self.global_structs.get(&name) {
                for struct_info in structs {
                    results.push((struct_info.clone(), score));
                }
            }
        }

        results
    }

    /// Search symbols with exact matching first, then fuzzy matching
    /// Supports queries: "symbol_name" or "module::symbol_name"
    /// Returns source code and file paths for matching symbols
    pub fn search_symbols(&self, query: &str) -> PyResult<Vec<String>> {
        let mut results = Vec::new();

        // Parse query to check if it contains module specification
        let (module_filter, symbol_name) = if query.contains("::") {
            let parts: Vec<&str> = query.split("::").collect();
            if parts.len() == 2 {
                (Some(parts[0].to_string()), parts[1].to_string())
            } else {
                (None, query.to_string())
            }
        } else {
            (None, query.to_string())
        };

        // Try exact matches first
        let mut found_exact = false;

        // Search functions
        if let Some(functions) = self.global_functions.get(&symbol_name) {
            for func in functions {
                if let Some(ref module_filter) = module_filter {
                    if func.module.contains(module_filter) {
                        results.push(format!(
                            "filepath: {}\nsourcecode:\n{}",
                            self.get_module_file_path(&func.module)
                                .unwrap_or_else(|| "unknown".to_string()),
                            func.source_code
                        ));
                        found_exact = true;
                    }
                } else {
                    results.push(format!(
                        "filepath: {}\nsourcecode:\n{}",
                        self.get_module_file_path(&func.module)
                            .unwrap_or_else(|| "unknown".to_string()),
                        func.source_code
                    ));
                    found_exact = true;
                }
            }
        }

        // Search structs
        if let Some(structs) = self.global_structs.get(&symbol_name) {
            for struct_info in structs {
                if let Some(ref module_filter) = module_filter {
                    if struct_info.module.contains(module_filter) {
                        results.push(format!(
                            "filepath: {}\nsourcecode:\n{}",
                            self.get_module_file_path(&struct_info.module)
                                .unwrap_or_else(|| "unknown".to_string()),
                            struct_info.source_code
                        ));
                        found_exact = true;
                    }
                } else {
                    results.push(format!(
                        "filepath: {}\nsourcecode:\n{}",
                        self.get_module_file_path(&struct_info.module)
                            .unwrap_or_else(|| "unknown".to_string()),
                        struct_info.source_code
                    ));
                    found_exact = true;
                }
            }
        }

        // If no exact matches found, try fuzzy matching
        if !found_exact {
            let threshold = 0.6; // Standard threshold for fuzzy matching

            // Fuzzy search functions
            let function_names: Vec<String> = self.global_functions.keys().cloned().collect();
            let fuzzy_func_matches =
                FuzzyMatcher::find_fuzzy_matches(&symbol_name, &function_names, threshold);

            for (name, _score) in fuzzy_func_matches {
                if let Some(functions) = self.global_functions.get(&name) {
                    for func in functions {
                        if let Some(ref module_filter) = module_filter {
                            if func.module.contains(module_filter) {
                                results.push(format!(
                                    "filepath: {}\nsourcecode:\n{}",
                                    self.get_module_file_path(&func.module)
                                        .unwrap_or_else(|| "unknown".to_string()),
                                    func.source_code
                                ));
                            }
                        } else {
                            results.push(format!(
                                "filepath: {}\nsourcecode:\n{}",
                                self.get_module_file_path(&func.module)
                                    .unwrap_or_else(|| "unknown".to_string()),
                                func.source_code
                            ));
                        }
                    }
                }
            }

            // Fuzzy search structs
            let struct_names: Vec<String> = self.global_structs.keys().cloned().collect();
            let fuzzy_struct_matches =
                FuzzyMatcher::find_fuzzy_matches(&symbol_name, &struct_names, threshold);

            for (name, _score) in fuzzy_struct_matches {
                if let Some(structs) = self.global_structs.get(&name) {
                    for struct_info in structs {
                        if let Some(ref module_filter) = module_filter {
                            if struct_info.module.contains(module_filter) {
                                results.push(format!(
                                    "filepath: {}\nsourcecode:\n{}",
                                    self.get_module_file_path(&struct_info.module)
                                        .unwrap_or_else(|| "unknown".to_string()),
                                    struct_info.source_code
                                ));
                            }
                        } else {
                            results.push(format!(
                                "filepath: {}\nsourcecode:\n{}",
                                self.get_module_file_path(&struct_info.module)
                                    .unwrap_or_else(|| "unknown".to_string()),
                                struct_info.source_code
                            ));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get file path for a module by name
    fn get_module_file_path(&self, module_name: &str) -> Option<String> {
        self.modules
            .iter()
            .find(|m| m.name == module_name)
            .map(|m| m.file_path.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "AdvancedProjectAnalyzer(modules={}, functions={}, structs={})",
            self.modules.len(),
            self.global_functions
                .values()
                .map(|v| v.len())
                .sum::<usize>(),
            self.global_structs.values().map(|v| v.len()).sum::<usize>()
        )
    }
}

/// Basic Move project analyzer for simple symbol lookup
#[pyclass]
pub struct ProjectAnalyzer {
    #[pyo3(get)]
    pub modules: Vec<ModuleInfo>,
    #[pyo3(get)]
    pub global_functions: HashMap<String, Vec<FunctionInfo>>,
    #[pyo3(get)]
    pub global_structs: HashMap<String, Vec<StructInfo>>,
    #[pyo3(get)]
    pub dependency_graph: HashMap<String, Vec<String>>,
}

#[pymethods]
impl ProjectAnalyzer {
    #[new]
    pub fn new() -> Self {
        ProjectAnalyzer {
            modules: Vec::new(),
            global_functions: HashMap::new(),
            global_structs: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }

    /// Parse a single Move file and add to project
    pub fn add_file(&mut self, file_path: &str) -> PyResult<bool> {
        let path = Path::new(file_path);
        match parse_move_file(path)? {
            Some(module_info) => {
                self.add_module(module_info);
                Ok(true)
            }
            None => Ok(false), // No module found or parse error
        }
    }

    /// Parse an entire directory and add all Move files
    pub fn add_directory(&mut self, dir_path: &str) -> PyResult<usize> {
        let path = Path::new(dir_path);
        let move_files = discover_move_files(path)?;
        let mut added_count = 0;

        for file_path in move_files {
            if let Some(module_info) = parse_move_file(&file_path)? {
                self.add_module(module_info);
                added_count += 1;
            }
        }

        // Build dependency graph after all modules are loaded
        self.build_dependency_graph();

        Ok(added_count)
    }

    /// Add a module to the project and update global indices
    pub fn add_module(&mut self, module_info: ModuleInfo) {
        // Update global function index
        for func in &module_info.functions {
            self.global_functions
                .entry(func.name.clone())
                .or_insert_with(Vec::new)
                .push(func.clone());
        }

        // Update global struct index
        for struct_info in &module_info.structs {
            self.global_structs
                .entry(struct_info.name.clone())
                .or_insert_with(Vec::new)
                .push(struct_info.clone());
        }

        // Add module to collection
        self.modules.push(module_info);
    }

    /// Build dependency graph from module dependencies
    pub fn build_dependency_graph(&mut self) {
        self.dependency_graph.clear();

        for module in &self.modules {
            self.dependency_graph
                .insert(module.name.clone(), module.dependencies.clone());
        }
    }

    /// Get all module names
    pub fn get_module_names(&self) -> Vec<String> {
        self.modules.iter().map(|m| m.name.clone()).collect()
    }

    /// Get module by name
    pub fn get_module(&self, name: &str) -> Option<ModuleInfo> {
        self.modules.iter().find(|m| m.name == name).cloned()
    }

    /// Find functions by name across all modules
    pub fn find_function(&self, name: &str) -> Vec<FunctionInfo> {
        self.global_functions.get(name).cloned().unwrap_or_default()
    }

    /// Find structs by name across all modules
    pub fn find_struct(&self, name: &str) -> Vec<StructInfo> {
        self.global_structs.get(name).cloned().unwrap_or_default()
    }

    /// Get all functions in a specific module
    pub fn get_module_functions(&self, module_name: &str) -> Vec<FunctionInfo> {
        self.modules
            .iter()
            .find(|m| m.name == module_name)
            .map(|m| m.functions.clone())
            .unwrap_or_default()
    }

    /// Get all structs in a specific module
    pub fn get_module_structs(&self, module_name: &str) -> Vec<StructInfo> {
        self.modules
            .iter()
            .find(|m| m.name == module_name)
            .map(|m| m.structs.clone())
            .unwrap_or_default()
    }

    /// Get module dependencies
    pub fn get_module_dependencies(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph
            .get(module_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get modules that depend on the given module
    pub fn get_dependent_modules(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph
            .iter()
            .filter(|(_, deps)| deps.contains(&module_name.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Get dependency chain for a module (recursive dependencies)
    pub fn get_dependency_chain(&self, module_name: &str) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut chain = Vec::new();
        collect_dependencies_recursive(
            &self.dependency_graph,
            module_name,
            &mut visited,
            &mut chain,
        );
        chain
    }

    fn __repr__(&self) -> String {
        format!(
            "ProjectAnalyzer(modules={}, total_functions={}, total_structs={})",
            self.modules.len(),
            self.global_functions
                .values()
                .map(|v| v.len())
                .sum::<usize>(),
            self.global_structs.values().map(|v| v.len()).sum::<usize>()
        )
    }
}
