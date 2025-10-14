use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use move_compiler::parser::syntax::parse_file_string;
use std::sync::{Arc, RwLock};
use move_compiler::shared::{CompilationEnv, Flags, PackageConfig};
use move_compiler::diagnostics::warning_filters::WarningFiltersBuilder;
use move_compiler::editions::{Flavor, Edition};
use move_command_line_common::files::FileHash;
use move_compiler::parser::ast::{ModuleMember, Visibility};


/// Trie node for efficient prefix searching
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    symbols: Vec<String>, // Store symbol identifiers at this node
}

impl TrieNode {
    #[allow(dead_code)]
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_end: false,
            symbols: Vec::new(),
        }
    }
    
    #[allow(dead_code)]
    fn insert(&mut self, word: &str, symbol_id: String) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }
        current.is_end = true;
        current.symbols.push(symbol_id);
    }
    
    #[allow(dead_code)]
    fn search_prefix(&self, prefix: &str) -> Vec<String> {
        let mut current = self;
        for ch in prefix.chars() {
            if let Some(node) = current.children.get(&ch) {
                current = node;
            } else {
                return Vec::new();
            }
        }
        
        let mut results = Vec::new();
        self.collect_all_symbols(current, &mut results);
        results
    }
    
    #[allow(dead_code)]
    fn collect_all_symbols(&self, node: &TrieNode, results: &mut Vec<String>) {
        if node.is_end {
            results.extend(node.symbols.iter().cloned());
        }
        for child in node.children.values() {
            self.collect_all_symbols(child, results);
        }
    }
}

/// Inverted index for efficient symbol searching
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct InvertedIndex {
    // Map from token to list of symbol IDs containing that token
    token_to_symbols: HashMap<String, Vec<String>>,
    // Map from symbol ID to full symbol info
    symbol_info: HashMap<String, (String, String)>, // (name, module)
}

impl InvertedIndex {
    #[allow(dead_code)]
    fn new() -> Self {
        InvertedIndex {
            token_to_symbols: HashMap::new(),
            symbol_info: HashMap::new(),
        }
    }
    
    #[allow(dead_code)]
    fn add_symbol(&mut self, symbol_id: String, name: &str, module: &str) {
        // Tokenize the symbol name
        let tokens = self.tokenize(name);
        
        // Add to inverted index
        for token in tokens {
            self.token_to_symbols
                .entry(token)
                .or_insert_with(Vec::new)
                .push(symbol_id.clone());
        }
        
        // Store symbol info
        self.symbol_info.insert(symbol_id, (name.to_string(), module.to_string()));
    }
    
    #[allow(dead_code)]
    fn search_tokens(&self, query_tokens: &[String]) -> Vec<String> {
        if query_tokens.is_empty() {
            return Vec::new();
        }
        
        // Find symbols containing all query tokens (AND operation)
        let mut result_sets: Vec<&Vec<String>> = Vec::new();
        
        for token in query_tokens {
            if let Some(symbols) = self.token_to_symbols.get(token) {
                result_sets.push(symbols);
            } else {
                return Vec::new(); // If any token is not found, no results
            }
        }
        
        // Find intersection of all sets
        if result_sets.is_empty() {
            return Vec::new();
        }
        
        let mut result: Vec<String> = result_sets[0].clone();
        for set in result_sets.iter().skip(1) {
            result.retain(|item| set.contains(item));
        }
        
        result
    }
    
    #[allow(dead_code)]
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        
        // Split by common delimiters and convert to lowercase
        let words: Vec<&str> = text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .collect();
        
        for word in words {
            let lower_word = word.to_lowercase();
            tokens.push(lower_word.clone());
            
            // Add camelCase/PascalCase tokens
            let camel_tokens = self.split_camel_case(word);
            for token in camel_tokens {
                if !token.is_empty() && token != lower_word {
                    tokens.push(token.to_lowercase());
                }
            }
        }
        
        tokens.sort();
        tokens.dedup();
        tokens
    }
    
    #[allow(dead_code)]
    fn split_camel_case(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        
        for ch in text.chars() {
            if ch.is_uppercase() && !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            current.push(ch);
        }
        
        if !current.is_empty() {
            tokens.push(current);
        }
        
        tokens
    }
}

/// Query cache for performance optimization
#[derive(Debug)]
struct QueryCache {
    function_cache: Arc<RwLock<HashMap<String, Vec<FunctionInfo>>>>,
    struct_cache: Arc<RwLock<HashMap<String, Vec<StructInfo>>>>,
    prefix_cache: Arc<RwLock<HashMap<String, Vec<String>>>>,
    max_size: usize,
}

impl QueryCache {
    fn new(max_size: usize) -> Self {
        QueryCache {
            function_cache: Arc::new(RwLock::new(HashMap::new())),
            struct_cache: Arc::new(RwLock::new(HashMap::new())),
            prefix_cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }
    

    

    
    fn clear(&self) {
        self.function_cache.write().unwrap().clear();
        self.struct_cache.write().unwrap().clear();
        self.prefix_cache.write().unwrap().clear();
    }
}
/// Fuzzy matching utilities for symbol search
struct FuzzyMatcher;

impl FuzzyMatcher {
    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
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
        
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1,      // deletion
                        matrix[i][j - 1] + 1       // insertion
                    ),
                    matrix[i - 1][j - 1] + cost    // substitution
                );
            }
        }
        
        matrix[len1][len2]
    }
    
    /// Calculate similarity score (0.0 to 1.0, higher is more similar)
    fn similarity_score(s1: &str, s2: &str) -> f64 {
        let max_len = std::cmp::max(s1.len(), s2.len());
        if max_len == 0 {
            return 1.0;
        }
        
        let distance = Self::levenshtein_distance(s1, s2);
        1.0 - (distance as f64 / max_len as f64)
    }
    
    /// Find fuzzy matches with minimum similarity threshold
    fn find_fuzzy_matches(query: &str, candidates: &[String], threshold: f64) -> Vec<(String, f64)> {
        let mut matches = Vec::new();
        
        for candidate in candidates {
            let score = Self::similarity_score(query, candidate);
            if score >= threshold {
                matches.push((candidate.clone(), score));
            }
        }
        
        // Sort by similarity score (descending)
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches
    }
    

    

}

#[pyclass]
#[derive(Clone, Debug)]
struct ModuleInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    address: String,
    #[pyo3(get)]
    file_path: String,
    #[pyo3(get)]
    dependencies: Vec<String>,
    #[pyo3(get)]
    functions: Vec<FunctionInfo>,
    #[pyo3(get)]
    structs: Vec<StructInfo>,
}

#[pymethods]
impl ModuleInfo {
    fn __repr__(&self) -> String {
        format!("ModuleInfo(name='{}', address='{}', file='{}', deps={}, funcs={}, structs={})", 
                self.name, self.address, self.file_path, self.dependencies.len(), 
                self.functions.len(), self.structs.len())
    }
}

/// Enhanced ProjectAnalyzer with advanced query capabilities
#[pyclass]
struct AdvancedProjectAnalyzer {
    #[pyo3(get)]
    modules: Vec<ModuleInfo>,
    #[pyo3(get)]
    global_functions: HashMap<String, Vec<FunctionInfo>>,
    #[pyo3(get)]
    global_structs: HashMap<String, Vec<StructInfo>>,
    #[pyo3(get)]
    dependency_graph: HashMap<String, Vec<String>>,
    
    // Simplified indexing - only keep cache for performance
    query_cache: QueryCache,
}

#[pymethods]
impl AdvancedProjectAnalyzer {
    #[new]
    fn new() -> Self {
        AdvancedProjectAnalyzer {
            modules: Vec::new(),
            global_functions: HashMap::new(),
            global_structs: HashMap::new(),
            dependency_graph: HashMap::new(),
            query_cache: QueryCache::new(1000), // Cache up to 1000 queries
        }
    }
    
    /// Parse a single Move file and add to project
    fn add_file(&mut self, file_path: &str) -> PyResult<bool> {
        let path = Path::new(file_path);
        match parse_move_file(path)? {
            Some(module_info) => {
                self.add_module(module_info);
                Ok(true)
            }
            None => Ok(false),
        }
    }
    
    /// Parse an entire directory and add all Move files
    fn add_directory(&mut self, dir_path: &str) -> PyResult<usize> {
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
    
    /// Add a module and update all indices
    fn add_module(&mut self, module_info: ModuleInfo) {
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
    
    /// Build dependency graph
    fn build_dependency_graph(&mut self) {
        self.dependency_graph.clear();
        for module in &self.modules {
            self.dependency_graph.insert(
                module.name.clone(),
                module.dependencies.clone()
            );
        }
        
        // Clear cache when indices are rebuilt
        self.query_cache.clear();
    }
    
    /// Fuzzy search functions
    fn fuzzy_search_functions(&self, query: &str, threshold: f64) -> Vec<(FunctionInfo, f64)> {
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
    
    /// Fuzzy search structs
    fn fuzzy_search_structs(&self, query: &str, threshold: f64) -> Vec<(StructInfo, f64)> {
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
    
    /// Advanced search with simplified criteria
    fn advanced_search(&self, 
                      query: &str, 
                      search_type: &str, 
                      match_type: &str, 
                      threshold: Option<f64>) -> PyResult<Vec<String>> {
        
        let results = match (search_type, match_type) {
            ("function", "exact") => {
                self.global_functions.get(query)
                    .map(|funcs| funcs.iter().map(|f| format!("{}::{}", f.module, f.name)).collect())
                    .unwrap_or_default()
            }
            ("function", "fuzzy") => {
                let threshold = threshold.unwrap_or(0.6);
                self.fuzzy_search_functions(query, threshold)
                    .iter()
                    .map(|(f, score)| format!("{}::{} (score: {:.2})", f.module, f.name, score))
                    .collect()
            }
            ("struct", "exact") => {
                self.global_structs.get(query)
                    .map(|structs| structs.iter().map(|s| format!("{}::{}", s.module, s.name)).collect())
                    .unwrap_or_default()
            }
            ("struct", "fuzzy") => {
                let threshold = threshold.unwrap_or(0.6);
                self.fuzzy_search_structs(query, threshold)
                    .iter()
                    .map(|(s, score)| format!("{}::{} (score: {:.2})", s.module, s.name, score))
                    .collect()
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid search_type or match_type. Use 'function'/'struct' and 'exact'/'fuzzy'"
                ));
            }
        };
        
        Ok(results)
    }
    
    /// Get cache statistics
    fn get_cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("function_cache_size".to_string(), 
                    self.query_cache.function_cache.read().unwrap().len());
        stats.insert("struct_cache_size".to_string(), 
                    self.query_cache.struct_cache.read().unwrap().len());
        stats.insert("max_cache_size".to_string(), self.query_cache.max_size);
        stats
    }
    
    /// Clear all caches
    fn clear_cache(&self) {
        self.query_cache.clear();
    }
    
    fn __repr__(&self) -> String {
        format!("AdvancedProjectAnalyzer(modules={}, functions={}, structs={}, cached_queries={})", 
                self.modules.len(),
                self.global_functions.values().map(|v| v.len()).sum::<usize>(),
                self.global_structs.values().map(|v| v.len()).sum::<usize>(),
                self.query_cache.function_cache.read().unwrap().len() + 
                self.query_cache.struct_cache.read().unwrap().len())
    }
}
#[pyclass]
struct ProjectAnalyzer {
    #[pyo3(get)]
    modules: Vec<ModuleInfo>,
    #[pyo3(get)]
    global_functions: HashMap<String, Vec<FunctionInfo>>,
    #[pyo3(get)]
    global_structs: HashMap<String, Vec<StructInfo>>,
    #[pyo3(get)]
    dependency_graph: HashMap<String, Vec<String>>,
}

#[pymethods]
impl ProjectAnalyzer {
    #[new]
    fn new() -> Self {
        ProjectAnalyzer {
            modules: Vec::new(),
            global_functions: HashMap::new(),
            global_structs: HashMap::new(),
            dependency_graph: HashMap::new(),
        }
    }
    
    /// Parse a single Move file and add to project
    fn add_file(&mut self, file_path: &str) -> PyResult<bool> {
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
    fn add_directory(&mut self, dir_path: &str) -> PyResult<usize> {
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
    fn add_module(&mut self, module_info: ModuleInfo) {
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
    fn build_dependency_graph(&mut self) {
        self.dependency_graph.clear();
        
        for module in &self.modules {
            self.dependency_graph.insert(
                module.name.clone(),
                module.dependencies.clone()
            );
        }
    }
    
    /// Get all module names
    fn get_module_names(&self) -> Vec<String> {
        self.modules.iter().map(|m| m.name.clone()).collect()
    }
    
    /// Get module by name
    fn get_module(&self, name: &str) -> Option<ModuleInfo> {
        self.modules.iter().find(|m| m.name == name).cloned()
    }
    
    /// Find functions by name across all modules
    fn find_function(&self, name: &str) -> Vec<FunctionInfo> {
        self.global_functions.get(name).cloned().unwrap_or_default()
    }
    
    /// Find structs by name across all modules
    fn find_struct(&self, name: &str) -> Vec<StructInfo> {
        self.global_structs.get(name).cloned().unwrap_or_default()
    }
    
    /// Get all functions in a specific module
    fn get_module_functions(&self, module_name: &str) -> Vec<FunctionInfo> {
        self.modules.iter()
            .find(|m| m.name == module_name)
            .map(|m| m.functions.clone())
            .unwrap_or_default()
    }
    
    /// Get all structs in a specific module
    fn get_module_structs(&self, module_name: &str) -> Vec<StructInfo> {
        self.modules.iter()
            .find(|m| m.name == module_name)
            .map(|m| m.structs.clone())
            .unwrap_or_default()
    }
    
    /// Get module dependencies
    fn get_module_dependencies(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph.get(module_name).cloned().unwrap_or_default()
    }
    
    /// Get modules that depend on the given module
    fn get_dependent_modules(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph.iter()
            .filter(|(_, deps)| deps.contains(&module_name.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Get dependency chain for a module (recursive dependencies)
    fn get_dependency_chain(&self, module_name: &str) -> Vec<String> {
        let mut visited = std::collections::HashSet::new();
        let mut chain = Vec::new();
        collect_dependencies_recursive(&self.dependency_graph, module_name, &mut visited, &mut chain);
        chain
    }
    
    fn __repr__(&self) -> String {
        format!("ProjectAnalyzer(modules={}, total_functions={}, total_structs={})", 
                self.modules.len(), 
                self.global_functions.values().map(|v| v.len()).sum::<usize>(),
                self.global_structs.values().map(|v| v.len()).sum::<usize>())
    }
}

// Helper function moved outside impl block
fn collect_dependencies_recursive(
    dependency_graph: &std::collections::HashMap<String, Vec<String>>,
    module_name: &str, 
    visited: &mut std::collections::HashSet<String>, 
    chain: &mut Vec<String>
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
#[pyclass]
#[derive(Clone, Debug)]
struct FunctionInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    module: String,
    #[pyo3(get)]
    visibility: String,
    #[pyo3(get)]
    parameters: Vec<String>,
    #[pyo3(get)]
    return_type: String,
    #[pyo3(get)]
    is_entry: bool,
    #[pyo3(get)]
    is_native: bool,
}

#[pymethods]
impl FunctionInfo {
    fn __repr__(&self) -> String {
        format!("FunctionInfo(name='{}', module='{}', visibility='{}')", 
                self.name, self.module, self.visibility)
    }
}

/// Struct information extracted from AST
#[pyclass]
#[derive(Clone, Debug)]
struct StructInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    module: String,
    #[pyo3(get)]
    fields: Vec<String>,
    #[pyo3(get)]
    abilities: Vec<String>,
    #[pyo3(get)]
    is_native: bool,
}

#[pymethods]
impl StructInfo {
    fn __repr__(&self) -> String {
        format!("StructInfo(name='{}', module='{}', fields={:?})", 
                self.name, self.module, self.fields)
    }
}

/// Symbol extraction result
#[pyclass]
struct SymbolExtractor {
    #[pyo3(get)]
    functions: Vec<FunctionInfo>,
    #[pyo3(get)]
    structs: Vec<StructInfo>,
    #[pyo3(get)]
    modules: Vec<String>,
}

#[pymethods]
impl SymbolExtractor {
    fn get_functions_by_name(&self, name: &str) -> Vec<FunctionInfo> {
        self.functions.iter()
            .filter(|f| f.name == name)
            .cloned()
            .collect()
    }
    
    fn get_structs_by_name(&self, name: &str) -> Vec<StructInfo> {
        self.structs.iter()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }
    
    fn get_functions_by_module(&self, module: &str) -> Vec<FunctionInfo> {
        self.functions.iter()
            .filter(|f| f.module == module)
            .cloned()
            .collect()
    }
    
    fn get_structs_by_module(&self, module: &str) -> Vec<StructInfo> {
        self.structs.iter()
            .filter(|s| s.module == module)
            .cloned()
            .collect()
    }
    
    fn __repr__(&self) -> String {
        format!("SymbolExtractor(modules={}, functions={}, structs={})", 
                self.modules.len(), self.functions.len(), self.structs.len())
    }
}

/// Discover Move files in a directory recursively
fn discover_move_files(dir_path: &Path) -> PyResult<Vec<PathBuf>> {
    let mut move_files = Vec::new();
    
    if !dir_path.exists() {
        return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(
            format!("Directory not found: {}", dir_path.display())
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
                        if !name.starts_with('.') && 
                           !["target", "build", "node_modules", "deps"].contains(&name) {
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

/// Parse a single Move file and extract module information
fn parse_move_file(file_path: &Path) -> PyResult<Option<ModuleInfo>> {
    let content = fs::read_to_string(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
            format!("Failed to read file {}: {}", file_path.display(), e)
        ))?;
    
    let file_hash = FileHash::new(&content);
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
    
    match move_compiler::parser::syntax::parse_file_string(&mut env, file_hash, &content, None) {
        Ok(definitions) => {
            // Extract module information from definitions
            for definition in &definitions {
                if let move_compiler::parser::ast::Definition::Module(module) = definition {
                    let extractor = extract_symbols_from_ast(&definitions);
                    
                    let module_name = format!("{}::{}", 
                        module.address.as_ref().map(|a| format!("{}", a)).unwrap_or_else(|| "unknown".to_string()),
                        module.name.0.to_string());
                    
                    let address = module.address.as_ref()
                        .map(|a| format!("{}", a))
                        .unwrap_or_else(|| "unknown".to_string());
                    
                    // Extract dependencies from use statements
                    let dependencies = extract_dependencies(module);
                    
                    return Ok(Some(ModuleInfo {
                        name: module_name,
                        address,
                        file_path: file_path.to_string_lossy().to_string(),
                        dependencies,
                        functions: extractor.functions,
                        structs: extractor.structs,
                    }));
                }
            }
            Ok(None) // No module found in file
        }
        Err(_) => Ok(None), // Parse error, skip file
    }
}

/// Extract module dependencies from use statements
fn extract_dependencies(module: &move_compiler::parser::ast::ModuleDefinition) -> Vec<String> {
    let mut dependencies = Vec::new();
    
    for member in &module.members {
        if let move_compiler::parser::ast::ModuleMember::Use(use_decl) = member {
            // Extract module name from use declaration
            let dep_name = format!("{:?}", use_decl.use_);
            dependencies.push(dep_name);
        }
    }
    
    dependencies
}
fn extract_symbols_from_ast(modules: &[move_compiler::parser::ast::Definition]) -> SymbolExtractor {
    let mut functions = Vec::new();
    let mut structs = Vec::new();
    let mut module_names = Vec::new();
    
    for definition in modules {
        if let move_compiler::parser::ast::Definition::Module(module) = definition {
            let module_name = format!("{}::{}", 
                module.address.as_ref().map(|a| format!("{}", a)).unwrap_or_else(|| "unknown".to_string()),
                module.name.0.to_string());
            module_names.push(module_name.clone());
            
            for member in &module.members {
                match member {
                    ModuleMember::Function(func) => {
                        let visibility = match func.visibility {
                            Visibility::Public(_) => "public".to_string(),
                            Visibility::Friend(_) => "friend".to_string(),
                            Visibility::Internal => "internal".to_string(),
                            _ => "unknown".to_string(),
                        };
                        
                        let parameters: Vec<String> = func.signature.parameters.iter()
                            .map(|param| {
                                let param_name = if let Some(name) = &param.0 {
                                    format!("{:?}", name)
                                } else {
                                    "unnamed".to_string()
                                };
                                let param_type = format!("{:?}", param.1).replace("\"", "");
                                format!("{}: {}", param_name, param_type)
                            })
                            .collect();
                        
                        let return_type = format!("{:?}", func.signature.return_type);
                        
                        let function_info = FunctionInfo {
                            name: func.name.0.to_string(),
                            module: module_name.clone(),
                            visibility,
                            parameters,
                            return_type,
                            is_entry: func.entry.is_some(),
                            is_native: matches!(func.body.value, move_compiler::parser::ast::FunctionBody_::Native),
                        };
                        
                        functions.push(function_info);
                    }
                    ModuleMember::Struct(struct_def) => {
                        let fields: Vec<String> = match &struct_def.fields {
                            move_compiler::parser::ast::StructFields::Named(field_vec) => {
                                field_vec.iter()
                                    .map(|field| {
                                        let field_name = format!("{:?}", field.0);
                                        let field_type = format!("{:?}", field.1).replace("\"", "");
                                        format!("{}: {}", field_name, field_type)
                                    })
                                    .collect()
                            }
                            move_compiler::parser::ast::StructFields::Native(_) => vec!["native".to_string()],
                            move_compiler::parser::ast::StructFields::Positional(types) => {
                                types.iter()
                                    .enumerate()
                                    .map(|(i, t)| format!("field_{}: {}", i, format!("{:?}", t)))
                                    .collect()
                            }
                        };
                        
                        let abilities: Vec<String> = struct_def.abilities.iter()
                            .map(|ability| format!("{:?}", ability.value))
                            .collect();
                        
                        let struct_info = StructInfo {
                            name: struct_def.name.0.to_string(),
                            module: module_name.clone(),
                            fields,
                            abilities,
                            is_native: matches!(struct_def.fields, move_compiler::parser::ast::StructFields::Native(_)),
                        };
                        
                        structs.push(struct_info);
                    }
                    _ => {} // Handle other members like constants, use declarations, etc.
                }
            }
        }
    }
    
    SymbolExtractor {
        functions,
        structs,
        modules: module_names,
    }
}

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

/// Unified search function with simplified query types
#[pyfunction]
fn search_symbols(
    analyzer: &AdvancedProjectAnalyzer,
    query: &str,
    symbol_type: &str,      // "function" or "struct"
    match_type: &str,       // "exact" or "fuzzy"
    threshold: Option<f64>  // For fuzzy matching
) -> PyResult<Vec<String>> {
    analyzer.advanced_search(query, symbol_type, match_type, threshold)
}

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
#[pyfunction]
fn extract_symbols(content: &str) -> PyResult<SymbolExtractor> {
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
    
    match parse_file_string(&mut env, file_hash, content, None) {
        Ok(parsed) => {
            let extractor = extract_symbols_from_ast(&*parsed);
            Ok(extractor)
        }
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Parse error: {:?}", err)
        )),
    }
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
    m.add_function(wrap_pyfunction!(search_symbols, m)?)?;
    
    // Data classes
    m.add_class::<FunctionInfo>()?;
    m.add_class::<StructInfo>()?;
    m.add_class::<SymbolExtractor>()?;
    m.add_class::<ModuleInfo>()?;
    m.add_class::<ProjectAnalyzer>()?;
    m.add_class::<AdvancedProjectAnalyzer>()?;
    
    Ok(())
}