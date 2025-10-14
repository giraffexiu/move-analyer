use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use pyo3::prelude::*;

use crate::types::{ModuleInfo, FunctionInfo, StructInfo};
use crate::parser::parse_move_file;
use crate::utils::{discover_move_files, collect_dependencies_recursive};

/// Trie node for efficient prefix searching
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    symbols: Vec<String>, // Store symbol identifiers at this node
}

impl TrieNode {
    #[allow(dead_code)]
    pub fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            is_end: false,
            symbols: Vec::new(),
        }
    }
    
    #[allow(dead_code)]
    pub fn insert(&mut self, word: &str, symbol_id: String) {
        let mut current = self;
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }
        current.is_end = true;
        current.symbols.push(symbol_id);
    }
    
    #[allow(dead_code)]
    pub fn search_prefix(&self, prefix: &str) -> Vec<String> {
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
pub struct InvertedIndex {
    // Map from token to list of symbol IDs containing that token
    token_to_symbols: HashMap<String, Vec<String>>,
    // Map from symbol ID to full symbol info
    symbol_info: HashMap<String, (String, String)>, // (name, module)
}

impl InvertedIndex {
    #[allow(dead_code)]
    pub fn new() -> Self {
        InvertedIndex {
            token_to_symbols: HashMap::new(),
            symbol_info: HashMap::new(),
        }
    }
    
    #[allow(dead_code)]
    pub fn add_symbol(&mut self, symbol_id: String, name: &str, module: &str) {
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
    pub fn search_tokens(&self, query_tokens: &[String]) -> Vec<String> {
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
pub struct QueryCache {
    function_cache: Arc<RwLock<HashMap<String, Vec<FunctionInfo>>>>,
    struct_cache: Arc<RwLock<HashMap<String, Vec<StructInfo>>>>,
    prefix_cache: Arc<RwLock<HashMap<String, Vec<String>>>>,
    max_size: usize,
}

impl QueryCache {
    pub fn new(max_size: usize) -> Self {
        QueryCache {
            function_cache: Arc::new(RwLock::new(HashMap::new())),
            struct_cache: Arc::new(RwLock::new(HashMap::new())),
            prefix_cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }
    
    pub fn clear(&self) {
        self.function_cache.write().unwrap().clear();
        self.struct_cache.write().unwrap().clear();
        self.prefix_cache.write().unwrap().clear();
    }
}

/// Fuzzy matching utilities for symbol search
pub struct FuzzyMatcher;

impl FuzzyMatcher {
    /// Calculate Damerau-Levenshtein distance between two strings
    /// This includes transposition operations which are common in typos
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
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
                
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(
                        matrix[i - 1][j] + 1,      // deletion
                        matrix[i][j - 1] + 1       // insertion
                    ),
                    matrix[i - 1][j - 1] + cost    // substitution
                );
                
                // Transposition (Damerau extension)
                if i > 1 && j > 1 
                    && s1_chars[i - 1] == s2_chars[j - 2] 
                    && s1_chars[i - 2] == s2_chars[j - 1] {
                    matrix[i][j] = std::cmp::min(
                        matrix[i][j],
                        matrix[i - 2][j - 2] + cost
                    );
                }
            }
        }
        
        matrix[len1][len2]
    }
    
    /// Calculate normalized similarity score (0.0 to 1.0, higher is more similar)
    /// Uses character count instead of byte length for better Unicode support
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
    
    /// Find fuzzy matches with configurable threshold
    /// Default threshold of 0.6 works well for most use cases
    pub fn find_fuzzy_matches(query: &str, candidates: &[String], threshold: f64) -> Vec<(String, f64)> {
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

/// Enhanced ProjectAnalyzer with advanced query capabilities
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
    
    // Simplified indexing - only keep cache for performance
    query_cache: QueryCache,
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
            query_cache: QueryCache::new(1000), // Cache up to 1000 queries
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
            None => Ok(false),
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
        
        self.build_dependency_graph();
        
        Ok(added_count)
    }
    
    /// Add a module and update all indices
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
    
    /// Build dependency graph
    pub fn build_dependency_graph(&mut self) {
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
    
    /// Fuzzy search structs
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
    
    /// Unified search function that tries exact match first, then fuzzy match
    /// Supports: function_name, module::function_name, struct_name, module::struct_name
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
                        results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                           self.get_module_file_path(&func.module).unwrap_or_else(|| "unknown".to_string()),
                                           func.module, func.name));
                        found_exact = true;
                    }
                } else {
                    results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                       self.get_module_file_path(&func.module).unwrap_or_else(|| "unknown".to_string()),
                                       func.module, func.name));
                    found_exact = true;
                }
            }
        }
        
        // Search structs
        if let Some(structs) = self.global_structs.get(&symbol_name) {
            for struct_info in structs {
                if let Some(ref module_filter) = module_filter {
                    if struct_info.module.contains(module_filter) {
                        results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                           self.get_module_file_path(&struct_info.module).unwrap_or_else(|| "unknown".to_string()),
                                           struct_info.module, struct_info.name));
                        found_exact = true;
                    }
                } else {
                    results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                       self.get_module_file_path(&struct_info.module).unwrap_or_else(|| "unknown".to_string()),
                                       struct_info.module, struct_info.name));
                    found_exact = true;
                }
            }
        }
        
        // If no exact matches found, try fuzzy matching
        if !found_exact {
            let threshold = 0.6; // Standard threshold for fuzzy matching
            
            // Fuzzy search functions
            let function_names: Vec<String> = self.global_functions.keys().cloned().collect();
            let fuzzy_func_matches = FuzzyMatcher::find_fuzzy_matches(&symbol_name, &function_names, threshold);
            
            for (name, _score) in fuzzy_func_matches {
                if let Some(functions) = self.global_functions.get(&name) {
                    for func in functions {
                        if let Some(ref module_filter) = module_filter {
                            if func.module.contains(module_filter) {
                                results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                                   self.get_module_file_path(&func.module).unwrap_or_else(|| "unknown".to_string()),
                                                   func.module, func.name));
                            }
                        } else {
                            results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                               self.get_module_file_path(&func.module).unwrap_or_else(|| "unknown".to_string()),
                                               func.module, func.name));
                        }
                    }
                }
            }
            
            // Fuzzy search structs
            let struct_names: Vec<String> = self.global_structs.keys().cloned().collect();
            let fuzzy_struct_matches = FuzzyMatcher::find_fuzzy_matches(&symbol_name, &struct_names, threshold);
            
            for (name, _score) in fuzzy_struct_matches {
                if let Some(structs) = self.global_structs.get(&name) {
                    for struct_info in structs {
                        if let Some(ref module_filter) = module_filter {
                            if struct_info.module.contains(module_filter) {
                                results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                                   self.get_module_file_path(&struct_info.module).unwrap_or_else(|| "unknown".to_string()),
                                                   struct_info.module, struct_info.name));
                            }
                        } else {
                            results.push(format!("filepath: {}\nsourcecode: {}::{}", 
                                               self.get_module_file_path(&struct_info.module).unwrap_or_else(|| "unknown".to_string()),
                                               struct_info.module, struct_info.name));
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Helper function to get file path for a module
    fn get_module_file_path(&self, module_name: &str) -> Option<String> {
        self.modules.iter()
            .find(|m| m.name == module_name)
            .map(|m| m.file_path.clone())
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("function_cache_size".to_string(), 
                    self.query_cache.function_cache.read().unwrap().len());
        stats.insert("struct_cache_size".to_string(), 
                    self.query_cache.struct_cache.read().unwrap().len());
        stats.insert("max_cache_size".to_string(), self.query_cache.max_size);
        stats
    }
    
    /// Clear all caches
    pub fn clear_cache(&self) {
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

/// Basic ProjectAnalyzer for simple use cases
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
            self.dependency_graph.insert(
                module.name.clone(),
                module.dependencies.clone()
            );
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
        self.modules.iter()
            .find(|m| m.name == module_name)
            .map(|m| m.functions.clone())
            .unwrap_or_default()
    }
    
    /// Get all structs in a specific module
    pub fn get_module_structs(&self, module_name: &str) -> Vec<StructInfo> {
        self.modules.iter()
            .find(|m| m.name == module_name)
            .map(|m| m.structs.clone())
            .unwrap_or_default()
    }
    
    /// Get module dependencies
    pub fn get_module_dependencies(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph.get(module_name).cloned().unwrap_or_default()
    }
    
    /// Get modules that depend on the given module
    pub fn get_dependent_modules(&self, module_name: &str) -> Vec<String> {
        self.dependency_graph.iter()
            .filter(|(_, deps)| deps.contains(&module_name.to_string()))
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Get dependency chain for a module (recursive dependencies)
    pub fn get_dependency_chain(&self, module_name: &str) -> Vec<String> {
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