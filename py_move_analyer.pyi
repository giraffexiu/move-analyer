"""Python type stubs for py_move_analyer module.

This module provides Move language parsing functionality implemented in Rust.
"""

from typing import List, Optional, Dict, Tuple, Union

class FunctionInfo:
    """Information about a Move function"""
    name: str
    module: str
    visibility: str
    parameters: List[str]
    return_type: str
    is_entry: bool
    is_native: bool
    
    def __repr__(self) -> str: ...

class StructInfo:
    """Information about a Move struct"""
    name: str
    module: str
    fields: List[str]
    abilities: List[str]
    is_native: bool
    
    def __repr__(self) -> str: ...

class SymbolExtractor:
    """Extractor for symbols from a single Move file"""
    functions: List[FunctionInfo]
    structs: List[StructInfo]
    modules: List[str]
    
    def get_functions_by_name(self, name: str) -> List[FunctionInfo]: ...
    def get_structs_by_name(self, name: str) -> List[StructInfo]: ...
    def get_functions_by_module(self, module: str) -> List[FunctionInfo]: ...
    def get_structs_by_module(self, module: str) -> List[StructInfo]: ...
    def __repr__(self) -> str: ...

class ModuleInfo:
    """Information about a Move module"""
    name: str
    address: str
    file_path: str
    dependencies: List[str]
    functions: List[FunctionInfo]
    structs: List[StructInfo]
    
    def __repr__(self) -> str: ...

class ProjectAnalyzer:
    """Basic project-level analyzer for multi-file Move projects"""
    modules: List[ModuleInfo]
    global_functions: Dict[str, List[FunctionInfo]]
    global_structs: Dict[str, List[StructInfo]]
    dependency_graph: Dict[str, List[str]]
    
    def __init__(self) -> None: ...
    def add_file(self, file_path: str) -> bool: ...
    def add_directory(self, dir_path: str) -> int: ...
    def get_module_names(self) -> List[str]: ...
    def get_module(self, name: str) -> Optional[ModuleInfo]: ...
    def find_function(self, name: str) -> List[FunctionInfo]: ...
    def find_struct(self, name: str) -> List[StructInfo]: ...
    def get_module_functions(self, module_name: str) -> List[FunctionInfo]: ...
    def get_module_structs(self, module_name: str) -> List[StructInfo]: ...
    def get_module_dependencies(self, module_name: str) -> List[str]: ...
    def get_dependent_modules(self, module_name: str) -> List[str]: ...
    def get_dependency_chain(self, module_name: str) -> List[str]: ...
    def __repr__(self) -> str: ...

class AdvancedProjectAnalyzer:
    """Enhanced project analyzer with advanced query capabilities"""
    modules: List[ModuleInfo]
    global_functions: Dict[str, List[FunctionInfo]]
    global_structs: Dict[str, List[StructInfo]]
    dependency_graph: Dict[str, List[str]]
    
    def __init__(self) -> None: ...
    def add_file(self, file_path: str) -> bool: ...
    def add_directory(self, dir_path: str) -> int: ...
    
    # Advanced search methods
    def fuzzy_search_functions(self, query: str, threshold: float) -> List[Tuple[FunctionInfo, float]]: ...
    def fuzzy_search_structs(self, query: str, threshold: float) -> List[Tuple[StructInfo, float]]: ...
    
    # Unified search interface
    def advanced_search(self, query: str, search_type: str, match_type: str, threshold: Optional[float] = None) -> List[str]: ...
    
    # Cache management
    def get_cache_stats(self) -> Dict[str, int]: ...
    def clear_cache(self) -> None: ...
    
    def __repr__(self) -> str: ...

# Basic functions
def parse(content: str) -> str:
    """Parse Move source code and return AST as debug string"""
    ...

def extract_symbols(source: str) -> SymbolExtractor:
    """Extract symbols (functions and structs) from Move source code"""
    ...

# Project analysis functions
def analyze_project(dir_path: str) -> ProjectAnalyzer:
    """Create a basic project analyzer from a directory"""
    ...

def analyze_files(file_paths: List[str]) -> ProjectAnalyzer:
    """Create a basic project analyzer from multiple files"""
    ...

# Advanced analysis functions
def analyze_project_advanced(dir_path: str) -> AdvancedProjectAnalyzer:
    """Create an advanced project analyzer from a directory"""
    ...

def analyze_files_advanced(file_paths: List[str]) -> AdvancedProjectAnalyzer:
    """Create an advanced project analyzer from multiple files"""
    ...

# Query functions
def search_symbols(
    analyzer: AdvancedProjectAnalyzer,
    query: str,
    symbol_type: str,  # "function" or "struct"
    match_type: str,   # "exact" or "fuzzy"
    threshold: Optional[float] = None
) -> List[str]:
    """Unified search function with simplified query types"""
    ...