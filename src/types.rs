use pyo3::prelude::*;

/// Function information extracted from AST
#[pyclass]
#[derive(Clone, Debug)]
pub struct FunctionInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub module: String,
    #[pyo3(get)]
    pub visibility: String,
    #[pyo3(get)]
    pub parameters: Vec<String>,
    #[pyo3(get)]
    pub return_type: String,
    #[pyo3(get)]
    pub is_entry: bool,
    #[pyo3(get)]
    pub is_native: bool,
    #[pyo3(get)]
    pub source_code: String,  // Complete function source code
    // Location information for source code extraction
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub file_path: Option<String>,
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
pub struct StructInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub module: String,
    #[pyo3(get)]
    pub fields: Vec<String>,
    #[pyo3(get)]
    pub abilities: Vec<String>,
    #[pyo3(get)]
    pub is_native: bool,
    #[pyo3(get)]
    pub source_code: String,  // Complete struct source code
    // Location information for source code extraction
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub file_path: Option<String>,
}

#[pymethods]
impl StructInfo {
    fn __repr__(&self) -> String {
        format!("StructInfo(name='{}', module='{}', fields={:?})", 
                self.name, self.module, self.fields)
    }
}

/// Module information containing all extracted data
#[pyclass]
#[derive(Clone, Debug)]
pub struct ModuleInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub address: String,
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub dependencies: Vec<String>,
    #[pyo3(get)]
    pub functions: Vec<FunctionInfo>,
    #[pyo3(get)]
    pub structs: Vec<StructInfo>,
}

#[pymethods]
impl ModuleInfo {
    fn __repr__(&self) -> String {
        format!("ModuleInfo(name='{}', address='{}', file='{}', deps={}, funcs={}, structs={})", 
                self.name, self.address, self.file_path, self.dependencies.len(), 
                self.functions.len(), self.structs.len())
    }
}

/// Symbol extraction result
#[pyclass]
pub struct SymbolExtractor {
    #[pyo3(get)]
    pub functions: Vec<FunctionInfo>,
    #[pyo3(get)]
    pub structs: Vec<StructInfo>,
    #[pyo3(get)]
    pub modules: Vec<String>,
}

#[pymethods]
impl SymbolExtractor {
    pub fn get_functions_by_name(&self, name: &str) -> Vec<FunctionInfo> {
        self.functions.iter()
            .filter(|f| f.name == name)
            .cloned()
            .collect()
    }
    
    pub fn get_structs_by_name(&self, name: &str) -> Vec<StructInfo> {
        self.structs.iter()
            .filter(|s| s.name == name)
            .cloned()
            .collect()
    }
    
    pub fn get_functions_by_module(&self, module: &str) -> Vec<FunctionInfo> {
        self.functions.iter()
            .filter(|f| f.module == module)
            .cloned()
            .collect()
    }
    
    pub fn get_structs_by_module(&self, module: &str) -> Vec<StructInfo> {
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