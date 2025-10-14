use std::path::Path;
use std::fs;
use pyo3::prelude::*;
use move_compiler::parser::syntax::parse_file_string;
use move_compiler::shared::{CompilationEnv, Flags, PackageConfig};
use move_compiler::diagnostics::warning_filters::WarningFiltersBuilder;
use move_compiler::editions::{Flavor, Edition};
use move_command_line_common::files::FileHash;
use move_compiler::parser::ast::{ModuleMember, Visibility};

use crate::types::{ModuleInfo, FunctionInfo, StructInfo, SymbolExtractor};

/// Format Move variable to a readable string
fn format_var(var: &move_compiler::parser::ast::Var) -> String {
    format!("{}", var.0)
}

/// Parse a single Move file and extract module information
pub fn parse_move_file(file_path: &Path) -> PyResult<Option<ModuleInfo>> {
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
pub fn extract_dependencies(module: &move_compiler::parser::ast::ModuleDefinition) -> Vec<String> {
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

/// Extract symbols from AST definitions with source code
pub fn extract_symbols_from_ast_with_source(
    modules: &[move_compiler::parser::ast::Definition], 
    _source_content: &str
) -> SymbolExtractor {
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
                                    format!("{:?}", name).replace("\"", "").trim().to_string()
                                } else {
                                    "unnamed".to_string()
                                };
                                let param_type = format_var(&param.1);
                                format!("{}: {}", param_name, param_type)
                            })
                            .collect();
                        
                        let return_type = format!("{:?}", func.signature.return_type).replace("\"", "");
                        
                        // Generate function source code from AST information
                        let source_code = generate_function_source_code(func, &visibility);
                        
                        let function_info = FunctionInfo {
                            name: func.name.0.to_string(),
                            module: module_name.clone(),
                            visibility,
                            parameters,
                            return_type,
                            is_entry: func.entry.is_some(),
                            is_native: matches!(func.body.value, move_compiler::parser::ast::FunctionBody_::Native),
                            source_code,
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
                        
                        // Generate struct source code from AST information
                        let source_code = generate_struct_source_code(struct_def);
                        
                        let struct_info = StructInfo {
                            name: struct_def.name.0.to_string(),
                            module: module_name.clone(),
                            fields,
                            abilities,
                            is_native: matches!(struct_def.fields, move_compiler::parser::ast::StructFields::Native(_)),
                            source_code,
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

/// Generate function source code from AST
fn generate_function_source_code(
    func: &move_compiler::parser::ast::Function, 
    visibility: &str
) -> String {
    let mut source = String::new();
    
    // Add entry modifier if present
    if func.entry.is_some() {
        source.push_str("entry ");
    }
    
    // Add visibility
    if visibility != "internal" {
        source.push_str(&format!("{} ", visibility));
    }
    
    // Add function keyword and name
    source.push_str(&format!("fun {}(", func.name.0));
    
    // Add parameters
    let params: Vec<String> = func.signature.parameters.iter()
        .map(|param| {
            let param_name = if let Some(name) = &param.0 {
                format!("{:?}", name)
            } else {
                "unnamed".to_string()
            };
            let param_type = format_var(&param.1);
            format!("{}: {}", param_name, param_type)
        })
        .collect();
    source.push_str(&params.join(", "));
    source.push(')');
    
    // Add return type if not unit
    let return_type = format!("{:?}", func.signature.return_type).replace("\"", "");
    if return_type != "()" && return_type != "unit" {
        source.push_str(&format!(": {}", return_type));
    }
    
    // Add body
    match &func.body.value {
        move_compiler::parser::ast::FunctionBody_::Native => {
            source.push_str(" native;");
        }
        move_compiler::parser::ast::FunctionBody_::Defined(_) => {
            source.push_str(" { /* function body */ }");
        }
    }
    
    source
}

/// Generate struct source code from AST
fn generate_struct_source_code(struct_def: &move_compiler::parser::ast::StructDefinition) -> String {
    let mut source = String::new();
    
    // Add struct keyword and name
    source.push_str(&format!("struct {}", struct_def.name.0));
    
    // Add abilities if any
    if !struct_def.abilities.is_empty() {
        let abilities: Vec<String> = struct_def.abilities.iter()
            .map(|ability| format!("{:?}", ability.value))
            .collect();
        source.push_str(&format!(" has {}", abilities.join(", ")));
    }
    
    // Add fields
    match &struct_def.fields {
        move_compiler::parser::ast::StructFields::Named(field_vec) => {
            source.push_str(" {\n");
            for field in field_vec {
                let field_name = format!("{:?}", field.0);
                let field_type = format!("{:?}", field.1).replace("\"", "");
                source.push_str(&format!("    {}: {},\n", field_name, field_type));
            }
            source.push('}');
        }
        move_compiler::parser::ast::StructFields::Native(_) => {
            source.push_str(" native;");
        }
        move_compiler::parser::ast::StructFields::Positional(types) => {
            source.push('(');
            let type_strs: Vec<String> = types.iter()
                .map(|t| format!("{:?}", t))
                .collect();
            source.push_str(&type_strs.join(", "));
            source.push(')');
        }
    }
    
    source
}

/// Extract symbols from AST definitions (backward compatibility)
pub fn extract_symbols_from_ast(modules: &[move_compiler::parser::ast::Definition]) -> SymbolExtractor {
    extract_symbols_from_ast_with_source(modules, "")
}

/// Parse Move source code and return the result as a string
pub fn parse_move_content(content: &str) -> PyResult<String> {
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

/// Extract symbols from Move source code content
pub fn extract_symbols_from_content(content: &str) -> PyResult<SymbolExtractor> {
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