#!/usr/bin/env python3

import py_move_analyer

# Test source code
test_source = '''
module my_addr::main_model {
    public fun init() {
    }
    
    public entry fun transfer(amount: u64) {
    }
    
    struct Coin has key, store {
        value: u64,
        owner: address,
    }
    
    struct Treasury has key {
        total_supply: u64,
    }
}
'''

print("=== Basic Parse Test ===")
f = py_move_analyer.parse(test_source)
print(type(f))
print(f)

print("\n=== Symbol Extraction Test ===")
extractor = py_move_analyer.extract_symbols(test_source)
print(f"Extractor: {extractor}")
print(f"Modules: {extractor.modules}")

print("\n=== Functions ===")
for func in extractor.functions:
    print(f"Function: {func.name}")
    print(f"  Module: {func.module}")
    print(f"  Visibility: {func.visibility}")
    print(f"  Parameters: {func.parameters}")
    print(f"  Return Type: {func.return_type}")
    print(f"  Is Entry: {func.is_entry}")
    print(f"  Is Native: {func.is_native}")
    print()

print("=== Structs ===")
for struct in extractor.structs:
    print(f"Struct: {struct.name}")
    print(f"  Module: {struct.module}")
    print(f"  Fields: {struct.fields}")
    print(f"  Abilities: {struct.abilities}")
    print(f"  Is Native: {struct.is_native}")
    print()

print("=== Query Tests ===")
print(f"Functions named 'init': {len(extractor.get_functions_by_name('init'))}")
print(f"Functions named 'transfer': {len(extractor.get_functions_by_name('transfer'))}")
print(f"Structs named 'Coin': {len(extractor.get_structs_by_name('Coin'))}")
print(f"Functions in module 'my_addr::main_model': {len(extractor.get_functions_by_module('my_addr::main_model'))}")
print(f"Structs in module 'my_addr::main_model': {len(extractor.get_structs_by_module('my_addr::main_model'))}")

print("\n=== Project Analysis Test ===")
# Test project analysis with test directory
try:
    project = py_move_analyer.analyze_project("./test")
    print(f"Project: {project}")
    print(f"Module names: {project.get_module_names()}")
    
    # Test global queries
    all_init_funcs = project.find_function("init")
    print(f"All 'init' functions across project: {len(all_init_funcs)}")
    
    for func in all_init_funcs:
        print(f"  - {func.name} in {func.module}")
    
except Exception as e:
    print(f"Project analysis failed (expected if no test directory): {e}")

print("\n=== Single File Analysis Test ===")
# Test with single file analysis
try:
    import tempfile
    import os
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.move', delete=False) as f:
        f.write(test_source)
        temp_file = f.name
    
    project = py_move_analyer.analyze_files([temp_file])
    print(f"Single file project: {project}")
    print(f"Modules: {project.get_module_names()}")
    
    # Clean up
    os.unlink(temp_file)
    
except Exception as e:
    print(f"Single file analysis failed: {e}")