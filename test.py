from py_move_analyer import parse

file = """
module my_addr::main_model;

fun init() {}
"""
f = parse(file)
print(type(f))
print(f)