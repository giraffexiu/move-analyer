"""Python type stubs for py_move_analyer module.

This module provides Move language parsing functionality implemented in Rust.
"""

from typing import Union

def parse(content: str) -> str:
    """Parse Move source code and return the result as a string.
    
    Args:
        content: The Move source code to parse as a string.
        
    Returns:
        A string representation of the parsed AST or error message.
        
    Example:
        >>> import py_move_analyer
        >>> result = py_move_analyer.parse("module 0x1::test { }")
        >>> print(result)
    """
    ...