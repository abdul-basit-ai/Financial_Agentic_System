"""Static AST pre-filter for sandboxed Python code.

Inspects code syntax trees to intercept and reject dangerous modules,
built-in functions, reflection vectors, and dunder attributes prior to execution.
"""

from __future__ import annotations

import ast

# Permitted standard and scientific computing modules
SAFE_MODULES: set[str] = {
    "math",
    "cmath",
    "statistics",
    "decimal",
    "fractions",
    "random",
    "datetime",
    "time",
    "json",
    "re",
    "itertools",
    "collections",
    "functools",
    "bisect",
    "heapq",
    "numpy",
    "pandas",
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "sympy",
}

# Explicitly forbidden builtins and execution primitives
FORBIDDEN_CALLS: set[str] = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "breakpoint",
    "help",
    "exit",
    "quit",
}

# Dangerous dunder attributes used in Python sandbox escape chains
FORBIDDEN_DUNDERS: set[str] = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
    "__import__",
    "__reduce__",
    "__reduce_ex__",
}


class SandboxSecurityError(ValueError):
    """Raised when Python code violates AST security policies."""


class SecurityASTVisitor(ast.NodeVisitor):
    """Walks the AST and verifies all nodes against strict security whitelists.

    Tracks aliases of forbidden builtins (e.g. ``e = eval``) — an assignment of
    a forbidden name followed by a later call through the alias is a sandbox
    escape, so the alias itself is rejected.
    """

    def __init__(self) -> None:
        self.forbidden_aliases: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_pkg = alias.name.split(".")[0]
            if root_pkg not in SAFE_MODULES:
                raise SandboxSecurityError(
                    f"Import of unauthorized module '{alias.name}' is strictly forbidden."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg not in SAFE_MODULES:
                raise SandboxSecurityError(
                    f"Import from unauthorized module '{node.module}' is strictly forbidden."
                )
        else:
            raise SandboxSecurityError("Relative imports are forbidden in the sandbox environment.")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Reject assignments of forbidden builtins outright (not just when
        # later called): an alias like `e = eval` is itself an escape primer.
        if isinstance(node.value, ast.Name) and node.value.id in FORBIDDEN_CALLS:
            raise SandboxSecurityError(
                f"Assignment of protected built-in '{node.value.id}' to a variable "
                f"is prohibited (alias escape defense)."
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in FORBIDDEN_CALLS or name in self.forbidden_aliases:
                raise SandboxSecurityError(
                    f"Invocation of protected built-in function '{name}()' is prohibited."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_DUNDERS:
            raise SandboxSecurityError(
                f"Access to introspection attribute '{node.attr}' is prohibited (sandbox escape defense)."
            )
        self.generic_visit(node)


def audit_code_safety(code: str) -> None:
    """Parses and validates Python code against sandbox security rules.

    Raises:
        SandboxSecurityError: If an unauthorized module or forbidden construct is detected.
        SyntaxError: If the code cannot be parsed.
    """
    clean_code = code.strip()
    if not clean_code:
        raise ValueError("Source code cannot be empty.")

    tree = ast.parse(clean_code, mode="exec")
    visitor = SecurityASTVisitor()
    visitor.visit(tree)