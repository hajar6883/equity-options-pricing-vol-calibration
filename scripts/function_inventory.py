import ast
import os

PROJECT_ROOT = "."

def extract_functions(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]

def walk_project(root):
    inventory = {}
    for dirpath, _, filenames in os.walk(root):
        if any(skip in dirpath for skip in [".venv", "__pycache__", ".git"]):
            continue

        for file in filenames:
            if file.endswith(".py"):
                path = os.path.join(dirpath, file)
                funcs = extract_functions(path)
                if funcs:
                    inventory[path] = funcs
    return inventory

if __name__ == "__main__":
    inventory = walk_project(PROJECT_ROOT)
    for file, funcs in inventory.items():
        print(f"\n{file}")
        for f in funcs:
            print(f"  - {f}")
