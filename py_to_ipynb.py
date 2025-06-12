import ast
import os
import nbformat
import re

def extract_imported_steps(py_file):
    with open(py_file, "r") as f:
        tree = ast.parse(f.read())
    imported_steps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Solo módulos propios (pipeline, pipeline.steps, etc.)
            if node.module.startswith("pipeline"):
                for n in node.names:
                    imported_steps.append(n.name)
    return imported_steps

def find_step_source(step_name, search_dirs):
    # Busca el archivo que contiene la clase step_name en los directorios dados
    for steps_dir in search_dirs:
        for root, _, files in os.walk(steps_dir):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    with open(fpath) as f:
                        code = f.read()
                        # Busca la definición de la clase o función
                        if re.search(rf"class {step_name}\b", code) or re.search(rf"def {step_name}\b", code):
                            return code
    return None

def remove_imports(code):
    # Elimina líneas de import propios (pipeline, pipeline.steps, etc.)
    lines = code.splitlines()
    filtered = [l for l in lines if not re.match(r"from pipeline|import pipeline", l)]
    return "\n".join(filtered)

def py_to_ipynb(py_file, search_dirs=["pipeline/steps", "pipeline"], output="notebook_autocontenido.ipynb"):
    imported_steps = extract_imported_steps(py_file)
    step_sources = []
    already_added = set()
    for step in imported_steps:
        if step in already_added:
            continue
        src = find_step_source(step, search_dirs)
        if src:
            step_sources.append(f"# {step}\n{remove_imports(src)}")
            already_added.add(step)
    # Lee el código principal y elimina imports propios
    with open(py_file) as f:
        main_code = remove_imports(f.read())
    # Arma el notebook
    nb = nbformat.v4.new_notebook()
    # Celda con los steps
    if step_sources:
        nb.cells.append(nbformat.v4.new_code_cell("\n\n".join(step_sources)))
    # Celda con el código principal
    nb.cells.append(nbformat.v4.new_code_cell(main_code))
    # Guarda
    with open(output, "w") as f:
        nbformat.write(nb, f)
    print(f"Notebook guardado en {output}")

if __name__ == "__main__":
    py_to_ipynb("notebook.py")