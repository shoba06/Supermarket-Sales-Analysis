import json, ast

with open("Supermarket_Sales_Analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Valid JSON: True")
print(f"nbformat : {nb['nbformat']}")
print(f"Total cells: {len(nb['cells'])}")

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
md_cells   = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
print(f"Code cells     : {len(code_cells)}")
print(f"Markdown cells : {len(md_cells)}")

errors = []
for i, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i+1, str(e)))
        print(f"SYNTAX ERROR in code cell {i+1}: {e}")
        print(src[:300])

if not errors:
    print(f"\nAll {len(code_cells)} code cells have valid Python syntax.")
else:
    print(f"\n{len(errors)} syntax error(s) found.")
