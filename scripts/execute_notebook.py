"""Execute and save the analysis notebook using the current project's Python."""

import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

path = Path("notebooks/01_urbanflow_analysis.ipynb")
notebook = nbformat.read(path, as_version=4)
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(
    notebook,
    km=manager,
    timeout=180,
    kernel_name="python3",
    resources={"metadata": {"path": str(Path.cwd())}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} cells without errors.")
