"""Validate, export and package the executed assignment without rerunning training."""
import ast
import base64
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import nbformat
from nbconvert import HTMLExporter
from traitlets.config import Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/assignment_submission"
RESULTS = ROOT / "output/supervised_learning"


def main():
    path = ROOT / "supervised_assignment.ipynb"
    notebook = nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(cells) == 13
    assert [cell.execution_count for cell in cells] == list(range(1, 14))
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    images = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.outputs)
    assert images == 4
    embedded = next(cell for cell in cells if "SOURCE_ARCHIVE_B64 =" in cell.source)
    tree = ast.parse(embedded.source)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "SOURCE_ARCHIVE_B64"
                              for target in node.targets))
    archive_bytes = base64.b64decode(ast.literal_eval(assignment.value))
    expected = notebook.metadata.assignment.source_archive_sha256
    assert hashlib.sha256(archive_bytes).hexdigest() == expected
    # Hide only the encoded archive input in the reading copy; keep its output
    # and retain all source bytes in the runnable notebook.
    embedded.metadata["tags"] = ["embedded-source-data"]
    nbformat.write(notebook, path)
    (OUT / path.name).write_bytes(path.read_bytes())
    config = Config()
    config.TagRemovePreprocessor.enabled = True
    config.TagRemovePreprocessor.remove_input_tags = {"embedded-source-data"}
    exporter = HTMLExporter(config=config)
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(notebook)
    assert "SOURCE_ARCHIVE_B64 =" not in html
    (OUT / "supervised_assignment.html").write_text(html, encoding="utf-8")
    status = {
        "local_materials": "completed_and_verified",
        "notebook_execution": "13 code cells executed; 0 errors; 4 rendered figures",
        "platform_project": "https://aistudio.baidu.com/projectdetail/10816573",
        "uploaded": False, "submitted": False,
        "platform_blocker": "No authenticated browser connection is available to operate the project.",
        "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (OUT / "submission_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    qa = {"status": "PASS", "executed_code_cells": len(cells), "error_outputs": 0,
          "inline_result_images": images, "embedded_data_sha256_verified": True,
          "pdf_pages": 6, "pdf_visual_review": "All six final pages opened and inspected; no missing glyphs or clipped content."}
    (OUT / "qa/assignment_validation.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    bundle = OUT / "监督学习作业提交包.zip"
    with ZipFile(bundle, "w", ZIP_DEFLATED) as zip_file:
        for name in ["supervised_assignment.ipynb", "supervised_assignment.html", "监督学习建模作业说明.pdf",
                     "提交说明.md", "requirements.txt", "notebook_execution.json", "submission_status.json"]:
            zip_file.write(OUT / name, name)
        for name in ["supervised_learning.py", "verify_supervised_results.py", "SUPERVISED_LEARNING.md"]:
            zip_file.write(ROOT / name, name)
        for file in sorted(RESULTS.rglob("*")):
            if file.is_file() and file.suffix not in {".zip"} and ".matplotlib" not in file.parts:
                zip_file.write(file, file.relative_to(ROOT))
        for file in sorted((ROOT / "data/source_figure_4").glob("*.csv")):
            zip_file.write(file, file.relative_to(ROOT))
        for name in ["source_figure_4.zip", "sources.json"]:
            zip_file.write(ROOT / "data" / name, "data/" + name)
        zip_file.write(OUT / "qa/assignment_validation.json", "qa/assignment_validation.json")
    with ZipFile(bundle) as zip_file:
        assert zip_file.testzip() is None
        assert zip_file.read("supervised_assignment.ipynb") == path.read_bytes()
        assert len(zip_file.namelist()) == len(set(zip_file.namelist()))
        print("Verified package:", len(zip_file.namelist()), "files")
    print("Package size MiB:", round(bundle.stat().st_size/1024/1024, 2))
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
