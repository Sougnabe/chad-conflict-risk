"""
Generate notebooks/capstone_pipeline.ipynb from the pipeline scripts.

The scripts remain the single source of truth: this generator copies their code into a
notebook, one section per step, with an explanation before each step. It only adapts what a
notebook needs (project root without __file__, figures shown inline, optional steps behind a
flag, a fixed analysis unit instead of command-line arguments).

Usage:
    python scripts/make_notebook.py
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/capstone_pipeline.ipynb
"""

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT = ROOT / "notebooks" / "capstone_pipeline.ipynb"

ROOT_LINE = "ROOT = Path(__file__).resolve().parents[1]\n"

STEPS = [
    ("download_acled.py", "1. Download conflict events (ACLED API)", "RUN_DOWNLOAD"),
    ("build_target.py", "2. Validated labels and target variable", None),
    ("eda_acled.py", "3. Exploratory analysis", None),
    ("build_panel.py", "4. Department x month panel", None),
    ("init_db.py", "5. Application database", None),
    ("gee_extract.py", "6. Satellite indicators (Google Earth Engine)", "RUN_GEE"),
    ("build_features.py", "7. Feature table (no look-ahead)", None),
    ("train_models.py", "8. Research model: training, evaluation and SHAP (quarterly horizon)", None),
]

INTRO = """# Farmer-Herder Conflict Risk in Southern Chad — Full Pipeline

Capstone project, BSc Software Engineering (Machine Learning), African Leadership University.

This notebook contains the complete code of the project, step by step, with its outputs.
It is generated from the scripts in `scripts/` (by `scripts/make_notebook.py`), which remain
the reference implementation used by the dashboard and the tests.

**Research question.** Do satellite-derived indicators of pasture conditions, rainfall and the
crop calendar improve the prediction of farmer-herder conflict risk in southern Chad, compared
with conflict history alone?

**Steps**
1. Download conflict events (ACLED API) — *skipped by default, data already downloaded*
2. Validated labels and target variable
3. Exploratory analysis
4. Department x month panel
5. Application database
6. Satellite indicators (Google Earth Engine) — *skipped by default, indicators already extracted*
7. Feature table (no look-ahead)
8. Research model (conflict history up to the previous month), then the operational model
   (only information really available at forecast time: conflict history delayed by 12
   months, satellite data up to the previous month), which produces the current outlook
9. Automated tests
10. Dashboard code

> Ethics: aggregated department-level data only, no ethnic or religious variables. Raw ACLED
> data are not redistributed (licence). The tool supports human decisions; it does not make them.
"""

SETUP = """from pathlib import Path

# Project root (works whether the notebook is opened from notebooks/ or from the project root)
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

# Optional steps (they need credentials and take time; their outputs are already in data/)
RUN_DOWNLOAD = False   # step 1: re-download ACLED events (needs ACLED credentials in .env)
RUN_GEE = False        # step 6: re-extract satellite indicators (needs Earth Engine, ~25 min)
print("Project root:", ROOT)"""


def split_docstring(src):
    m = re.match(r'\s*"""(.*?)"""\n', src, re.S)
    if not m:
        return "", src
    doc = m.group(1).strip()
    doc = re.split(r"\n\s*Usage:", doc)[0].strip()  # usage lines are for the command line
    return doc, src[m.end():]


def adapt(name, code):
    assert ROOT_LINE in code, f"{name}: ROOT line not found"
    code = code.replace(ROOT_LINE, "")
    code = code.replace("import matplotlib\n\nmatplotlib.use(\"Agg\")\n", "")
    code = code.replace("plt.close()", "plt.show()")
    if name == "download_acled.py":
        code = code.replace('if __name__ == "__main__":\n    main()', "if RUN_DOWNLOAD:\n    main()\nelse:\n    print(\"Skipped: ACLED data already downloaded (set RUN_DOWNLOAD = True to re-run).\")")
    if name == "gee_extract.py":
        assert "parser.parse_args().end" in code
        code = code.replace("parser.parse_args().end", "parser.parse_args([]).end")
    if name == "train_models.py":
        code = code.replace("import argparse\n", "")
        code = re.sub(
            r'parser = argparse\.ArgumentParser\(\)\n.*?UNIT, MODE = ARGS\.unit, ARGS\.mode\n',
            'from types import SimpleNamespace\n\n'
            '# Research model, quarterly horizon (the other runs follow at the end of this step)\n'
            'ARGS = SimpleNamespace(unit="quarter", mode="research", fixed_model=None)\n'
            'UNIT, MODE = ARGS.unit, ARGS.mode\n',
            code, flags=re.S)
        assert "SimpleNamespace" in code, "train_models.py: argument block not replaced"
    return code


def split_cells(code):
    """Split at top-level comment headers that follow a blank line."""
    lines = code.strip("\n").split("\n")
    cells, cur = [], []
    for i, line in enumerate(lines):
        header = line.startswith("# ") and i > 0 and lines[i - 1].strip() == ""
        if header and any(l.strip() for l in cur):
            cells.append("\n".join(cur).strip("\n"))
            cur = []
        cur.append(line)
    if any(l.strip() for l in cur):
        cells.append("\n".join(cur).strip("\n"))
    return cells


nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP)]

for script, title, flag in STEPS:
    src = (SCRIPTS / script).read_text(encoding="utf8")
    doc, code = split_docstring(src)
    code = adapt(script, code)
    cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n```text\n{doc}\n```"
                                          if doc else f"## {title}"))
    if flag == "RUN_GEE":
        body = "\n".join("    " + l if l.strip() else "" for l in code.strip("\n").split("\n"))
        cells.append(nbf.v4.new_code_cell(
            f"if RUN_GEE:\n{body}\nelse:\n    print(\"Skipped: satellite indicators already in "
            f"data/processed/satellite_features.csv (set RUN_GEE = True to re-extract).\")"))
    else:
        cells += [nbf.v4.new_code_cell(c) for c in split_cells(code)]

RUN = ("import subprocess, sys\n"
       "r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'train_models.py'), {args}],\n"
       "                   capture_output=True, text=True, cwd=ROOT)\n"
       "print(r.stdout[-4000:] or r.stderr[-4000:])")
cells.append(nbf.v4.new_markdown_cell(
    "### Research model, monthly horizon\n\nThe same code for the next-month horizon "
    "(results in `reports/metrics_research_month.csv`)."))
cells.append(nbf.v4.new_code_cell(RUN.format(args="'--mode', 'research', '--unit', 'month'")))
cells.append(nbf.v4.new_markdown_cell(
    "## 8b. Operational model and current outlook\n\n"
    "ACLED Research-tier data are published with a 12-month delay. The operational model "
    "therefore uses only what is really available when a forecast is issued: conflict history "
    "up to month t-13 (`o_` features), satellite data up to month t-1 and the season. It is "
    "trained and evaluated exactly like the research model, then produces a genuine forecast "
    "for the quarter after the latest satellite month.\n\n"
    "The dashboard uses logistic regression with satellite features (`logreg_OB`), fixed by "
    "design: an early-warning tool must react to current conditions, and logistic regression "
    "is the most stable model with so few positive cases. The automatic selection is still "
    "reported (`reports/selection_operational_quarter.json`)."))
cells.append(nbf.v4.new_code_cell(RUN.format(
    args="'--mode', 'operational', '--unit', 'quarter', '--fixed-model', 'logreg_OB'")))
cells.append(nbf.v4.new_code_cell(RUN.format(args="'--mode', 'operational', '--unit', 'month'")))
cells.append(nbf.v4.new_markdown_cell("### Current outlook (operational model)"))
cells.append(nbf.v4.new_code_cell(
    "import sqlite3\n"
    "import pandas as pd\n"
    "with sqlite3.connect(ROOT / 'data' / 'app.db') as con:\n"
    "    outlook = pd.read_sql(\"SELECT p.ym, d.department, d.province, p.risk FROM predictions p \"\n"
    "                          \"JOIN departments d USING (dept_id) \"\n"
    "                          \"WHERE p.model LIKE '%operational_quarter_forecast'\", con)\n"
    "outlook = outlook.sort_values('risk', ascending=False).reset_index(drop=True)\n"
    "outlook.index += 1\n"
    "print('Quarter starting', outlook['ym'].iloc[0])\n"
    "outlook.head(10)"))

cells.append(nbf.v4.new_markdown_cell(
    "## 9. Automated tests\n\nTwelve tests check label and panel consistency, the absence of "
    "look-ahead in every feature, the absence of outcome leakage, the stored predictions and "
    "the dashboard (`tests/test_pipeline.py`)."))
cells.append(nbf.v4.new_code_cell(
    "r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '--color=no', '-p', 'no:cacheprovider', str(ROOT / 'tests')],\n"
    "                   capture_output=True, text=True, cwd=ROOT)\n"
    "print(r.stdout[-3000:])"))

app_src = (ROOT / "app" / "app.py").read_text(encoding="utf8")
cells.append(nbf.v4.new_markdown_cell(
    "## 10. Dashboard code\n\nThe dashboard is a Streamlit application and runs outside the "
    "notebook:\n\n```\nstreamlit run app/app.py\n```\n\nIts full code is reproduced below "
    "for reference.\n\n```python\n" + app_src.strip() + "\n```"))

nb.cells = cells
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"{OUT}: {len(cells)} cells")
