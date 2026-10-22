#!/usr/bin/env python3
"""Build Jupyter notebooks from lightweight Markdown sources.

Source format (src/*.md):
  - Plain Markdown -> markdown cells. A line containing only `+++` forces a cell break.
  - ```python ... ```            -> code cell, executed here if it does not need torch & co.
  - ```python noexec ... ```     -> code cell, never executed here.
  - ```python exec ... ```       -> code cell, always executed (override auto-detection).

Execution: cells run sequentially in one namespace; stdout and matplotlib figures
are captured and embedded as notebook outputs. Cells that need PyTorch / HF
libraries are only statically checked (syntax + undefined names), because this
build environment has no PyTorch.
"""
import ast
import base64
import builtins
import contextlib
import io
import json
import re
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FENCE_RE = re.compile(r"^```python([^\n]*)\n(.*?)^```\s*$", re.S | re.M)
NEEDS_TORCH_RE = re.compile(
    r"\b(import torch|from torch|torch\.|nn\.|F\.|DataLoader|optim\.|"
    r"(?:from|import) (?:transformers|datasets|peft|tokenizers|evaluate|accelerate|seqeval|trl|gensim)\b|load_dataset)"
)


def parse_source(text):
    cells = []
    pos = 0
    for m in FENCE_RE.finditer(text):
        md = text[pos:m.start()]
        cells.extend(split_markdown(md))
        flag = m.group(1).strip()
        cells.append({"type": "code", "flag": flag, "source": m.group(2).rstrip("\n")})
        pos = m.end()
    cells.extend(split_markdown(text[pos:]))
    return cells


def split_markdown(md):
    out = []
    for chunk in re.split(r"^\+\+\+\s*$", md, flags=re.M):
        chunk = chunk.strip("\n")
        if chunk.strip():
            out.append({"type": "markdown", "source": chunk})
    return out


def flag_tokens(cell):
    return cell.get("flag", "").split()


def export_target(cell):
    for t in flag_tokens(cell):
        if t.startswith("export="):
            return t.split("=", 1)[1]
    return None


def should_exec(cell):
    if "noexec" in flag_tokens(cell):
        return False
    if "exec" in flag_tokens(cell):
        return True
    src = cell["source"]
    if src.lstrip().startswith(("%", "!")):
        return False
    return not NEEDS_TORCH_RE.search(src)


def strip_magics(src):
    return "\n".join("# " + l if l.lstrip().startswith(("%", "!")) else l for l in src.splitlines())


class NameCollector(ast.NodeVisitor):
    """Very small undefined-name checker (module-level, order-insensitive)."""

    def __init__(self):
        self.defined = set()
        self.used = []

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.defined.add(node.id)
        else:
            self.used.append((node.id, node.lineno))

    def visit_FunctionDef(self, node):
        self.defined.add(node.name)
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            self.defined.add(a.arg)
        if node.args.vararg:
            self.defined.add(node.args.vararg.arg)
        if node.args.kwarg:
            self.defined.add(node.args.kwarg.arg)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for a in node.args.args + node.args.kwonlyargs:
            self.defined.add(a.arg)
        if node.args.vararg:
            self.defined.add(node.args.vararg.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for a in node.names:
            self.defined.add((a.asname or a.name).split(".")[0])

    def visit_ImportFrom(self, node):
        for a in node.names:
            self.defined.add(a.asname or a.name)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.defined.add(node.name)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.defined.add(node.arg)

    def visit_alias(self, node):
        pass

    def visit_Global(self, node):
        self.defined.update(node.names)

    def visit_MatchAs(self, node):
        if node.name:
            self.defined.add(node.name)
        self.generic_visit(node)


def static_check(cells, name):
    errors = []
    coll = NameCollector()
    for i, c in enumerate(cells):
        if c["type"] != "code":
            continue
        src = strip_magics(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            errors.append(f"[{name}] cell {i}: SyntaxError {e}")
            continue
        coll.visit(tree)
    known = coll.defined | set(dir(builtins)) | {"__name__", "display", "get_ipython"}
    seen = set()
    for n, ln in coll.used:
        if n not in known and n not in seen:
            seen.add(n)
            errors.append(f"[{name}] undefined name: {n!r}")
    return errors


def _top_names(src):
    """(module-level names defined, free names loaded) for a cell. Rough but enough to track torch-only variables."""
    try:
        tree = ast.parse(strip_magics(src))
    except SyntaxError:
        return set(), set()
    top_defined, any_stored, loaded = set(), set(), set()

    def walk_top(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                top_defined.add(child.name)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                top_defined.add(child.id)
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                for al in child.names:
                    top_defined.add((al.asname or al.name).split(".")[0])
            walk_top(child)

    walk_top(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (any_stored if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
        elif isinstance(node, ast.arg):
            any_stored.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            any_stored.add(node.name)
    return top_defined, loaded - any_stored


_DISPLAY_SINK = []


def _install_ipython_stub():
    """Minimal IPython.display so notebooks can show GIF animations when built without Jupyter."""
    try:
        import IPython.display  # noqa: F401
        return
    except ImportError:
        pass
    import types
    ipy, disp = types.ModuleType("IPython"), types.ModuleType("IPython.display")

    class Image:
        def __init__(self, data=None, filename=None, format=None, **kw):
            self.data = Path(filename).read_bytes() if filename else data
            self.format = format or (Path(filename).suffix.lstrip(".") if filename else "png")

    def display(*objs, **kw):
        for o in objs:
            if isinstance(o, Image):
                fmt = "gif" if o.format.lower() == "gif" else "png"
                _DISPLAY_SINK.append({"output_type": "display_data", "metadata": {},
                                      "data": {f"image/{fmt}": base64.b64encode(o.data).decode(), "text/plain": "<Image>"}})
            else:
                print(repr(o))

    disp.Image, disp.display = Image, display
    ipy.display = disp
    ipy.get_ipython = lambda: None              # matplotlib probes these two attributes
    ipy.version_info = (9, 0, 0)
    sys.modules["IPython"], sys.modules["IPython.display"] = ipy, disp


def run_cells(cells, name, workdir):
    _install_ipython_stub()
    ns = {"__name__": "__main__"}
    torch_only = set()
    plt.show = lambda *a, **k: None
    import os
    old = os.getcwd()
    os.chdir(workdir)
    exec_count = 0
    try:
        for i, c in enumerate(cells):
            if c["type"] != "code":
                continue
            c["outputs"] = []
            c["exec_count"] = None
            defined, loaded = _top_names(c["source"])
            if not should_exec(c) or ("exec" not in flag_tokens(c) and loaded & torch_only):
                c["executed"] = False
                torch_only |= defined
                continue
            torch_only -= defined
            c["executed"] = True
            exec_count += 1
            c["exec_count"] = exec_count
            buf = io.StringIO()
            t0 = time.time()
            plt.close("all")
            try:
                with contextlib.redirect_stdout(buf):
                    tree = ast.parse(c["source"])
                    last_expr = None
                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = ast.Expression(tree.body.pop().value)
                    exec(compile(tree, f"<{name}:cell{i}>", "exec"), ns)
                    if last_expr is not None:
                        val = eval(compile(last_expr, f"<{name}:cell{i}>", "eval"), ns)
                        if val is not None and not _is_mpl(val):
                            print(repr(val))
            except Exception:
                sys.stderr.write(f"\n### ERROR in {name} cell {i}\n{c['source'][:800]}\n")
                traceback.print_exc()
                raise
            dt = time.time() - t0
            text = buf.getvalue()
            if text:
                c["outputs"].append({"output_type": "stream", "name": "stdout", "text": text})
            c["outputs"].extend(_DISPLAY_SINK); _DISPLAY_SINK.clear()
            for num in plt.get_fignums():
                fig = plt.figure(num)
                png = io.BytesIO()
                fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
                c["outputs"].append({
                    "output_type": "display_data",
                    "data": {"image/png": base64.b64encode(png.getvalue()).decode(),
                             "text/plain": "<Figure>"},
                    "metadata": {},
                })
            plt.close("all")
            print(f"  cell {i:3d} ok  {dt:6.2f}s  figs={sum(o['output_type']=='display_data' for o in c['outputs'])}")
    finally:
        os.chdir(old)


def _is_mpl(v):
    mod = type(v).__module__ or ""
    return mod.startswith("matplotlib") or (isinstance(v, (list, tuple)) and v and (type(v[0]).__module__ or "").startswith("matplotlib"))


def to_ipynb(cells):
    nb_cells = []
    for c in cells:
        lines = c["source"].splitlines(keepends=True)
        if c["type"] == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            outs = []
            for o in c.get("outputs", []):
                if o["output_type"] == "stream":
                    outs.append({"output_type": "stream", "name": "stdout",
                                 "text": o["text"].splitlines(keepends=True)})
                else:
                    outs.append(o)
            nb_cells.append({
                "cell_type": "code", "metadata": {}, "execution_count": c.get("exec_count"),
                "outputs": outs, "source": lines,
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU", "colab": {"provenance": []},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


EXPORT_ROOT = Path(__file__).resolve().parent.parent / "projet_protannotml"


def export_cells(cells, name):
    """Cells flagged `export=pkg/file.py` are concatenated into that file (nbdev-style)."""
    targets = {}
    for c in cells:
        if c["type"] == "code" and export_target(c):
            targets.setdefault(export_target(c), []).append(c["source"])
    for rel, sources in targets.items():
        path = EXPORT_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f'"""Exported from notebook {name}.ipynb — edit the notebook source (src/{name}.md), not this file."""\n\n'
        path.write_text(header + "\n\n\n".join(src.strip("\n") for src in sources) + "\n", encoding="utf-8")
        print(f"  exported {len(sources)} cell(s) -> {path.relative_to(EXPORT_ROOT.parent)}")


def build(src_path, out_dir, execute=True):
    src_path = Path(src_path)
    cells = parse_source(src_path.read_text(encoding="utf-8"))
    name = src_path.stem
    export_cells(cells, name)
    errs = static_check(cells, name)
    for e in errs:
        print("STATIC:", e)
    if execute:
        run_cells(cells, name, out_dir)
    nb = to_ipynb(cells)
    out = Path(out_dir) / f"{name}.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(c["type"] == "code" for c in cells)
    n_exec = sum(c.get("executed", False) for c in cells)
    print(f"Built {out}  ({len(cells)} cells, {n_code} code, {n_exec} executed here, {len(errs)} static warnings)")
    # also dump a json sidecar so the HTML reader can know which cells ran
    return out, errs


if __name__ == "__main__":
    args = sys.argv[1:]
    execute = "--no-exec" not in args
    args = [a for a in args if not a.startswith("--")]
    out_dir = Path(__file__).resolve().parent.parent / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for a in args:
        build(a, out_dir, execute=execute)
