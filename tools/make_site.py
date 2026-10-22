#!/usr/bin/env python3
"""Render all course notebooks into one self-contained, theme-aware HTML reader."""
import html
import json
import re
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT = ROOT / "site" / "index.html"

MODULES = [
    ("c00", "c00_maths_a_la_main", "Les maths à la main", "3 h 30"),
    ("m00", "m00_fondements", "Fondements des réseaux de neurones", "3–4 h"),
    ("m01", "m01_pytorch", "PyTorch en pratique", "2–3 h"),
    ("m02", "m02_texte_vers_nombres", "Du texte aux nombres", "2 h 30"),
    ("m03", "m03_embeddings", "Les word embeddings", "2 h 30"),
    ("m04", "m04_rnn", "Modèles de langue et RNN", "3 h"),
    ("m05", "m05_seq2seq_attention", "Seq2seq et attention", "2 h 30"),
    ("m06", "m06_transformer", "Le Transformer de zéro", "4 h"),
    ("m07", "m07_pretraining_hf", "Pré-entraînement et Hugging Face", "3 h"),
    ("m08", "m08_adaptation", "Adaptation et évaluation", "3 h 30"),
    ("m09", "m09_protein_lm", "Modèles de langage de protéines", "3 h"),
    ("m10", "m10_ner", "NER en profondeur", "3 h 30"),
    ("m11", "m11_contrastive", "Apprentissage contrastif", "3 h"),
    ("m12", "m12_distillation", "Distillation de connaissances", "2 h 30"),
    ("m13", "m13_multitask", "Multi-tâche en profondeur", "3 h 30"),
    ("m14", "m14_llm_finetuning", "Affiner un LLM (SFT, LoRA, DPO)", "4 h"),
    ("m15", "m15_agents_tools_mcp", "Agents, outils et MCP", "4 h"),
    ("m16", "m16_rag", "RAG : recherche documentaire", "4 h"),
    ("m17", "m17_kg_rag", "KG-RAG : RDF, SPARQL, Jena Fuseki", "4 h"),
    ("m18", "m18_multiagents_langgraph", "Multi-agents : LangChain, LangGraph", "3 h 30"),
    ("parcours", "parcours_protannotml", "Parcours raccourci ProtAnnotML", "lecture"),
    ("pf01", "pf01_donnees_imgt", "Fil rouge 1 · Données IMGT", "2 h 30"),
    ("pf02", "pf02_splits_homologie", "Fil rouge 2 · Splits par homologie", "1 h 30"),
    ("pf03", "pf03_baselines", "Fil rouge 3 · Alignement et décodage", "2 h 30"),
    ("pf04", "pf04_modele_plm", "Fil rouge 4 · Modèle PLM multi-tâche", "3 h"),
    ("pf05", "pf05_evaluation", "Fil rouge 5 · Évaluation statistique", "2 h"),
    ("pf06", "pf06_echelle_deploiement", "Fil rouge 6 · Jean Zay et déploiement", "2 h"),
    ("m19", "m19_interpretabilite", "Interprétabilité des PLM", "3 h"),
    ("m20", "m20_proteines_structure", "Protéines et structure", "3 h"),
    ("m21", "m21_kg_rag_imgt_mabkg", "KG-RAG sur IMGT/mAb-KG", "4 h"),
    ("lectures", "lectures_commentees", "Liste de lecture commentée", "20–30 h"),
]

FENCE = re.compile(r"(^```[^\n]*\n.*?^```[ \t]*$)", re.S | re.M)
INLINE_CODE = re.compile(r"(`[^`\n]+`)")
MATH = re.compile(r"(\$\$.+?\$\$|\$[^\$\n]+?\$)", re.S)


def md_to_html(src):
    """Markdown -> HTML while protecting LaTeX from the Markdown parser."""
    store = []

    def keep(text):
        store.append(text)
        return f"\x00{len(store) - 1}\x00"

    parts = FENCE.split(src)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:                      # fenced block: leave it to markdown
            out.append(part)
            continue
        sub = INLINE_CODE.split(part)
        for j, s in enumerate(sub):
            if j % 2 == 1:
                out.append(s)
            else:
                out.append(MATH.sub(lambda m: keep(html.escape(m.group(0), quote=False)), s))
    text = "".join(out)
    text = re.sub(r"<details>", '<details markdown="1">', text)
    rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html", "sane_lists"])
    rendered = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], rendered)
    rendered = re.sub(r"<thead>\s*<tr>\s*(?:<th[^>]*>\s*</th>\s*)+</tr>\s*</thead>", "", rendered)   # drop empty headers
    rendered = rendered.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    return rendered


FORMATTER = HtmlFormatter(nowrap=True)


def code_html(src, executed):
    body = highlight(src, PythonLexer(), FORMATTER)
    if executed:
        badge = '<span class="badge run">exécuté ici · NumPy</span>'
    else:
        badge = '<span class="badge torch">PyTorch / HF · à exécuter chez toi ou sur Colab</span>'
    return f'<div class="cell"><div class="cellbar">{badge}</div><pre class="code"><code>{body}</code></pre></div>'


def outputs_html(outputs):
    chunks = []
    for o in outputs:
        if o["output_type"] == "stream":
            txt = "".join(o["text"])
            chunks.append(f'<pre class="out">{html.escape(txt)}</pre>')
        elif "image/gif" in o.get("data", {}):
            chunks.append(f'<figure class="fig"><img loading="lazy" alt="Animation générée par la cellule précédente" '
                          f'src="data:image/gif;base64,{o["data"]["image/gif"]}"></figure>')
        elif "image/png" in o.get("data", {}):
            chunks.append(f'<figure class="fig"><img loading="lazy" alt="Figure générée par la cellule précédente" '
                          f'src="data:image/png;base64,{o["data"]["image/png"]}"></figure>')
    return "".join(chunks)


def short_label(key):
    """Sidebar number: 00-21 for modules, F1-F6 for the capstone, symbols for the extra pages."""
    if re.fullmatch(r"m\d+", key):
        return f"{int(key[1:]):02d}"
    if key.startswith("pf"):
        return f"F{int(key[2:])}"
    return {"parcours": "→", "lectures": "≡", "c00": "∂"}.get(key, "·")


def long_label(key):
    if re.fullmatch(r"m\d+", key):
        return f"Module {int(key[1:])}"
    if key.startswith("pf"):
        return f"Fil rouge ProtAnnotML · {int(key[2:])}"
    return {"parcours": "Parcours", "lectures": "Lectures", "c00": "Chapitre 0"}.get(key, key)


def render_module(key, stem, title, duration, idx):
    nb = json.loads((NB_DIR / f"{stem}.ipynb").read_text(encoding="utf-8"))
    parts = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            parts.append(md_to_html(src))
        else:
            executed = c.get("execution_count") is not None
            parts.append(code_html(src, executed))
            parts.append(outputs_html(c.get("outputs", [])))
    prev_btn = (f'<a class="navbtn" href="#{MODULES[idx - 1][0]}">← {MODULES[idx - 1][2]}</a>' if idx > 0 else "<span></span>")
    next_btn = (f'<a class="navbtn next" href="#{MODULES[idx + 1][0]}">{MODULES[idx + 1][2]} →</a>' if idx < len(MODULES) - 1 else "<span></span>")
    return (f'<article class="module" id="view-{key}" data-key="{key}" hidden>'
            f'<div class="modmeta"><span class="modnum">{long_label(key)}</span><span class="dur">{duration}</span>'
            f'<span class="nbfile">{stem}.ipynb</span></div>'
            f'{"".join(parts)}<nav class="pager">{prev_btn}{next_btn}</nav></article>')


def build():
    modules_html = "".join(render_module(k, s, t, d, i) for i, (k, s, t, d) in enumerate(MODULES))
    nav_items = "".join(
        f'<li><a href="#{k}" data-key="{k}"><span class="n">{short_label(k)}</span><span class="t">{t}</span>'
        f'<span class="d">{d}</span></a></li>' for i, (k, s, t, d) in enumerate(MODULES))
    options = "".join(f'<option value="{k}">{short_label(k)} · {t}</option>' for i, (k, s, t, d) in enumerate(MODULES))
    page = TEMPLATE.replace("{{NAV}}", nav_items).replace("{{OPTIONS}}", options).replace("{{MODULES}}", modules_html)
    OUT.parent.mkdir(exist_ok=True)
    # the chapter-0 lab ships next to the site (site/labo.html): link it relatively so it works on any host
    page = page.replace("https://claude.ai/artifact/DVmnXWxXSnQk8DBo5oRjR8", "labo.html")
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


TEMPLATE = r"""<title>Cours Deep Learning NLP</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: lab-notebook reader. Fixed module rail on the left (desktop), sticky module picker on top (phone), one reading column. */
:root {
  --paper: #f6f7f9;
  --sheet: #ffffff;
  --ink: #17202c;
  --ink-2: #4a5566;
  --ink-3: #7a8494;
  --rule: #dfe3ea;
  --accent: #1f6fd1;
  --accent-soft: #e6effb;
  --torch: #c2531f;
  --torch-soft: #fbece4;
  --run: #137a55;
  --run-soft: #e3f4ec;
  --code-bg: #f1f3f7;
  --tok-k: #7b3fc4; --tok-s: #1d7a4c; --tok-c: #7a8494; --tok-n: #b4530e; --tok-f: #1f6fd1; --tok-b: #9b2c5a;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "Literata", Georgia, "Times New Roman", serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --rail: 300px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #11151b; --sheet: #161b23; --ink: #e6e9ef; --ink-2: #b3bac6; --ink-3: #8790a0; --rule: #2a313d;
    --accent: #6aa7f0; --accent-soft: #1b2a3f; --torch: #f08a5d; --torch-soft: #3a2219; --run: #58c79a; --run-soft: #15302a;
    --code-bg: #1c222c;
    --tok-k: #c39af5; --tok-s: #7fd3a2; --tok-c: #8790a0; --tok-n: #f3a865; --tok-f: #7fb5f5; --tok-b: #f08fb4;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #11151b; --sheet: #161b23; --ink: #e6e9ef; --ink-2: #b3bac6; --ink-3: #8790a0; --rule: #2a313d;
  --accent: #6aa7f0; --accent-soft: #1b2a3f; --torch: #f08a5d; --torch-soft: #3a2219; --run: #58c79a; --run-soft: #15302a;
  --code-bg: #1c222c;
  --tok-k: #c39af5; --tok-s: #7fd3a2; --tok-c: #8790a0; --tok-n: #f3a865; --tok-f: #7fb5f5; --tok-b: #f08fb4;
  color-scheme: dark;
}
html { scroll-padding-top: 72px; }
body { background: var(--paper); color: var(--ink); font-family: var(--font-body); font-size: 17px; line-height: 1.62; }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--paper); }
a { color: var(--accent); }
a:focus-visible, select:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.shell { display: grid; grid-template-columns: var(--rail) minmax(0, 1fr); min-height: 100vh; }
.rail { position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; height: 100vh; overflow-y: auto;
  border-right: 1px solid var(--rule); padding-block: 28px 24px; padding-left: max(20px, env(safe-area-inset-left, 0px)); padding-right: 16px; }
.brand { font-family: var(--font-display); font-weight: 700; font-size: 1.35rem; line-height: 1.15; letter-spacing: 0.01em; margin: 0 0 6px; text-wrap: balance; }
.brand-sub { color: var(--ink-3); font-family: var(--font-display); font-size: 0.86rem; margin: 0 0 22px; }
.rail ol { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 2px; }
.rail a { display: grid; grid-template-columns: 2.1em 1fr; grid-template-rows: auto auto; column-gap: 6px; padding: 8px 10px; border-radius: 6px;
  color: var(--ink-2); text-decoration: none; font-family: var(--font-display); font-size: 0.98rem; line-height: 1.25; }
.rail a .n { grid-row: 1 / span 2; font-variant-numeric: tabular-nums; color: var(--ink-3); font-weight: 600; padding-top: 1px; }
.rail a .d { font-size: 0.78rem; color: var(--ink-3); letter-spacing: 0.02em; }
.rail a:hover { background: var(--accent-soft); color: var(--ink); }
.rail a[aria-current="page"] { background: var(--accent-soft); color: var(--ink); box-shadow: inset 3px 0 0 var(--accent); }
.rail a[aria-current="page"] .n { color: var(--accent); }
.legend { margin-top: 26px; border-top: 1px solid var(--rule); padding-top: 16px; font-family: var(--font-display); font-size: 0.84rem; color: var(--ink-2);
  display: flex; flex-direction: column; gap: 8px; }

.topbar { display: none; }
main { padding-block: 36px 80px; padding-left: max(28px, env(safe-area-inset-left, 0px)); padding-right: max(28px, env(safe-area-inset-right, 0px)); min-width: 0; }
.module { max-width: 50rem; margin: 0 auto; min-width: 0; }
.modmeta { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: baseline; font-family: var(--font-display); font-size: 0.82rem;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-3); margin-bottom: 6px; }
.modmeta .modnum { color: var(--accent); font-weight: 700; }
.modmeta .nbfile { text-transform: none; letter-spacing: 0; font-family: var(--font-mono); font-size: 0.78rem; }

h1, h2, h3, h4 { font-family: var(--font-display); line-height: 1.2; text-wrap: balance; color: var(--ink); }
h1 { font-size: 2.3rem; font-weight: 700; margin: 0 0 18px; letter-spacing: -0.005em; }
h2 { font-size: 1.6rem; font-weight: 700; margin: 2.4em 0 0.6em; padding-top: 0.9em; border-top: 1px solid var(--rule); }
h3 { font-size: 1.22rem; font-weight: 600; margin: 1.8em 0 0.4em; }
h4 { font-size: 1.05rem; font-weight: 600; margin: 1.4em 0 0.3em; }
p, li { max-width: 68ch; }
ul, ol { padding-left: 1.3em; }
li + li { margin-top: 0.25em; }
blockquote { margin: 1.2em 0; padding: 12px 16px; background: var(--sheet); border: 1px solid var(--rule); border-radius: 8px; color: var(--ink-2); }
blockquote p { margin: 0.35em 0; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2em 0; }
code { font-family: var(--font-mono); font-size: 0.84em; background: var(--code-bg); padding: 0.08em 0.32em; border-radius: 4px; }
pre code { background: none; padding: 0; font-size: inherit; }
pre { font-family: var(--font-mono); }
pre:not(.code):not(.out) { background: var(--code-bg); padding: 12px 14px; border-radius: 8px; overflow-x: auto; font-size: 0.8rem; line-height: 1.45; }

.tbl { overflow-x: auto; margin: 1.1em 0; border: 1px solid var(--rule); border-radius: 8px; background: var(--sheet); }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; line-height: 1.45; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-family: var(--font-display); font-weight: 600; color: var(--ink); background: var(--accent-soft); }
tr:last-child td { border-bottom: 0; }

.cell { margin: 1.2em 0 0; border: 1px solid var(--rule); border-radius: 10px; background: var(--code-bg); overflow: hidden; }
.cellbar { display: flex; justify-content: flex-end; padding: 6px 10px 0; }
.badge { font-family: var(--font-display); font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.badge.run { color: var(--run); background: var(--run-soft); }
.badge.torch { color: var(--torch); background: var(--torch-soft); }
pre.code { margin: 0; padding: 8px 16px 14px; overflow-x: auto; font-size: 0.78rem; line-height: 1.5; color: var(--ink); }
pre.out { margin: 0; padding: 10px 16px; overflow-x: auto; font-size: 0.76rem; line-height: 1.45; color: var(--ink-2);
  border: 1px solid var(--rule); border-top: 0; border-radius: 0 0 10px 10px; background: var(--sheet); white-space: pre; }
.fig { margin: 0; padding: 10px; border: 1px solid var(--rule); border-top: 0; border-radius: 0 0 10px 10px; background: #fcfcfb; overflow-x: auto; }
.fig img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.cell + pre.out, .cell + .fig { margin-top: 0; }
pre.out + .fig { border-radius: 0 0 10px 10px; }

.k, .kn, .kc, .kd, .ow { color: var(--tok-k); font-weight: 600; }
.s, .s1, .s2, .sa, .sd, .se, .si, .sh { color: var(--tok-s); }
.c, .c1, .cm, .ch { color: var(--tok-c); font-style: italic; }
.mi, .mf, .mh, .mo { color: var(--tok-n); }
.nf, .nc, .fm { color: var(--tok-f); font-weight: 600; }
.nb, .bp { color: var(--tok-b); }
.nd { color: var(--tok-n); }

details { margin: 1em 0; border: 1px solid var(--rule); border-radius: 8px; background: var(--sheet); padding: 6px 14px; }
summary { cursor: pointer; font-family: var(--font-display); font-weight: 600; color: var(--accent); padding: 4px 0; }
mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; max-width: 100%; padding: 4px 0; }

.pager { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-top: 3em; padding-top: 1.4em; border-top: 1px solid var(--rule); }
.navbtn { font-family: var(--font-display); font-weight: 600; text-decoration: none; padding: 10px 14px; border: 1px solid var(--rule); border-radius: 8px; background: var(--sheet); color: var(--ink); }
.navbtn:hover { border-color: var(--accent); color: var(--accent); }

@media (max-width: 900px) {
  .shell { grid-template-columns: minmax(0, 1fr); }
  .rail { display: none; }
  .topbar { display: flex; position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; gap: 10px; align-items: center;
    background: var(--paper); border-bottom: 1px solid var(--rule); padding-block: 10px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
  .topbar .brand { font-size: 1rem; margin: 0; white-space: nowrap; }
  .topbar select { flex: 1; min-width: 0; font-family: var(--font-display); font-size: 0.95rem; padding: 7px 8px; border-radius: 6px;
    border: 1px solid var(--rule); background: var(--sheet); color: var(--ink); }
  main { padding-block: 22px 60px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
  body { font-size: 16px; }
  h1 { font-size: 1.75rem; }
  h2 { font-size: 1.35rem; }
}
@media (prefers-reduced-motion: no-preference) { .navbtn, .rail a { transition: background-color .15s, color .15s, border-color .15s; } }
</style>

<div class="topbar">
  <p class="brand">Deep NLP</p>
  <label for="module-picker" class="sr" hidden>Module</label>
  <select id="module-picker" aria-label="Choisir un module">{{OPTIONS}}</select>
</div>
<div class="shell">
  <aside class="rail" aria-label="Modules du cours">
    <p class="brand">Deep Learning pour le NLP</p>
    <p class="brand-sub">De la rétropropagation aux modèles de langage de protéines</p>
    <nav><ol>{{NAV}}</ol></nav>
    <div class="legend">
      <span><span class="badge run">exécuté ici</span> parties NumPy, sorties incluses</span>
      <span><span class="badge torch">PyTorch / HF</span> à exécuter sur ta machine ou Colab</span>
    </div>
  </aside>
  <main id="main">{{MODULES}}</main>
</div>

<script>
window.MathJax = {
  tex: { inlineMath: [["$", "$"]], displayMath: [["$$", "$$"]], processEscapes: true },
  svg: { fontCache: "global" },
  options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] },
  startup: { typeset: false }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
<script>
(function () {
  const views = Array.from(document.querySelectorAll(".module"));
  const keys = views.map(v => v.dataset.key);
  const links = Array.from(document.querySelectorAll(".rail a"));
  const picker = document.getElementById("module-picker");
  const typeset = new Set();

  function typesetView(view) {
    if (typeset.has(view.dataset.key)) return;
    const run = () => {
      if (window.MathJax && MathJax.typesetPromise) {
        typeset.add(view.dataset.key);
        MathJax.typesetPromise([view]).catch(err => console.error(err));
      } else {
        setTimeout(run, 150);
      }
    };
    run();
  }

  function show(key, scroll) {
    if (!keys.includes(key)) key = keys[0];
    views.forEach(v => { v.hidden = v.dataset.key !== key; });
    links.forEach(a => a.toggleAttribute("aria-current", false));
    const active = links.find(a => a.dataset.key === key);
    if (active) active.setAttribute("aria-current", "page");
    picker.value = key;
    const view = views.find(v => v.dataset.key === key);
    typesetView(view);
    try { localStorage.setItem("deepnlp:last", key); } catch (e) {}
    if (scroll) window.scrollTo({ top: 0 });
    document.title = "Cours Deep Learning NLP";
  }

  window.addEventListener("hashchange", () => show(location.hash.slice(1), true));
  picker.addEventListener("change", () => { location.hash = picker.value; });

  let start = location.hash.slice(1);
  if (!keys.includes(start)) {
    try { start = localStorage.getItem("deepnlp:last") || keys[0]; } catch (e) { start = keys[0]; }
  }
  show(start, false);
})();
</script>
"""

if __name__ == "__main__":
    build()
