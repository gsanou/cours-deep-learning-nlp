#!/usr/bin/env python3
"""Wrap an artifact-style page (no <html>/<head>) into a standalone HTML document for GitHub Pages.

Usage: python tools/wrap_page.py site/labo_chapitre0.html site/labo.html
"""
import sys
from pathlib import Path

HEAD = ('<!doctype html>\n<html lang="fr">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n')

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
body = src.read_text(encoding="utf-8")
cut = body.index("<div class=\"wrap\">")                 # <title>, metas, fonts and <style> go into <head>
dst.write_text(HEAD + body[:cut] + "</head>\n<body>\n" + body[cut:] + "\n</body>\n</html>\n", encoding="utf-8")
print(f"wrote {dst}")
