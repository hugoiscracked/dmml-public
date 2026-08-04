#!/usr/bin/env python3
"""Inject the GLOSS lookup into docs/index.html from docs/glossary.html.

The skill tree's "Key terms" tooltips reuse the glossary's definitions. Rather than
copy them by hand, this pulls the definitions for exactly the slugs referenced in
index.html's NODE_TERMS out of glossary.html's TERMS array and writes them into the
`const GLOSS = {...}` line. Re-run after editing either the glossary or NODE_TERMS.

    python docs/gen_gloss.py
"""
import re, json, sys, pathlib

here = pathlib.Path(__file__).resolve().parent
gl = (here / "glossary.html").read_text()
idx_path = here / "index.html"
idx = idx_path.read_text()

def slug(s):  # must match the JS slug() in both pages
    return re.sub(r'^-|-$', '', re.sub(r'[^a-z0-9]+', '-', s.lower()))

# All glossary terms -> {slug: {name, def}}
gjs = gl.split("const TERMS = [", 1)[1]
gmap = {}
for m in re.finditer(r'\{t:"((?:[^"\\]|\\.)*)"[^}]*?d:"((?:[^"\\]|\\.)*)"', gjs):
    name, d = m.group(1), m.group(2)
    gmap[slug(name)] = {"name": name, "def": d}

# Slugs referenced by NODE_TERMS in index.html
nt = idx.split("const NODE_TERMS = {", 1)[1].split("\n};", 1)[0]
needed = sorted(set(re.findall(r'"([a-z0-9-]+)"', nt)))

missing = [s for s in needed if s not in gmap]
if missing:
    sys.exit(f"ERROR: NODE_TERMS references slugs not found in the glossary: {missing}")

sub = {s: gmap[s] for s in needed}
block = "const GLOSS = " + json.dumps(sub, ensure_ascii=False) + ";"
new = re.sub(r'const GLOSS = \{[^\n]*\};', block, idx, count=1)
idx_path.write_text(new)
print(f"Injected {len(sub)} glossary definitions into index.html "
      f"({len(gmap)} terms available).")
