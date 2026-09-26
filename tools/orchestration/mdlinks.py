#!/usr/bin/env python3
"""Check that relative links/images in Markdown files resolve. usage: mdlinks.py <file-or-dir> ..."""
import re, sys, pathlib
bad = 0
files = []
for a in sys.argv[1:]:
    p = pathlib.Path(a)
    files += sorted(p.rglob("*.md")) if p.is_dir() else [p]
for f in files:
    text = f.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    for m in re.finditer(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
        link = m.group(1)
        if re.match(r"^(https?:|mailto:|#)", link): continue
        target = (f.parent / link.split("#")[0]).resolve()
        if not target.exists():
            print(f"BROKEN {f}: {link}"); bad += 1
print(f"{len(files)} files checked, {bad} broken links"); sys.exit(1 if bad else 0)
