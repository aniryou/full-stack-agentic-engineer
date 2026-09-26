"""MkDocs hooks for the guide site (registered under `hooks:` in mkdocs.yml).

Cache-busting for the site's own stylesheets. MkDocs links `extra_css` at a fixed path, and GitHub Pages serves it
with `Cache-Control: max-age=600`, so right after a deploy a browser can pair the new HTML with the previous
`extra.css`. When a deploy renames classes (as the landing page's did), the page renders unstyled until the cache
expires. Appending a hash of the file's content to the URL makes every change to the CSS a new URL.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def on_config(config, **kwargs):
    docs = Path(config["docs_dir"])
    busted = []
    for entry in config["extra_css"]:
        path = str(entry)
        src = docs / path
        if "?" not in path and "://" not in path and src.is_file():
            digest = hashlib.sha256(src.read_bytes()).hexdigest()[:10]
            path = f"{path}?v={digest}"
        busted.append(path)
    config["extra_css"] = busted
    return config
