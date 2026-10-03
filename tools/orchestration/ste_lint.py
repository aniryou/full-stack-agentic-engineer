#!/usr/bin/env python3
"""A heuristic ASD-STE100 (Simplified Technical English) linter for Markdown and notebook prose.

    python3 tools/orchestration/ste_lint.py <file.md|notebook.ipynb|dir> ...   # findings, then a summary per file
    python3 tools/orchestration/ste_lint.py --summary <paths>                  # the summary only
    python3 tools/orchestration/ste_lint.py --json <paths>                     # machine-readable summary
    python3 tools/orchestration/ste_lint.py --no-fail <paths>                  # exit 0 even with errors

The rules are the ones in tools/orchestration/STE100-STYLE.md that a program can check:

  ERROR  long-sentence     more than 25 words in a sentence (S2)
  ERROR  long-paragraph    more than six sentences in a paragraph (P2)
  ERROR  modal             should / may / might / could / would / shall / ought (W5)
  ERROR  contraction       don't, it's, you're, ... (W6)
  ERROR  dash              an em dash, a spaced en dash or a spaced hyphen used as a dash (S4)
  ERROR  arrow             an arrow in prose (S4)
  WARN   long-step         more than 20 words in a numbered list item, a procedure (S2)
  WARN   ing               an -ing word outside the exceptions (V2)
  WARN   passive           a probable passive: be + past participle (V3)
  WARN   semicolon         a semicolon chain (L2)
  WARN   abbreviation      etc., e.g., i.e., vs., & (W6)
  WARN   word              a word the standard replaces, with the replacement (W7, STE100-STYLE.md 3.1)

Skipped: fenced code, inline code (counted as one word), headings, link targets, images, HTML tags, mathematics
(counted as one word), table separator rows, ALL-CAPS tokens (statuses), the `(verify)` tag. Table cells and list
items are checked as paragraphs of their own (a cell is not held to the six-sentence rule).

Standard library only. The output is `path:line: LEVEL rule: detail | sentence`. Exit code 1 when any file has an
error, unless --no-fail.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

MAX_WORDS = 25
SKIP_DIRS = {".ipynb_checkpoints", "_run_outputs", ".pytest_cache", ".git", "node_modules", ".venv", "venv"}
MAX_WORDS_STEP = 20
MAX_SENTENCES = 6

ING_ALLOW = {
    "long-running", "nothing", "something", "anything", "everything", "thing", "things", "during", "string",
    "strings", "ring", "king", "wing", "spring", "bring", "sing", "sting", "swing", "sling", "morning", "evening",
    "warning", "warnings", "logging", "monitoring", "tracing", "streaming", "routing", "ordering", "pricing",
    "batching", "caching", "sharding", "training", "embedding", "embeddings", "fine-tuning", "billing", "mapping",
    "mappings", "setting", "settings", "ceiling", "spring", "offspring", "ping", "pending",  # pending: status word
}
IRREGULAR_PARTICIPLES = (
    "built", "done", "made", "taken", "given", "kept", "held", "sent", "put", "lost", "found", "won", "begun",
    "chosen", "broken", "written", "read", "run", "seen", "known", "shown", "thrown", "bought", "caught", "taught",
    "told", "said", "set", "cut", "hit", "left", "met", "paid", "spent", "split", "understood", "hidden", "driven",
    "re-driven", "forgotten", "gotten", "led", "fed", "bound", "wound", "rebuilt", "overwritten", "rerun", "re-run",
    "undone", "redone", "woken", "stuck", "spun", "dealt", "meant", "sought", "brought", "thought", "fought",
    "sold", "laid", "lit", "shut", "slid", "sped", "swept", "swung", "torn", "worn", "born", "drawn", "grown",
    "flown", "blown", "frozen", "stolen", "struck", "swum", "sung", "rung", "sunk", "shrunk", "beaten", "bitten",
    "eaten", "fallen", "risen", "ridden", "shaken", "spoken", "stricken", "sworn", "woven",
)
MODALS = re.compile(r"\b(should|may|might|could|would|shall|ought)\b", re.I)
CONTRACTION = re.compile(
    r"\b(\w+n't|\w+'re|\w+'ve|\w+'ll|I'd|you'd|we'd|they'd|he'd|she'd|it'd|I'm|it's|let's|that's|there's|here's|"
    r"what's|who's|where's|how's|when's|why's)\b", re.I)
ARROWS = re.compile(r"[→←⇒⇐⟶⟵↔⇔↑↓]|-->|<--|->|=>|<=(?!\d)")
EMDASH = re.compile(r"—|(?<=\s)–(?=\s)|(?<=\w)\s-\s(?=\w)|(?<=\S)\s—|—\s(?=\S)")
SEMICOLON = re.compile(r";")
ABBREV = re.compile(r"\b(etc\.?|e\.g\.|i\.e\.|vs\.?|cf\.|approx\.|a\.k\.a\.)(?=\s|$|[,;:)])|\s&\s", re.I)
PASSIVE = re.compile(
    r"\b(am|is|are|was|were|be|been|being|get|gets|got|gotten)\b"
    r"(\s+(?:not|also|then|never|always|still|only|already|now|usually|often|\w+ly))?\s+"
    r"(\w+(?:ed|en)|" + "|".join(IRREGULAR_PARTICIPLES) + r")\b", re.I)
WORDS = {
    "accomplish": "do", "achieve": "get / do", "carry out": "do", "perform": "do", "performs": "does",
    "performed": "did", "performing": "do", "additional": "more", "additionally": "also", "in addition": "also",
    "furthermore": "also", "allow": "let / permit", "allows": "lets / permits", "allowed": "let / permitted",
    "enable": "let / permit", "enables": "lets / permits", "enabled": "(a state: on)", "alter": "change",
    "modify": "change", "modifies": "changes", "modified": "changed", "amount": "quantity",
    "appropriate": "correct / applicable", "assure": "make sure", "ensure": "make sure", "ensures": "makes sure",
    "ensured": "made sure", "verify": "make sure", "verifies": "makes sure", "verified": "made sure",
    "avoid": "prevent / do not", "avoids": "prevents", "big": "large", "huge": "large", "choose": "select",
    "chooses": "selects", "chose": "selected", "pick": "select", "picks": "selects", "picked": "selected",
    "commence": "start", "initiate": "start", "initiates": "starts", "comprise": "contain / have",
    "consist of": "contain / have", "consists of": "contains / has", "consequently": "thus", "therefore": "thus",
    "hence": "thus", "demonstrate": "show", "demonstrates": "shows", "indicate": "show", "indicates": "shows",
    "determine": "find / calculate / decide", "determines": "finds / calculates / decides", "due to": "because of",
    "employ": "use", "employs": "uses", "utilize": "use", "utilise": "use", "enough": "sufficient",
    "establish": "make / find", "fix": "repair / a correction", "fixes": "repairs / corrections",
    "fixed": "repaired / corrected", "happen": "occur", "happens": "occurs", "happened": "occurred",
    "however": "but", "in order to": "to", "little": "small", "tiny": "small", "locate": "find", "obtain": "get",
    "obtains": "gets", "acquire": "get", "acquires": "gets", "prior to": "before", "proceed": "continue",
    "proceeds": "continues", "provide": "supply / give", "provides": "supplies / gives", "quick": "fast",
    "quickly": "fast", "regarding": "about", "remain": "stay", "remains": "stays", "remained": "stayed",
    "require": "must / necessary", "requires": "must / is necessary", "required": "necessary", "retain": "keep",
    "retains": "keeps", "subsequently": "then", "terminate": "stop / end", "terminates": "stops / ends",
    "very": "(remove)", "quite": "(remove)", "rather": "(remove)", "essentially": "(remove)",
    "basically": "(remove)", "simply": "(remove)", "whether": "if", "whilst": "while", "wrong": "incorrect",
    "a lot of": "many / much", "lots of": "many / much", "as well as": "and", "plenty": "sufficient",
}
WORDS_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(w) for w in WORDS), key=len, reverse=True)) + r")\b", re.I)
ALLCAPS = re.compile(r"^[A-Z0-9_./-]+$")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
FENCE = re.compile(r"^\s*(```|~~~)")
TABLE_ROW = re.compile(r"^\s*\|")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
NUMBERED = re.compile(r"^\s*\d+[.)]\s+")
QUOTE = re.compile(r"^\s*>\s?")
ALERT_TAG = re.compile(r"^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*")
HRULE = re.compile(r"^\s*([-*_]\s*){3,}$")
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[]?[A-Z0-9])")
PROTECT = [("e.g.", "e.g"), ("i.e.", "i.e"), ("etc.", "etc"), ("vs.", "vs"), ("approx.", "approx"),
           ("cf.", "cf"), ("a.k.a.", "aka"), ("No.", "No")]


def clean_inline(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " MATH ", text, flags=re.S)
    text = re.sub(r"(?<!\\)\$[^$\n]+?\$", " MATH ", text)
    text = re.sub(r"``.+?``", " CODE ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " CODE ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)
    text = HTML_TAG.sub(" ", text)
    text = text.replace("(verify)", "(VERIFYTAG)")
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"(?<![\w*])\*(?=\S)|(?<=\S)\*(?![\w*])", "", text)
    text = text.replace("\\$", "$")
    return re.sub(r"\s+", " ", text).strip()


def sentences(text: str) -> list[str]:
    for a, b in PROTECT:
        text = text.replace(a, b)
    return [s.strip() for s in SENT_END.split(text) if s.strip()]


def words(sentence: str) -> list[str]:
    out = []
    for tok in sentence.split():
        core = re.sub(r"^[^\w$]+|[^\w)]+$", "", tok)
        if re.search(r"\w", core):
            out.append(core)
    return out


class Finding:
    __slots__ = ("line", "level", "rule", "detail", "sentence")

    def __init__(self, line, level, rule, detail, sentence):
        self.line, self.level, self.rule, self.detail, self.sentence = line, level, rule, detail, sentence


def check_block(kind: str, line: int, text: str, findings: list[Finding], stats: dict) -> None:
    text = clean_inline(text)
    if not text:
        return
    sents = sentences(text)
    if kind == "para" and len(sents) > MAX_SENTENCES:
        findings.append(Finding(line, "ERROR", "long-paragraph", f"{len(sents)} sentences", sents[0][:80]))
    for s in sents:
        ws = words(s)
        n = len(ws)
        stats["sentences"] += 1
        stats["words"] += n
        stats["max_words"] = max(stats["max_words"], n)
        excerpt = s if len(s) <= 110 else s[:107] + "..."
        if n > MAX_WORDS:
            findings.append(Finding(line, "ERROR", "long-sentence", f"{n} words", excerpt))
        elif kind == "step" and n > MAX_WORDS_STEP:
            findings.append(Finding(line, "WARN", "long-step", f"{n} words", excerpt))
        for m in MODALS.finditer(s):
            findings.append(Finding(line, "ERROR", "modal", m.group(0), excerpt))
        for m in CONTRACTION.finditer(s):
            findings.append(Finding(line, "ERROR", "contraction", m.group(0), excerpt))
        if EMDASH.search(s):
            findings.append(Finding(line, "ERROR", "dash", "dash used in prose", excerpt))
        if ARROWS.search(s):
            findings.append(Finding(line, "ERROR", "arrow", "arrow in prose", excerpt))
        if SEMICOLON.search(s):
            findings.append(Finding(line, "WARN", "semicolon", "split into sentences", excerpt))
        for m in ABBREV.finditer(s):
            findings.append(Finding(line, "WARN", "abbreviation", m.group(0).strip(), excerpt))
        for m in WORDS_RE.finditer(s):
            w = m.group(0).lower()
            findings.append(Finding(line, "WARN", "word", f"{w} -> {WORDS[w]}", excerpt))
        for m in PASSIVE.finditer(s):
            findings.append(Finding(line, "WARN", "passive", m.group(0), excerpt))
            stats["passive"] += 1
        for w in ws:
            lw = w.lower()
            if lw.endswith("ing") and len(lw) >= 5 and lw not in ING_ALLOW and not ALLCAPS.match(w) \
                    and w not in ("CODE", "MATH") and not lw.startswith("code"):
                findings.append(Finding(line, "WARN", "ing", w, excerpt))
                stats["ing"] += 1


def lint_markdown(text: str, line_offset: int = 0) -> tuple[list[Finding], dict]:
    findings: list[Finding] = []
    stats = {"sentences": 0, "words": 0, "max_words": 0, "passive": 0, "ing": 0}
    text = HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    lines = text.split("\n")
    in_fence = False
    block: list[str] = []
    kind = "para"
    start = 0

    def flush():
        nonlocal block
        if block:
            check_block(kind, start + line_offset, " ".join(block), findings, stats)
        block = []

    for i, raw in enumerate(lines, 1):
        if FENCE.match(raw):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = raw.rstrip()
        if not line.strip() or HRULE.match(line):
            flush()
            continue
        if HEADING.match(line):
            flush()
            continue
        if TABLE_ROW.match(line):
            flush()
            if TABLE_SEP.match(line):
                continue
            for cell in re.split(r"(?<!\\)\|", line.strip().strip("|")):
                if cell.strip():
                    check_block("cell", i + line_offset, cell, findings, stats)
            continue
        stripped = line
        if QUOTE.match(stripped):
            stripped = QUOTE.sub("", stripped)
            stripped = ALERT_TAG.sub("", stripped)
            if not stripped.strip():
                flush()
                continue
        if re.match(r"^\s*</?(details|summary)", stripped):
            flush()
            inner = HTML_TAG.sub(" ", stripped).strip()
            if inner:
                check_block("para", i + line_offset, inner, findings, stats)
            continue
        if LIST_ITEM.match(stripped):
            flush()
            kind = "step" if NUMBERED.match(stripped) else "item"
            start = i
            block = [LIST_ITEM.sub("", stripped, count=1)]
            continue
        if not block:
            kind, start = "para", i
        block.append(stripped.strip())
    flush()
    return findings, stats


def lint_notebook(path: pathlib.Path) -> tuple[list[Finding], dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    findings: list[Finding] = []
    stats = {"sentences": 0, "words": 0, "max_words": 0, "passive": 0, "ing": 0}
    for idx, cell in enumerate(d["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        f, s = lint_markdown(src, line_offset=idx * 1000)   # line = cell index * 1000 + line in cell
        findings += f
        for k in stats:
            stats[k] = max(stats[k], s[k]) if k == "max_words" else stats[k] + s[k]
    return findings, stats


def lint_path(path: pathlib.Path):
    if path.suffix == ".ipynb":
        return lint_notebook(path)
    return lint_markdown(path.read_text(encoding="utf-8"))


def collect(args: list[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for a in args:
        p = pathlib.Path(a)
        if p.is_dir():
            out += sorted(q for q in p.rglob("*") if q.suffix in (".md", ".ipynb")
                          and not SKIP_DIRS & set(q.parts))
        else:
            out.append(p)
    return out


def main(argv: list[str]) -> int:
    try:
        return _main(argv)
    except BrokenPipeError:
        return 0


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--summary", action="store_true", help="print the per-file summary only")
    ap.add_argument("--json", action="store_true", help="print a JSON summary only")
    ap.add_argument("--no-fail", action="store_true", help="exit 0 even when a file has errors")
    ap.add_argument("--rule", action="append", help="only report these rules (repeatable)")
    a = ap.parse_args(argv)
    report = {}
    total_err = 0
    for path in collect(a.paths):
        findings, stats = lint_path(path)
        if a.rule:
            findings = [f for f in findings if f.rule in a.rule]
        label = (lambda n: f"cell{n // 1000}:{n % 1000}") if path.suffix == ".ipynb" else str
        errs = [f for f in findings if f.level == "ERROR"]
        warns = [f for f in findings if f.level == "WARN"]
        by_rule: dict[str, int] = {}
        for f in findings:
            by_rule[f.rule] = by_rule.get(f.rule, 0) + 1
        report[str(path)] = {"errors": len(errs), "warnings": len(warns), "by_rule": by_rule,
                             "sentences": stats["sentences"], "words": stats["words"],
                             "mean_words": round(stats["words"] / stats["sentences"], 1) if stats["sentences"] else 0,
                             "max_words": stats["max_words"], "passive": stats["passive"], "ing": stats["ing"]}
        total_err += len(errs)
        if not a.summary and not a.json:
            for f in sorted(findings, key=lambda f: (f.line, f.level, f.rule)):
                print(f"{path}:{label(f.line)}: {f.level} {f.rule}: {f.detail} | {f.sentence}")
    if a.json:
        totals = {k: sum(r[k] for r in report.values()) for k in ("errors", "warnings", "sentences", "words", "passive", "ing")}
        print(json.dumps({"files": report, "totals": totals}, indent=1))
    else:
        print()
        print(f"{'file':<80} {'err':>4} {'warn':>5} {'sent':>5} {'mean':>5} {'max':>4} {'pass':>5} {'ing':>4}  rules")
        for p, r in report.items():
            rules = ", ".join(f"{k}={v}" for k, v in sorted(r["by_rule"].items(), key=lambda kv: -kv[1]))
            print(f"{p[-80:]:<80} {r['errors']:>4} {r['warnings']:>5} {r['sentences']:>5} {r['mean_words']:>5} "
                  f"{r['max_words']:>4} {r['passive']:>5} {r['ing']:>4}  {rules}")
        print(f"\n{len(report)} files, {total_err} errors, {sum(r['warnings'] for r in report.values())} warnings")
    return 0 if (a.no_fail or total_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
