/*
 * Small presentation fixes for content pages (see the table and "(verify)" rules in site/stylesheets/extra.css).
 * Runs on every page; it only adds classes and wraps text, so it changes no words.
 *
 * 1. Phone-width tables: every cell of a column whose cells are all short (at most 14 characters: "~5 h", "T0 → T1",
 *    "03") gets .fse-narrow, which drops the 10em floor the prose columns need.
 * 2. Numeric columns: a column whose every body cell is a number ("65.53 GB", "~320 GB/s", "12.06", "2×", "–")
 *    gets .fse-num on all its cells, header included: right-aligned, tabular figures, no wrapping.
 * 3. "(verify)" — the repo's mark on a dated product fact — is wrapped in <span class="fse-verify"> so it reads as
 *    a tag beside the fact rather than as part of the sentence. Code, headings and links are left alone.
 * 5. Inline maths and the punctuation around it stay on one line. Browsers may break a line between an inline
 *    formula (an atomic inline box) and the comma, colon or bracket next to it, so ", a" or ":" can start a line.
 *    Each inline formula is wrapped with the bracket before it and the punctuation after it in a no-wrap span;
 *    runs once the page is parsed (arithmatex spans) and again after MathJax has typeset (notebook pages, see
 *    site/javascripts/mathjax.js).
 * 4. A code block on a Markdown page that is a little wider than the column (the layer READMEs' stack diagram is
 *    117 characters; formula fences and ASCII diagrams are alike) gets .fse-shrink-1 / .fse-shrink-2, a 10% or 20%
 *    smaller font, when that makes it fit; anything wider keeps its horizontal scroll. Notebook cells are real code
 *    and are left alone.
 */
(function () {
  var SHORT = 14;
  var NUMERIC = /^[~≈<>≤≥±−–+-]?\s?[$€£]?\d[\d.,]*\s?(?:[%×x]|[kKMGTP]?i?B(?:\/s)?|ms|µs|us|s|h|min|GB\/s|TB\/s|TFLOPS?|GFLOPS?|FLOP\/B|W|kW|tok\/s|tokens?)?(?:\s?\(.*\))?$/;
  var DASH = /^[-–—]$/;
  function isNum(t) { return t.length <= 14 && (NUMERIC.test(t) || DASH.test(t)); }
  function mark() {
    document.querySelectorAll(".md-typeset table:not([class])").forEach(function (table) {
      var rows = Array.prototype.slice.call(table.rows);
      if (!rows.length) return;
      var cols = Math.max.apply(null, rows.map(function (r) { return r.cells.length; }));
      for (var c = 0; c < cols; c++) {
        var cells = rows.map(function (r) { return r.cells[c]; }).filter(Boolean);
        var body = cells.filter(function (cell) { return cell.tagName === "TD"; });
        var texts = body.map(function (cell) { return cell.textContent.trim(); });
        var shortCol = body.length > 0 && texts.every(function (t) { return t.length <= SHORT; });
        if (shortCol) cells.forEach(function (cell) { cell.classList.add("fse-narrow"); });
        var filled = texts.filter(function (t) { return t.length > 0 && !DASH.test(t); });
        var numCol = filled.length > 0 && texts.every(isNum);
        if (numCol && c > 0) cells.forEach(function (cell) { cell.classList.add("fse-num"); });
      }
    });
  }
  function tagVerify() {
    var root = document.querySelector(".md-content");
    if (!root) return;
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        if (n.nodeValue.indexOf("(verify)") < 0) return NodeFilter.FILTER_REJECT;
        var p = n.parentElement;
        if (!p || p.closest("pre, code, a, h1, h2, h3, h4, h5, h6, .fse-verify, .jp-OutputArea")) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (n) {
      var parts = n.nodeValue.split("(verify)");
      var frag = document.createDocumentFragment();
      parts.forEach(function (part, i) {
        if (part) frag.appendChild(document.createTextNode(part));
        if (i < parts.length - 1) {
          var span = document.createElement("span");
          span.className = "fse-verify";
          span.textContent = "(verify)";
          frag.appendChild(span);
        }
      });
      n.parentNode.replaceChild(frag, n);
    });
  }
  function shrinkWidePre() {
    document.querySelectorAll(".md-content pre").forEach(function (pre) {
      if (pre.closest(".jupyter-wrapper")) return;
      var code = pre.querySelector("code") || pre;
      pre.classList.remove("fse-shrink-1", "fse-shrink-2");
      var ratio = code.scrollWidth / Math.max(1, code.clientWidth);
      if (ratio <= 1.005) return;
      if (ratio <= 1.11) pre.classList.add("fse-shrink-1");
      else if (ratio <= 1.25) pre.classList.add("fse-shrink-2");
    });
  }
  var AFTER = /^[,.;:!?)\]]+/, BEFORE = /[(\[]$/;
  function glueMath() {
    var nodes = document.querySelectorAll(".md-typeset span.arithmatex, .md-typeset mjx-container");
    nodes.forEach(function (el) {
      if (el.getAttribute && el.getAttribute("display") === "true") return;
      if (el.closest(".fse-mathglue")) return;
      var target = el.closest("span.arithmatex") || el;
      if (target.parentElement && target.parentElement.classList.contains("fse-mathglue")) return;
      var next = target.nextSibling, prev = target.previousSibling, after = "", before = "", m;
      if (next && next.nodeType === 3 && (m = next.nodeValue.match(AFTER))) { after = m[0]; next.nodeValue = next.nodeValue.slice(m[0].length); }
      if (prev && prev.nodeType === 3 && (m = prev.nodeValue.match(BEFORE))) { before = m[0]; prev.nodeValue = prev.nodeValue.slice(0, -m[0].length); }
      if (!after && !before) return;
      var wrap = document.createElement("span");
      wrap.className = "fse-mathglue";
      target.parentNode.insertBefore(wrap, target);
      if (before) wrap.appendChild(document.createTextNode(before));
      wrap.appendChild(target);
      if (after) wrap.appendChild(document.createTextNode(after));
    });
  }
  window.fseGlueMath = glueMath;
  function run() { mark(); tagVerify(); shrinkWidePre(); glueMath(); }
  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(shrinkWidePre, 150);
  });
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(run);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
