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
  function run() { mark(); tagVerify(); }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(run);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
  } else {
    run();
  }
})();
