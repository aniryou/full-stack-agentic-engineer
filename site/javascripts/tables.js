/*
 * Phone-width tables (see the table rules in site/stylesheets/extra.css): mark every cell of a column whose cells
 * are all short (at most 14 characters: "~5 h", "T0 → T1", "03") with .fse-narrow, which drops the 10em floor the
 * prose columns need. Runs on every page; it only adds a class, so it does nothing on wide screens.
 */
(function () {
  var SHORT = 14;
  function mark() {
    document.querySelectorAll(".md-typeset table:not([class])").forEach(function (table) {
      var rows = Array.prototype.slice.call(table.rows);
      if (!rows.length) return;
      var cols = Math.max.apply(null, rows.map(function (r) { return r.cells.length; }));
      for (var c = 0; c < cols; c++) {
        var cells = rows.map(function (r) { return r.cells[c]; }).filter(Boolean);
        var body = cells.filter(function (cell) { return cell.tagName === "TD"; });
        var shortCol = body.length > 0 && body.every(function (cell) {
          return cell.textContent.trim().length <= SHORT;
        });
        if (shortCol) cells.forEach(function (cell) { cell.classList.add("fse-narrow"); });
      }
    });
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(mark);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mark);
  } else {
    mark();
  }
})();
