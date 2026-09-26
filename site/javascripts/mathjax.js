/*
 * MathJax 3 configuration for the guide site. tools/site/hooks.py loads this file, then MathJax 3.2.2 from
 * jsDelivr, only on pages whose content has math.
 *
 * Inline math is \( ... \) only, never $ ... $: the repo is full of dollar amounts ("tiers unlock at $20 / $100"),
 * which must stay text. tools/site/build_site_content.py rewrites notebook TeX written as $...$ into \( ... \).
 * Display math: $$ ... $$ (notebooks) and \[ ... \] (notebooks, and Markdown pages through pymdownx.arithmatex).
 */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Never look for math in code, program output or the navigation.
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code", "annotation", "annotation-xml"],
    ignoreHtmlClass: "md-nav|md-header|md-footer|jp-OutputArea|highlight-ipynb"
  }
};
