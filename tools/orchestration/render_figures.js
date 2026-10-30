#!/usr/bin/env node
// render_figures.js: a writer's check for the figures in the Markdown documents (tools/orchestration/FIGURE-STYLE.md).
// It renders an SVG file, or every ```mermaid block of a Markdown file, to PNG with headless Chromium, so that the
// writer can look at each figure, and it says whether each figure fits one screen (FIGURE-STYLE.md §3: shown in a
// column of COL px, at most MAX_SHOWN_H px tall and scaled down by no more than MIN_SCALE). It fails on a Mermaid
// block that does not parse (the site and GitHub would show an error box there), on an SVG that is not well-formed
// XML (loaded as an image, as GitHub and the site load it, the browser would show a broken image;
// tools/ci/tests/test_svg_wellformed.py guards the same in CI) and on a figure that does not fit.
//
//   node tools/orchestration/render_figures.js svg <file.svg> <out.png>   screenshot one SVG at its natural size
//   node tools/orchestration/render_figures.js md <file.md> <outdir>      parse + render every mermaid block; one PNG
//                                                                        and one SVG per block; a verdict per block
//   node tools/orchestration/render_figures.js check [<outdir>]          every mermaid block of every tracked Markdown
//                                                                        file and every tracked SVG: a verdict each,
//                                                                        and the measurements written to
//                                                                        tools/orchestration/figures-manifest.tsv,
//                                                                        which tools/ci/tests/test_figures_fit.py reads
//
// Prerequisites, in any folder on NODE_PATH (not in the repository): `npm i playwright mermaid` and a Chromium for
// Playwright (`npx playwright install chromium`). The site loads mermaid@11 (Material for MkDocs); install the same
// major. CI does not run this script; it reads the manifest the check mode writes.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const cp = require('child_process');
const { chromium } = require('playwright');
const MERMAID = require.resolve('mermaid/dist/mermaid.min.js');

// One screen (FIGURE-STYLE.md §3). The same three numbers are in tools/ci/tests/test_figures_fit.py.
const COL = 800;          // px: the content column the site and GitHub give a figure on a laptop
const MAX_SHOWN_H = 600;  // px: what a 768 px laptop screen shows below the page header, without a scroll
const MIN_SCALE = 0.7;    // a wider figure is scaled down to the column; 16 px labels must stay about 11 px
const ROOT = path.resolve(__dirname, '..', '..');
const MANIFEST = path.join(ROOT, 'tools', 'orchestration', 'figures-manifest.tsv');

function verdict(w, h) {
  const scale = Math.min(1, COL / w);
  const shown = Math.round(h * scale);
  const why = [];
  if (shown > MAX_SHOWN_H) why.push(`too tall: ${shown} px when shown, the limit is ${MAX_SHOWN_H}`);
  if (scale < MIN_SCALE) why.push(`too wide: shown at ${scale.toFixed(2)} of its size, the limit is ${MIN_SCALE}`);
  return { shown, scale, ok: why.length === 0, why: why.join('; ') || 'fits' };
}

const normalize = (code) => code.split('\n').map((l) => l.trimEnd()).join('\n').trim() + '\n';
const sha1 = (text) => crypto.createHash('sha1').update(text, 'utf8').digest('hex');

async function mermaidPage(browser) {
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  await page.setContent('<html><body style="margin:16px;background:#fff"><div id="d"></div></body></html>');
  await page.addScriptTag({ content: fs.readFileSync(MERMAID, 'utf8') });
  await page.evaluate(() => mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: 'default' }));
  return page;
}

// Render one Mermaid block on a prepared page: {svg, width, height} or a thrown parse error.
async function renderMermaid(page, code) {
  const svg = await page.evaluate(async (code) => {
    await mermaid.parse(code);
    const { svg } = await mermaid.render('g' + Math.random().toString(36).slice(2), code);
    document.getElementById('d').innerHTML = svg;
    return svg;
  }, code);
  const el = await page.$('#d svg');
  const box = await el.boundingBox();
  return { svg, el, width: Math.round(box.width), height: Math.round(box.height) };
}

// Load an SVG file the way GitHub and the site do, as an image: the browser then parses it as strict XML, so a
// duplicate attribute or an unclosed tag fails here instead of showing a broken image on the page.
async function loadSvg(page, src) {
  const svg = fs.readFileSync(src, 'utf8');
  const parseError = await page.evaluate((text) => {
    const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
    const err = doc.querySelector('parsererror');
    return err ? err.textContent.split('\n').slice(0, 2).join(' ') : null;
  }, svg);
  if (parseError) throw new Error(`${src}: not well-formed XML: ${parseError}`);
  const dataUrl = 'data:image/svg+xml;base64,' + Buffer.from(svg, 'utf8').toString('base64');
  await page.setContent(`<html><body style="margin:0;background:#fff"><img id="f" src="${dataUrl}"></body></html>`);
  const ok = await page.evaluate(() => new Promise((resolve) => {
    const img = document.getElementById('f');
    if (img.complete) return resolve(img.naturalWidth > 0);
    img.onload = () => resolve(img.naturalWidth > 0);
    img.onerror = () => resolve(false);
  }));
  if (!ok) throw new Error(`${src}: the browser could not load it as an image`);
  const el = await page.$('#f');
  const box = await el.boundingBox();
  return { el, width: Math.round(box.width), height: Math.round(box.height) };
}

const blocksOf = (text) => {
  const re = /^[ \t]{0,3}```mermaid[^\n]*\n([\s\S]*?)^[ \t]{0,3}```[ \t]*$/gm;   // a fence on a line of its own
  const out = [];
  let m;
  while ((m = re.exec(text))) out.push({ line: text.slice(0, m.index).split('\n').length, code: m[1] });
  return out;
};

async function main() {
  const [mode, src, out] = process.argv.slice(2);
  if (!mode || (mode !== 'check' && (!src || !out))) {
    console.log('usage: render_figures.js svg <file.svg> <out.png> | md <file.md> <outdir> | check [<outdir>]');
    process.exit(2);
  }
  const browser = await chromium.launch();
  try {
    if (mode === 'svg') {
      const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
      const { el, width, height } = await loadSvg(page, src);
      await el.screenshot({ path: out });
      const v = verdict(width, height);
      console.log(`${v.ok ? 'ok  ' : 'FAIL'} ${src} ${width}x${height} css px, ${v.why} -> ${out}`);
      if (!v.ok) process.exitCode = 1;
    } else if (mode === 'md') {
      const text = fs.readFileSync(src, 'utf8');
      const page = await mermaidPage(browser);
      let i = 0, bad = 0;
      fs.mkdirSync(out, { recursive: true });
      for (const b of blocksOf(text)) {
        i++;
        const stem = `${path.basename(src).replace(/\.md$/, '')}-m${i}`;
        try {
          const r = await renderMermaid(page, b.code);
          fs.writeFileSync(path.join(out, `${stem}.svg`), r.svg);
          await r.el.screenshot({ path: path.join(out, `${stem}.png`) });
          const v = verdict(r.width, r.height);
          if (!v.ok) bad++;
          console.log(`${v.ok ? 'ok  ' : 'FAIL'} block ${i} (line ${b.line}) ${r.width}x${r.height}, ${v.why} -> ${stem}.png`);
        } catch (e) {
          bad++;
          console.log(`FAIL block ${i} (line ${b.line}): ${String(e.message || e).split('\n').slice(0, 3).join(' | ')}`);
        }
      }
      console.log(`${i} mermaid blocks, ${bad} failed (a parse error, or a figure that does not fit one screen)`);
      if (bad) process.exitCode = 1;
    } else if (mode === 'check') {
      const outdir = src || null;
      if (outdir) fs.mkdirSync(outdir, { recursive: true });
      const files = cp.execSync('git ls-files --cached --others --exclude-standard', { cwd: ROOT }).toString().split('\n').filter(Boolean);   // tracked and new files
      const page = await mermaidPage(browser);
      const rows = [];
      let bad = 0;
      for (const f of files.filter((f) => f.endsWith('.md'))) {
        const text = fs.readFileSync(path.join(ROOT, f), 'utf8');
        if (!/^[ \t]{0,3}```mermaid/m.test(text)) continue;
        let i = 0;
        for (const b of blocksOf(text)) {
          i++;
          const hash = sha1(normalize(b.code));
          try {
            const r = await renderMermaid(page, b.code);
            const v = verdict(r.width, r.height);
            if (outdir) await r.el.screenshot({ path: path.join(outdir, `${f.replace(/\//g, '__').replace(/\.md$/, '')}-m${i}.png`) });
            rows.push(['mermaid', f, String(i), String(b.line), hash, String(r.width), String(r.height), String(v.shown), v.scale.toFixed(2), v.ok ? 'fits' : v.why]);
            if (!v.ok) bad++;
            console.log(`${v.ok ? 'ok  ' : 'FAIL'} ${f} block ${i} (line ${b.line}) ${r.width}x${r.height}, ${v.why}`);
          } catch (e) {
            bad++;
            const why = 'parse error: ' + String(e.message || e).split('\n')[0].slice(0, 120);
            rows.push(['mermaid', f, String(i), String(b.line), hash, '0', '0', '0', '0', why]);
            console.log(`FAIL ${f} block ${i} (line ${b.line}): ${why}`);
          }
        }
      }
      const svgPage = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
      for (const f of files.filter((f) => f.endsWith('.svg'))) {
        try {
          const r = await loadSvg(svgPage, path.join(ROOT, f));
          const v = verdict(r.width, r.height);
          if (outdir) await r.el.screenshot({ path: path.join(outdir, f.replace(/\//g, '__').replace(/\.svg$/, '') + '.png') });
          rows.push(['svg', f, '-', '0', '-', String(r.width), String(r.height), String(v.shown), v.scale.toFixed(2), v.ok ? 'fits' : v.why]);
          if (!v.ok) bad++;
          console.log(`${v.ok ? 'ok  ' : 'FAIL'} ${f} ${r.width}x${r.height}, ${v.why}`);
        } catch (e) {
          bad++;
          console.log(`FAIL ${f}: ${String(e.message || e).split('\n')[0]}`);
        }
      }
      rows.sort((a, b) => (a[1] + '\t' + a[2].padStart(4, '0')).localeCompare(b[1] + '\t' + b[2].padStart(4, '0')));
      const header = '# Every figure, measured by `node tools/orchestration/render_figures.js check` (FIGURE-STYLE.md §3): kind, document, block, line, sha1 of the Mermaid source, width, height, height when shown in a ' + COL + ' px column, scale, verdict.\n';
      fs.writeFileSync(MANIFEST, header + 'kind\tdoc\tblock\tline\tsha1\tw\th\tshown_h\tscale\tverdict\n' + rows.map((r) => r.join('\t')).join('\n') + '\n');
      console.log(`${rows.length} figures, ${bad} do not fit or do not render; wrote ${path.relative(ROOT, MANIFEST)}`);
      if (bad) process.exitCode = 1;
    } else {
      console.log('usage: render_figures.js svg <file.svg> <out.png> | md <file.md> <outdir> | check [<outdir>]');
      process.exitCode = 2;
    }
  } finally {
    await browser.close();
  }
}

main().catch((e) => { console.error(e); process.exit(2); });
