#!/usr/bin/env node
// render_figures.js: a writer's check for the figures in the Markdown documents (tools/orchestration/FIGURE-STYLE.md).
// It renders an SVG file, or every ```mermaid block of a Markdown file, to PNG with headless Chromium, so that the
// writer can look at each figure, and it fails on a Mermaid block that does not parse (the site and GitHub would
// show an error box there).
//
//   node tools/orchestration/render_figures.js svg <file.svg> <out.png>   screenshot one SVG at its natural size
//   node tools/orchestration/render_figures.js md <file.md> <outdir>      parse + render every mermaid block; one PNG
//                                                                        and one SVG per block; a summary line per block
//
// Prerequisites, in any folder on NODE_PATH (not in the repository): `npm i playwright mermaid` and a Chromium for
// Playwright (`npx playwright install chromium`). The site loads mermaid@11 (Material for MkDocs); install the same
// major. CI does not run this script.
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const MERMAID = require.resolve('mermaid/dist/mermaid.min.js');

async function main() {
  const [mode, src, out] = process.argv.slice(2);
  if (!mode || !src || !out) {
    console.log('usage: render_figures.js svg <file.svg> <out.png> | md <file.md> <outdir>');
    process.exit(2);
  }
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  try {
    if (mode === 'svg') {
      const svg = fs.readFileSync(src, 'utf8');
      await page.setContent(`<html><body style="margin:0;background:#fff;display:inline-block">${svg}</body></html>`);
      const el = await page.$('svg');
      if (!el) throw new Error(`${src}: no <svg> element`);
      await el.screenshot({ path: out });
      const box = await el.boundingBox();
      console.log(`wrote ${out} (${Math.round(box.width)}x${Math.round(box.height)} css px)`);
    } else if (mode === 'md') {
      const text = fs.readFileSync(src, 'utf8');
      const re = /```mermaid[^\n]*\n([\s\S]*?)```/g;
      let m, i = 0, bad = 0;
      fs.mkdirSync(out, { recursive: true });
      await page.setContent('<html><body style="margin:16px;background:#fff"><div id="d"></div></body></html>');
      await page.addScriptTag({ content: fs.readFileSync(MERMAID, 'utf8') });
      await page.evaluate(() => mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: 'default' }));
      while ((m = re.exec(text))) {
        i++;
        const line = text.slice(0, m.index).split('\n').length;
        const code = m[1];
        const stem = `${path.basename(src).replace(/\.md$/, '')}-m${i}`;
        try {
          const svg = await page.evaluate(async (code) => {
            await mermaid.parse(code);
            const { svg } = await mermaid.render('g' + Math.random().toString(36).slice(2), code);
            document.getElementById('d').innerHTML = svg;
            return svg;
          }, code);
          fs.writeFileSync(path.join(out, `${stem}.svg`), svg);
          const el = await page.$('#d svg');
          await el.screenshot({ path: path.join(out, `${stem}.png`) });
          const box = await el.boundingBox();
          console.log(`ok   block ${i} (line ${line}) ${Math.round(box.width)}x${Math.round(box.height)} -> ${stem}.png`);
        } catch (e) {
          bad++;
          console.log(`FAIL block ${i} (line ${line}): ${String(e.message || e).split('\n').slice(0, 3).join(' | ')}`);
        }
      }
      console.log(`${i} mermaid blocks, ${bad} failed`);
      if (bad) process.exitCode = 1;
    } else {
      console.log('usage: render_figures.js svg <file.svg> <out.png> | md <file.md> <outdir>');
      process.exitCode = 2;
    }
  } finally {
    await browser.close();
  }
}

main().catch((e) => { console.error(e); process.exit(2); });
