#!/usr/bin/env node
// svg2png.js: render many SVG files to PNG in one headless Chromium, as a browser shows them (loaded as an <img>, so a
// file that a browser rejects fails here too). Used by tools/book/build_book.py for the Kindle edition.
//
//   node tools/book/svg2png.js <jobs.json> [scale]
//
// jobs.json: [{"svg": "/abs/in.svg", "png": "/abs/out.png"}, ...]; scale (default 2) is the device scale factor.
// Prerequisites: playwright and a Chromium for it (the same as tools/orchestration/render_figures.js).
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

async function main() {
  const [jobsPath, scaleArg] = process.argv.slice(2);
  if (!jobsPath) { console.error('usage: svg2png.js <jobs.json> [scale]'); process.exit(2); }
  const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8'));
  const scale = Number(scaleArg || 2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: scale });
  let failed = 0;
  for (const job of jobs) {
    // as the repository's renderer does: a data URL (a file:// image is blocked from a blank page), parsed as XML first
    const svg = fs.readFileSync(job.svg, 'utf8');
    const parseError = await page.evaluate((text) => {
      const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
      const err = doc.querySelector('parsererror');
      return err ? err.textContent.split('\n').slice(0, 2).join(' ') : null;
    }, svg);
    if (parseError) { console.error(`FAIL ${job.svg}: not well-formed XML: ${parseError}`); failed++; continue; }
    const dataUrl = 'data:image/svg+xml;base64,' + Buffer.from(svg, 'utf8').toString('base64');
    await page.setContent(`<html><body style="margin:0;background:#fff"><img id="f" src="${dataUrl}"></body></html>`);
    const ok = await page.evaluate(() => new Promise((resolve) => {
      const img = document.getElementById('f');
      if (img.complete) return resolve(img.naturalWidth > 0);
      img.onload = () => resolve(img.naturalWidth > 0);
      img.onerror = () => resolve(false);
      setTimeout(() => resolve(false), 15000);
    }));
    if (!ok) { console.error(`FAIL ${job.svg}: the browser does not load it as an image`); failed++; continue; }
    const el = await page.$('#f');
    await el.screenshot({ path: job.png, type: 'png', omitBackground: false });
  }
  await browser.close();
  console.log(`${jobs.length - failed} of ${jobs.length} rendered`);
  process.exit(failed ? 1 : 0);
}
main().catch((e) => { console.error(e); process.exit(1); });
