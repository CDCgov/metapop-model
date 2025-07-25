import puppeteer from 'puppeteer';

const args = process.argv.slice(2);
let url;
if (args.length == 1) {
    url = args[0];
} else {
    url = 'https://rconnect.edav.cdc.gov/measles-simulator-dev'
}

async function st_open_expander(el) {
  const isopen = await el.$('[open]');
  if (!isopen) {
    el = await el.$('[data-testid="stExpanderToggleIcon"]');
    el.scrollIntoView();
    el.click();
  }
}

async function st_find_and_open_expanders(page) {
  for (const e of await page.$$('.stExpander')) {
    await st_open_expander(e);
  }
}

async function rm_svg(el) {
  return await el.evaluate(el => {
    const clone = el.cloneNode(true);
    const svgs = clone.querySelectorAll('svg');
    svgs.forEach(svg => svg.remove());
    return clone.outerHTML;
  });
}

(async () => {
  // Launch the browser and open a new blank page
  const browser = await puppeteer.launch({
    args: [
      '--ignore-certificate-errors',
      '--ignore-ssl-errors',
      '--allow-running-insecure-content',
    ]
  });
  const page = await browser.newPage();

  await page.goto(url, { waitUntil: 'networkidle2' });

  // Wait for the page to load completely
  await Promise.all([
    page.waitForSelector('.stVegaLiteChart', { timeout: 10000 }),
    page.waitForSelector('.stElementToolbar', { timeout: 10000 }),
  ]);

  await st_find_and_open_expanders(page);

  // Elements to extract - using specific selectors to avoid duplicates
  const elementsToExtract = [
    '.stText',
    '.stHeading',
    '.stMarkdown',
    '.stDataFrame',
    '[data-testid="stWidgetLabel"]',
    '.stTooltipHoverTarget'
  ];

  const elems = await page.$$(elementsToExtract.join(', '));

  let content = '';

  for (let el of elems) {

      let last_ok = true;
      const isTooltipTarget = await el.evaluate(el => el.classList.contains('stTooltipHoverTarget'));

      if (isTooltipTarget) {

          // try {
            el = await el.$('svg');
            await el.hover();

            // Some tooltips don't have content, e.g., those in the dataframe widget
            try {
              await page.waitForSelector('.stTooltipContent', { timeout: 2000 });
            } catch {
              last_ok = false;
              continue;
            }

            el = await page.$('.stTooltipContent');

            // Clear tooltip after capturing
            await page.hover('#cdc-measles-outbreak-simulator');
            await page.waitForFunction(() => !document.querySelector('.stTooltipContent'), { timeout: 2000 });

      }

      if (last_ok) {
        // Get HTML with SVGs removed
        let htmlContent = await rm_svg(el);
        content += htmlContent;
      }

  }

  console.log('<html><body>' + content + '</body></html>');

  await browser.close();

})();
