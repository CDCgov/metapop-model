import puppeteer from 'puppeteer';
import path from 'path';

const args = process.argv.slice(2);
let url;
if (args.length == 1) {
    url = args[0];
} else {
    url = 'https://rconnect.edav.cdc.gov/measles-simulator-dev'
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

  // Navigate the page to a URL
  await page.goto(url, { waitUntil: 'networkidle2' });

  // Wait for the page to load completely
  await page.waitForSelector('.stVegaLiteChart', { timeout: 10000 });

  // Remove SVG elements to avoid issues with docx conversion
  await page.evaluate(() => {
    const svgElements = document.querySelectorAll('svg');
    svgElements.forEach(element => {
      element.remove();
    });
  });

  const content = await page.$$eval('.stText, .stHeading, .stMarkdown, .stDataFrame', elements => {
    return elements.map(el => el.outerHTML).join('');
  });

  console.log('<html><body>' + content + '</body></html>');

  await browser.close();
})();
