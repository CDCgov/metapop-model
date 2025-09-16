#!/usr/bin/env python3
"""
Script to extract HTML content from measles simulator and convert to Word document.
Uses Playwright for web scraping and pypandoc for document conversion.
"""

import asyncio
import os
import sys
import tempfile
from datetime import datetime

try:
    import pypandoc
    from playwright.async_api import async_playwright
except ImportError:
    print("Missing required packages. Please install with:")
    print("pip install pypandoc playwright")
    print("playwright install")
    sys.exit(1)


async def st_open_expander(element):
    """Open a Streamlit expander if it's not already open."""
    is_open = await element.query_selector("[open]")
    if not is_open:
        toggle_icon = await element.query_selector(
            '[data-testid="stExpanderToggleIcon"]'
        )
        if toggle_icon:
            await toggle_icon.scroll_into_view_if_needed()
            await toggle_icon.click()


async def st_find_and_open_expanders(page):
    """Find and open all Streamlit expanders on the page."""
    expanders = await page.query_selector_all(".stExpander")
    for expander in expanders:
        await st_open_expander(expander)


async def rm_svg(element):
    """Remove SVG elements from an element and return clean HTML."""
    return await element.evaluate("""
        el => {
            // Clone the element to avoid modifying the original
            const clone = el.cloneNode(true);
            // Remove all SVG elements from the clone
            const svgs = clone.querySelectorAll('svg');
            svgs.forEach(svg => svg.remove());
            return clone.outerHTML;
        }
    """)


async def extract_html_content(url=None):
    """
    Extract HTML content from the measles simulator using Playwright.

    Args:
        url (str, optional): URL to extract from. If None, uses default.

    Returns:
        str: HTML content from the extraction
    """
    if not url:
        url = "https://cdcposit.cdc.gov/measles-simulator/"

    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            args=[
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
                "--allow-running-insecure-content",
            ]
        )

        page = await browser.new_page()

        try:
            # Navigate to the page
            await page.goto(url, wait_until="networkidle")

            # Wait for page to load completely
            await asyncio.gather(
                page.wait_for_selector(".stVegaLiteChart", timeout=10000),
                page.wait_for_selector(".stElementToolbar", timeout=10000),
            )

            # Open all expanders
            await st_find_and_open_expanders(page)

            # Elements to extract
            elements_to_extract = [
                ".stText",
                ".stHeading",
                ".stMarkdown",
                ".stDataFrame",
                '[data-testid="stWidgetLabel"]',
                ".stTooltipIcon > .stTooltipHoverTarget",
            ]

            elems = await page.query_selector_all(", ".join(elements_to_extract))

            content = ""

            for el in elems:
                is_tooltip_target = await el.evaluate(
                    'el => el.classList.contains("stTooltipHoverTarget")'
                )

                if is_tooltip_target:
                    # Find SVG inside tooltip target
                    svg_el = await el.query_selector("svg")
                    await svg_el.hover()

                    await page.wait_for_selector(".stTooltipContent", timeout=2000)

                    el = await page.query_selector(".stTooltipContent")

                    # Clear tooltip after capturing
                    await page.hover("#cdc-measles-outbreak-simulator")
                    await page.wait_for_function(
                        '() => !document.querySelector(".stTooltipContent")',
                        timeout=2000,
                    )

                # Get HTML with SVGs removed
                html_content = await rm_svg(el)
                content += html_content

            return f"<html><body>{content}</body></html>"

        finally:
            await browser.close()


def html_to_docx(html_content, output_path):
    """
    Convert HTML content to a Word document using pypandoc.

    Args:
        html_content (str): HTML content to convert
        output_path (str): Path where to save the DOCX file
    """
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False
    ) as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        # Convert HTML file to DOCX using pypandoc
        pypandoc.convert_file(temp_file_path, "docx", outputfile=str(output_path))
        print(f"Document saved: {output_path}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


async def main():
    """Main function to orchestrate the extraction and conversion."""
    try:
        # Parse command line arguments
        url = None
        if len(sys.argv) > 1:
            url = sys.argv[1]

        # Extract content using Playwright
        print("Extracting content from measles simulator...")
        html_content = await extract_html_content(url)

        if not html_content.strip():
            print("Warning: No content extracted from the website")
            return

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/Measles Simulator Text {timestamp}.docx"

        # Convert to Word document
        print("Converting to Word document...")
        html_to_docx(html_content, output_path)

        print(f"Successfully created: {output_path}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
