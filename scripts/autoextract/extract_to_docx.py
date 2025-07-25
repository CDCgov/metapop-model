#!/usr/bin/env python3
"""
Script to extract HTML content from measles simulator and convert to Word document.
Runs the Node.js autoextract script and converts output to DOCX format.
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime

import pypandoc


def run_node_extraction(url=None):
    """
    Run the Node.js autoextract script and capture its output.

    Args:
        url (str, optional): URL to extract from. If None, uses default.

    Returns:
        str: HTML content from the extraction
    """
    node_script = "scripts/autoextract/autoextract.js"

    # Build command
    cmd = ["node", node_script]
    if url:
        cmd.append(url)

    try:
        # Run the Node.js script and capture output
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        return result.stdout

    except subprocess.CalledProcessError as e:
        print(f"Error running js script: {e}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print("Error: Node.js not found. Please install Node.js first.")
        print("Install with: npm install -g node")
        print("Or visit: https://nodejs.org/")
        raise


def html_to_docx(html_content, output_path):
    """
    Convert HTML content to a Word document using pypandoc.

    Args:
        html_content (str): HTML content to convert
        output_path (Path): Path where to save the DOCX file
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


def main():
    """Main function to orchestrate the extraction and conversion."""
    try:
        # Parse command line arguments
        url = None
        if len(sys.argv) > 1:
            url = sys.argv[1]

        # Run Node.js extraction
        print("Extracting content from measles simulator...")
        html_content = run_node_extraction(url)

        if not html_content.strip():
            print("Warning: No content extracted from the website")
            return

        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"Measles Simulator Text {timestamp}.docx"

        # Convert to Word document
        print("Converting to Word document...")
        html_to_docx(html_content, output_path)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
