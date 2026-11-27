"""Run with the source checkout and only its requirements.txt dependencies."""

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebooklm_mcp_server import pdf_to_png, png_to_pdf


async def main():
    with TemporaryDirectory() as directory:
        source = Path(directory) / "source.pdf"
        with fitz.open() as doc:
            doc.new_page().insert_text((50, 50), "PDF conversion test")
            doc.save(source)

        images = await pdf_to_png(str(source), dpi=72)
        assert images["page_count"] == 1
        image = Path(images["pages"][0])
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        result = await png_to_pdf(image_directory=images["output_directory"])
        with fitz.open(result["output_path"]) as doc:
            assert doc.is_pdf and doc.page_count == 1
    print("One PDF page converted to PNG and reopened as a one-page PDF.")


if __name__ == "__main__":
    asyncio.run(main())
