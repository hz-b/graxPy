"""Extract the first raster figure from a page of the paper's PDF.

This is a provenance helper for the bundled Fig. 4 image.  It only requires
``pypdf`` and can be run with a local copy of the open-access article.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    """Extract the first image from the requested one-based PDF page."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--page", type=int, default=7)
    args = parser.parse_args()
    images = PdfReader(args.pdf).pages[args.page - 1].images
    if not images:
        raise ValueError(f"Page {args.page} has no raster images")
    image = images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(image.data)
    print(f"Extracted {image.name} from page {args.page} to {args.output}")


if __name__ == "__main__":
    main()
