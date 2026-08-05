# Qiita LaTeX-to-Markdown Converter

This directory contains the tools used to combine the numbered LaTeX documents
in `doc/study` into a single Qiita-ready Markdown article.

## Structure

- `main.py`: Entry point. Converts figures, processes `doc/study/[1-4]*.tex`,
  adds the introduction from `0_main.tex`, and writes `combined_output.md`.
- `processor.py`: Parses LaTeX files, tracks labels and counters, and dispatches
  supported environments to the converters.
- `converters.py`: Converts sections, references, links, lists, equations,
  theorem-like environments, proofs, figures, and tables to Markdown.
- `postprocessing.py`: Applies document-wide and Qiita-specific cleanup.
- `citation.py`: Converts `\citep` commands to Markdown footnotes using
  `cite_mapping.json`; it can also regenerate that mapping from `0_main.bbl`.
- `pdf_handler.py`: Converts the first page of figure PDFs to PNG with PyMuPDF.
- `utils.py`: Provides brace parsing and language-branch preprocessing helpers.
- `config.py`: Defines supported environments and shared configuration.
- `image_versions.json`: Stores image URL version numbers for cache busting.
- `combined_output.md`: Generated Qiita article.
