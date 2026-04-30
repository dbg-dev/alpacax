# alpacax


## Alpaca Dataset Pipeline

Fetches, validates, formats, tokenizes, masks, pads, and splits an Alpaca-format
instruction dataset using a railway-oriented Result pattern for error handling.

Python: 3.12+
Dependencies: pip install requests pydantic returns

Design goals

- Keep I/O at the boundary and make transformations independently testable.
- Validate remote data shape before running the pipeline.
- Avoid unbounded padding by using a configurable maximum sequence length.
- Support either truncating long samples or skipping them with recorded reasons.
- Mask prompt labels by default for causal-LM instruction fine-tuning.
- Return partial success with skipped row errors whenever possible.


* * *

## Project Docs

For how to install uv and Python, see [installation.md](docs/installation.md).

For development workflows, see [development.md](docs/development.md).

For instructions on publishing to PyPI, see [publishing.md](docs/publishing.md).

* * *

*This project was built from
[simple-modern-uv](https://github.com/jlevy/simple-modern-uv).*
