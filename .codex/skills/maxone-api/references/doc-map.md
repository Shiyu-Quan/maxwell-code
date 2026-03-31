# Doc Map

Use this file to jump into the bundled mirror without loading unrelated pages. If the workspace has a fresher mirror, the search script can use that instead.

## Mirror Layout

- `docs/index.html`: overview of the Python and C++ split, plus section entry points.
- `docs/tutorial_index.html`: tutorial landing page.
- `docs/tutorial/python_api_tutorial.html`: Python experiment walkthrough from initialization through recording.
- `docs/tutorial/closed_loop_tutorial.html`: Python setup plus C++ closed-loop streaming and triggering.
- `docs/section_api/subsections/api_python.html`: Python API reference.
- `docs/section_api/subsections/api_cpp.html`: C/C++ API reference.
- `docs/section_faq/subsections/api_python.html`: Python FAQ and hardware caveats.
- `docs/section_faq/subsections/api_cpp.html`: C++ FAQ and filter/performance notes.
- `docs/examples/python/`: Python example scripts for stimulation, simultaneous stimulation, interleaving, and saving.
- `docs/examples/cpp/`: C++ examples for raw and filtered streaming.
- `docs/glossary.html`: terminology.

## Task To Page Routing

- Initialization, utilities, multiwell helpers, or `mx.send(...)`: open `api_python.html`.
- Array routing, stimulation-unit allocation, DAC usage, or sequence setup: open `api_python.html` and `api_python` FAQ.
- Saving raw traces or recording groups: open `api_python.html` and the Python tutorial.
- Real-time triggers, `sendSequence`, or stream consumers: open `api_cpp.html` and the closed-loop tutorial.
- FIR vs IIR, amplitude units, `.cfg` parsing, or toolchain issues: open the relevant FAQ page.
- Need runnable examples: open files under `docs/examples/python/` or `docs/examples/cpp/`.

## Prefer Search Before Deep Reads

Run the search script when the exact page is unclear:

```bash
python scripts/search_docs.py --query "query_stimulation_at_electrode"
python scripts/search_docs.py --query "FIR IIR latency" --section cpp
python scripts/search_docs.py --query "wheel virtual environment" --section faq
```

The script resolves docs roots in this order:

- `MAXONE_DOCS_ROOT`
- `./docs` in the current workspace
- `docs/` inside this skill
