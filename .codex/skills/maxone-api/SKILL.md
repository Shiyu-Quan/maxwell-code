---
name: maxone-api
description: Assist with MaxOne/MaxTwo MaxLab Live API programming. Use when Codex needs to design, write, review, debug, or explain Python `maxlab` control scripts, C/C++ closed-loop streaming code, stimulation/routing logic, recording and saving flows, or FAQ-style troubleshooting against the bundled mirror of `https://api-docs.mxwbio.com`, with optional override from a workspace `docs/` directory or `MAXONE_DOCS_ROOT`.
---

# MaxOne API

Assume the skill ships with a bundled docs mirror under `docs/` inside the skill directory. If the current workspace also has a `docs/` mirror, or `MAXONE_DOCS_ROOT` is set, the search script may use that instead. Keep `SKILL.md` lean: read one small reference first, then use the search script for exact names or snippets, and only then open raw HTML pages if needed.

## Route The Request

- Read `references/python-api.md` for Python control scripts, `mx.Array`, `mx.Sequence`, `mx.StimulationUnit`, `mx.DAC`, `mx.Saving`, initialization, stimulation, and multiwell utilities.
- Read `references/cpp-api.md` for `DataStreamerRaw_*`, `DataStreamerFiltered_*`, `sendSequence`, `FilterType`, `Status`, and closed-loop C/C++ patterns.
- Read `references/workflows.md` for end-to-end experiment structure, example-file routing, and mixed Python+C++ closed-loop setup.
- Read `references/faq.md` for routing pitfalls, DAC limits, `.cfg` handling, LSB conversion, concurrency cautions, and filter/performance questions.
- Read `references/doc-map.md` only when you need page-level navigation into the bundled mirror.
- Run `python scripts/search_docs.py --query "..."` when you need exact API names, nearby wording from the mirror, or the best page to open next.

## Follow This Workflow

1. Classify the task.
- Distinguish Python control, C++ real-time streaming, mixed Python+C++ closed-loop, or FAQ/troubleshooting.
- Distinguish MaxOne vs MaxTwo, single-well vs multiwell, and whether stimulation, recording, or raw trace saving is involved.

2. Load the smallest useful context.
- Start with one reference file.
- Run the search script before reading multiple references or raw mirrored pages.
- Open raw HTML under the bundled `docs/` mirror only when the references and search results still leave ambiguity.

3. Write or review code with MaxOne-specific constraints.
- Prefer `import maxlab as mx` in Python.
- Treat `ApiObject`-derived commands as declarative objects that must be sent via `mx.send(...)`; do not handcraft protocol messages.
- Split Python code into explicit phases: initialization, electrode selection/routing, stimulation-unit setup, sequence construction, saving/recording, run/cleanup.
- Treat C++ closed-loop as a runtime trigger layer on top of Python-prepared sequences and routing.
- Use `verifyStatus` only for unrecoverable C++ errors because it exits the process.

4. Make assumptions explicit.
- State hardware assumptions that are not derivable from the request: device type, active well, stimulation amplitude units, electrode/channel mapping, and whether mxwserver is already running.
- Call out hardware risks explicitly: unrouted stimulation electrodes, multiple electrodes on one stimulation unit, DAC simultaneity limits, offset timing, and stream-close cleanup.

## Honor These Defaults

- Default lookup order is `MAXONE_DOCS_ROOT`, then workspace `docs/`, then the bundled `docs/` mirror.
- Prefer Python unless the request requires real-time closed-loop response or stream-level processing.
- Assume concurrent unsynchronized access to the same `mxwserver` is unsafe.
- Avoid inventing electrode mappings or sequence names; derive them from the request or mark them as placeholders.

## Use The Search Script

```bash
python scripts/search_docs.py --query "group_define"
python scripts/search_docs.py --query "DataStreamerFiltered_open" --section cpp
python scripts/search_docs.py --query "offset compensation" --section faq --limit 5
```
