# Dedupe Refactor Plan (Entrypoints-First Architecture)

## Why this refactor is needed
The current `Dedupe` file mixes many concerns in one place:
- Google Apps Script UI/menu handlers
- job preparation and upload/download wiring
- large embedded Python runners (`?runner=1`, evolve runner, CD runner)
- model/training/matching logic and diagnostics

When these concerns are coupled, a small string-escaping fix can accidentally affect a distant runtime path. The goal is to make each flow explicit and isolated.

## Target design principle
**Entrypoints first, implementation second.**

At the top of the main script, keep only:
1. Public GAS handlers (menu actions / web app handlers)
2. A single route table for `doGet`/`doPost`
3. Thin wrappers that delegate to dedicated modules

Everything else moves into focused modules.

## Proposed module layout

### 1) `src/entrypoints.gs`
Owns only public callable functions:
- `dl_prepareLocalJobAndShowCommand_`
- `dl_prepareIncrementalTrainingJob_`
- `cd_prepareMatchingJob_`
- `diagnostic_analyzeMatching`
- `doGet`
- `doPost`

Rules:
- No business logic inside these handlers
- No embedded Python arrays here
- Only validation + delegation + user-facing status

### 2) `src/router.gs`
Web app route dispatcher:
- Maps query flags (`runner=1`, `job=1`, `result=1`, `evolve_runner=1`, etc.) to dedicated handlers
- Centralizes error formatting and HTTP responses

### 3) `src/jobs/*.gs`
Separate job builders:
- `training-job-builder.gs`
- `incremental-job-builder.gs`
- `cd-job-builder.gs`
- `evolution-job-builder.gs`

Each builder returns a typed payload object and owns token/url assembly.

### 4) `src/python_templates/*.py.tpl`
Store Python runners as template files instead of massive inline arrays:
- `training_runner.py.tpl`
- `cd_runner.py.tpl`
- `evolution_runner.py.tpl`
- `diagnostic_runner.py.tpl`

Inject shared snippets during build step (or concatenation utility), not by hand-editing giant JS arrays.

### 5) `src/python_shared/unified_matching.py.tpl`
Single shared matching module used by all templates.

### 6) `src/services/*.gs`
Google-side infrastructure utilities:
- Drive IO
- Sheet IO
- chunked upload/post helpers
- progress state
- logging helpers

### 7) `src/domain/*.gs`
Domain-only logic (no transport/UI):
- donor normalization helpers
- blocking rule utilities
- model metadata helpers

## Safe migration sequence

### Step A: Freeze behavior with route-level checks
- Capture baseline route outputs for: `runner`, `job`, `result`, `evolve_runner`
- Record expected content type and shape

### Step B: Introduce router without changing behavior
- Keep existing functions, but pass through a new route map
- Add unknown-route guard with clear error response

### Step C: Extract entrypoint wrappers
- Move public handlers into `entrypoints.gs`
- Delegate back to legacy internals temporarily

### Step D: Extract Python templates first (highest risk area)
- Move runner payloads to `.py.tpl`
- Add template compilation helper that injects shared module
- Validate compiled Python with `py_compile` in CI/local checks

### Step E: Extract job builders and services
- Move URL/token/payload logic out of entrypoints
- Keep interfaces stable

### Step F: Decompose domain helpers
- Move pure functions last, with snapshot tests where feasible

## Guardrails to prevent regressions
- **One owner per route**: exactly one file handles each query mode.
- **No escaped-newline print literals in templates**: use explicit blank-line `print("")` where needed.
- **Template compile check**: generated Python must pass `python3 -m py_compile`.
- **Entrypoint size cap**: entrypoint functions should stay short (e.g., <30 LOC each).
- **No cross-module hidden globals**: pass required state explicitly.

## Practical first PR set

### PR 1 (low risk)
- Add `router.gs` and route table
- Keep legacy logic untouched behind delegates

### PR 2 (medium risk)
- Extract runner templates + compile helper
- Add automated generated-script compile checks

### PR 3 (medium risk)
- Extract job builders and service utilities

### PR 4 (higher risk)
- Move domain logic and remove legacy monolith sections

## Definition of done
- Public functions are grouped at top-level entrypoint file only
- Runner/template code no longer embedded as giant JS string arrays
- Each route has a single handler module
- Generated Python passes compile checks consistently
- Existing workflow order and outputs remain unchanged
