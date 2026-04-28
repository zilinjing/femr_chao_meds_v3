# Repository Guidelines

## Project Structure & Module Organization
- `src/femr/` — Python package code. Key submodules: `models/`, `featurizers/`, `labelers/`, `transforms/`, `post_etl_pipelines/`.
- `tests/` — pytest suite organized by package area (e.g., `tests/models/`).
- `tools/` — utility scripts (e.g., Stanford OMOP helpers).
- `tutorials/` — notebooks and examples; excluded from some lint checks.

## Build, Test, and Development Commands
- Dev activate: `python conda activate tte`


## Testing Guidelines
- Framework: pytest. Place tests under `tests/` mirroring package paths.
- Naming: files `test_*.py`, functions `test_*`.
- Add unit tests for new logic and regression tests for fixed bugs.
- Prefer small, deterministic tests; use fixtures in `tests/femr_test_tools.py` when applicable.


## Security & Configuration Tips
- Do not commit PHI, datasets, or large artifacts (wandb runs, logs). Use `.gitignore` and environment variables for paths/secrets.
- Reproducibility: record seeds, versions, and command lines in PR descriptions for modeling changes.

## Communication style
- Be objective, rigorous, and critical.
- Do not flatter, overpraise, or agree too quickly.
- Maintain a sober, evidence-seeking tone when discussing ideas; do not present speculative claims as established facts.

## Workflow
- Do not rely only on internal model knowledge for research discussions when external evidence is important.

## Coding workflow
- Before writing substantial code, first propose a minimal implementation plan.
- Prefer the smallest runnable slice first.
- Keep changes localized unless a broader refactor is explicitly requested.

## Tool usage
- Use external tools such as GitHub or paper/document access when external evidence is needed.
- When discussing ideas, related work, or external implementations, proactively search for relevant evidence when it is likely to improve accuracy or sharpen criticism.
- Use subagents only for clearly separable tasks.
- Avoid spawning subagents for casual brainstorming or small edits.
- For complex tasks, it is acceptable to spawn a small number of subagents when doing so materially improves parallel exploration, implementation, or verification.


# Coding instruction

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.


## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

This works for common files shared by other methods like prepare.py, pretrain.py, embedding.py

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

For task-specific files like survivehr_causal.py or survivehr_head.py, you can revise them to be simple, concise, clear and easy to understand

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

