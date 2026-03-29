# Agent Instructions

These instructions apply to any LLM or AI agent working in this repository.

## Tracking Files — Always Update

After completing any implementation work, **always** update both tracking files before finishing:

### 1. `CHANGELOG.md` (repo root)
Add an entry under `[Unreleased]` describing what was added or changed. Use standard Keep a Changelog conventions:
- `Added` — new files or features
- `Changed` — modifications to existing code or config
- `Fixed` — bug fixes

### 2. `plans/implementation-status.md`
- Update the **status** column for the relevant stage: `pending` → `in-progress` → `done` (or `blocked`)
- Append a row to the **Change Log** table with today's date, the stage number, and a short description of what was done

**Why this matters:** Multiple LLM agents are executing this plan across separate conversations. These files are the shared coordination layer. Without updates, progress tracking breaks down.

## Plan Reference

The full multi-stage implementation plan is at [`plans/implementation-plan.md`](plans/implementation-plan.md). Read it before starting any stage.
