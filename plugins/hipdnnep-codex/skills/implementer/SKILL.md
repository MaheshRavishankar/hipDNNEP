---
name: implementer
description: Implement a hipDNNEP bead in an isolated git worktree, validate both MLIR and non-MLIR paths, and use a separate review pass before signaling ready for human review.
---

# hipDNNEP Implementer

Use this skill when the user asks Codex to implement a bead, pick up the next
task, or carry a hipDNNEP change through coding, review, and validation.

## Core Rules

- Do not modify the main checkout for feature work.
- Create and use a git worktree for the bead.
- Keep bead status current with `br`.
- Run a separate review pass before reporting ready.
- Validate both standard and MLIR builds unless the task is explicitly limited.

## Durable Memory

Do not depend on private agent memory. Persist important state in:

- bead comments via `br comments add`
- git commits
- PR description and replies

## Standard Flow

1. Claim the bead with `br update <bead-id> --status in_progress`.
2. Read the bead details with `br show <bead-id> --json`.
3. Create a worktree on a `users/<author>/...` branch.
4. Create bead-local build directories or `CMakeUserPresets.json` if needed.
5. Implement the change in the worktree, not in the main checkout.
6. Write or update tests.
7. Run a review pass on the diff before final validation.
8. Run:
   - `cmake --preset RelWithDebInfo`
   - `cmake --build --preset RelWithDebInfo`
   - `ctest --preset RelWithDebInfo --output-on-failure`
   - `cmake --preset RelWithDebInfo-MLIR`
   - `cmake --build --preset RelWithDebInfo-MLIR`
   - `ctest --preset RelWithDebInfo-MLIR --output-on-failure`
9. Record the result in bead comments and stop for human review unless the user
   explicitly asks to continue to PR creation or landing.

## Isolated Codex Execution

For a fully isolated implementer session, prefer the Codex sandbox launcher:

```bash
scripts/hipdnn-codex-sandbox.sh <bead-id> -- "<prompt-or-codex-args>"
```

This mirrors the Claude sandbox workflow without replacing it.
