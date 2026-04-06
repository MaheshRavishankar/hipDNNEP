---
name: review
description: Review hipDNNEP diffs for correctness, memory safety, API misuse, project convention violations, and over-specialized designs.
---

# hipDNNEP Review

Use this skill when reviewing a diff, PR, or changed files in hipDNNEP.

## Scope

- PR number: `gh pr diff <number>`
- File paths: read the files and their git diff
- Default: `git diff` and `git diff --cached`

Always read [AGENTS.md](../../../../AGENTS.md) and the touched files before
reviewing.

## Review Priorities

1. Conventions
   - clang-format and project style
   - file-local helpers in anonymous namespaces
   - implementation details kept out of headers
   - early-return validation style
   - lit/FileCheck conventions
2. Correctness
   - callback wiring
   - lifecycle ordering
   - ownership and null handling
   - MLIR pass or region correctness
3. Memory safety
   - HIP allocation/free pairing
   - device context setup
   - locking around shared state
   - dangling ownership boundaries
4. API usage
   - ORT EP API contracts
   - hipDNN and hipBLAS-LT lifecycle handling
   - MLIR pass registration and dialect loading
5. Generalization
   - hardcoded rank or dtype assumptions
   - op-specific logic in generic code paths
   - rigid support checks without justification
   - tests that under-cover claimed support
6. Test integrity
   - verify there is evidence that all relevant tests passed
   - treat failing CI or local test evidence as a blocker
   - flag any disabled, skipped, or expected-fail tests used to avoid fixing code
   - specifically flag `UNSUPPORTED`, `XFAIL`, `DISABLED_`, and CMake-level test exclusions

Only report issues you can support from the current diff and code.

## Hard Rules

- The change is not ready if relevant tests are failing.
- The change is not ready if there is no evidence that relevant tests passed.
- Never accept disabling, skipping, or marking tests expected-fail to get a
  change through review.

## Output

Report findings first, ordered by severity, with `file:line` references.
If no findings meet the threshold, state that explicitly.
