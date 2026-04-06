---
name: pm
description: Sweep hipDNNEP beads and PRs, classify what needs action, and orchestrate implementer work without doing implementation directly.
---

# hipDNNEP Program Manager

Use this skill when the user asks for a sweep of active work, wants to process
pending beads, or wants Codex to determine what needs attention across beads
and PRs.

## Gather State

Run:

```bash
br list --json
br ready --json
br blocked --json
git -C /home/mahesh/onnxruntime/hipDNNEP worktree list
```

For beads with PRs, inspect:

```bash
gh pr view <number> --repo MaheshRavishankar/hipDNNEP --json state,mergeable,reviewDecision,statusCheckRollup
gh api repos/MaheshRavishankar/hipDNNEP/pulls/<number>/comments --paginate
gh api repos/MaheshRavishankar/hipDNNEP/pulls/<number>/reviews --paginate
```

## Classification

- Merged PR, bead still open: clean up worktree, branch, build dirs, and close bead.
- New unaddressed review feedback: queue implementer follow-up.
- Failing code-related checks: queue implementer investigation.
- Ready bead with no PR: queue implementation.
- Approved PR with passing checks: report ready to land.
- Blocked bead: report blocker only.

## Execution Rule

Do not implement code in the PM sweep. Delegate implementation work to the
`implementer` workflow, preferably in isolated Codex sandbox sessions:

```bash
scripts/hipdnn-codex-sandbox.sh <bead-id> -- "<task prompt>"
```

Launch independent beads in parallel when they do not overlap.
