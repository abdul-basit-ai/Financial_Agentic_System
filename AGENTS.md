# Agent instructions

## Worklog requirement

Maintain exactly one worklog at `worklog.md` in the repository root.

- Keep the worklog's goal, current instructions, owners, unfinished work, decisions, and their consequences up to date.
- Reconcile new guidance with work already underway. Do not discard existing entries merely because instructions or priorities change.
- Record decisions, the reasoning that led to them, and the consequences or follow-up work they create.
- Before starting work after compaction or handoff, read `worklog.md` and verify the current repository state.
- Notify affected agents directly, one-to-one, when decisions or instructions change work that an agent owns.
- Update the worklog as state changes, not only at the end of a task.
- Do not maintain duplicate worklogs or create project-local copies.
- Never write comments in the code and if you find any comment remove them, I dont want any comments in any file.

## Worklog format

Use these sections in `worklog.md`:

```markdown
# Worklog

## Goal

## Current instructions

## Owners

## Unfinished work

## Decisions

## State
```

Keep entries concise and factual. Use checkable tasks in `Unfinished work`, and include implementation status in `State`.

## Manager skill

When delegating a task that has more than one peer, give the peers all of the following before they start:

1. **One concrete outcome:** define the deliverable and a measurable completion condition.
2. **Shared constraints:** state protected boundaries, required compatibility, stop conditions, and escalation rules.
3. **Complementary responsibilities:** divide work by an authorized subsystem or artifact; each peer must own substantive work.
4. **Initial write split:** name the files or sections each peer may change. Do not assign overlapping edits.
5. **Assembler:** name one peer responsible for integrating the peers' outputs and reporting the combined result.

Peers must coordinate directly with each other. They may adjust their split within their already authorized subsystem when both agree, but they must not enter another peer's lane or edit overlapping files.

Escalate to the manager instead of acting independently if a proposed change would:

- change the agreed outcome or acceptance criteria;
- cross a protected boundary;
- move ownership into another peer's lane;
- require changing another peer's work or shared interfaces;
- introduce an unresolved dependency or conflict between lanes.

## Coordination skill

For work that can be divided into independent lanes:

- Split the work into parallel sub-agent lanes only when they have independent, concrete outcomes and non-overlapping ownership.
- Give each lane an explicit deliverable, relevant files or surfaces, and an acceptance check.
- Require agents working under shared dependencies to message the relevant peer directly so the dependency can be resolved without routing every detail through the manager.
- Ask agents to report back immediately when one of the following occurs:
  - the agreed outcome or acceptance criteria have changed;
  - a dependency cannot be resolved directly;
  - protected boundaries or lane ownership would need to change;
  - a lane is blocked and the overall plan must be revised.
- Keep coordination factual and narrow; do not duplicate or overwrite another lane's work.
- The designated assembler must integrate the lane outputs, reconcile interface details, run the smallest relevant validation, and verify the final combined result.
