# agents/

Role/persona definitions for this project's own multi-agent pipeline (e.g. a data-ingestion agent, a training agent, an evaluation agent, a drift-monitoring agent) — **not** Claude Code subagent tool definitions, which live in [`.claude/agents/`](../.claude/agents/) instead.

One file per agent role: `agents/<agent-name>.md`, describing its responsibility, inputs/outputs, and how it should use its folder in [`agent-memory/`](../agent-memory/).

Use [`template.md`](./template.md) as the starting point.
