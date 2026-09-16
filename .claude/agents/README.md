# .claude/agents/

Claude Code **subagent** definitions (`.md` files with YAML frontmatter: `name`, `description`, `tools`, `model`). These are tool-level agents invoked via the `Agent` tool inside Claude Code sessions.

Not to be confused with [`agents/`](../../agents/) at the project root, which documents the research project's own multi-agent pipeline (data/training/evaluation agents), not Claude Code tooling.

Example frontmatter:

```markdown
---
name: my-subagent
description: When to use this subagent
tools: Read, Grep, Glob
model: sonnet
---

Instructions for the subagent.
```
