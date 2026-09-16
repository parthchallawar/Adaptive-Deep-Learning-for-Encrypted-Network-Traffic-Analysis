# agent-memory/

Persistent memory for each agent defined in [`agents/`](../agents/). One subfolder per agent:

```
agent-memory/
  <agent-name>/
    memory.md       # running notes / decisions / state
    runs/           # optional: per-run logs or artifacts
```

This is project-local, versionable memory for the project's own agents — separate from Claude Code's own session memory system.
