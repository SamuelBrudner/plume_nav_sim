---
description: make beads quickly
---

# Workflow

- Identify the beads already "under discussion" (from recent context or user prompts).
- For each bead, determine if it is blocked. If not blocked, it's ready to launch.
- For every launchable bead, run one command per bead:

  ```bash
  echo "Please claim and complete bead <ID>. When you are done, update beads that depend on <ID> with any needed context about the work you completed. Then close bead <ID>." | codex exec --full-auto
  ```

- Record what was launched (bead ID + status) in your response to the user.
