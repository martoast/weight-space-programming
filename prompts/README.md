# Agent prompts (Experiment 8 and 9)

The red team, the blue-team auditors and the improve agent were Claude Code subagents. These are the prompts they
received (paths point at the original machine; the auditors' "rules" section is what kept the experiment blind).
- `red_team.md`: plants 10 sealed bugs in GPT-2 small.
- `blue_team_auditor.md`: one fresh auditor per broken checkpoint (case number substituted). Cases 01-06 received an
  earlier version without the "keep all files in work/" isolation rule; it was added after a shared-scratchpad collision
  (see RESEARCH_NOTES.md, Experiment 8 integrity notes).
- `improve_agent.md`: tries to improve a healthy Qwen2.5-0.5B at math/code by reasoned weight edits.
