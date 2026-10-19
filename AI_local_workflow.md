# AI Development Workflow

## Purpose

This document defines how substantial development work is administered when using:

- ChatGPT as technical lead / supervisor
- OpenCode CLI/TUI as the persistent implementation environment
- Local and remote models/agents available through OpenCode
- Git as the authoritative source of code history and working state

The objective is to achieve high implementation quality with minimal supervisory
LLM calls, while retaining traceability, test discipline, and control over
repository changes.

---

# 1. Roles

## 1.1 ChatGPT — Technical Lead / Supervisor

ChatGPT is responsible for:

- understanding the user's objective;
- turning the objective into a bounded implementation programme;
- defining architectural constraints;
- deciding what constitutes completion;
- sending substantial autonomous work packages to OpenCode;
- reviewing OpenCode's conclusions, diffs, tests, and evidence;
- intervening only when necessary;
- deciding whether additional investigation or implementation is required;
- reporting important milestones and blockers to the user.

ChatGPT should NOT micromanage OpenCode with many small prompts.

Prefer:

1 initial comprehensive instruction
+
0–3 supervisory interventions

for a normal substantial development task.

---

## 1.2 OpenCode — Implementation Lead

OpenCode is responsible for executing the assigned work package autonomously.

OpenCode should:

- inspect the repository before modifying it;
- understand existing architecture before implementation;
- use appropriate models/subagents when beneficial;
- investigate before assuming root causes;
- modify code;
- add or update tests;
- run tests;
- inspect failures;
- iterate until the defined completion criteria are met;
- inspect the final diff;
- provide an evidence-based completion report.

OpenCode should not stop after every intermediate discovery.

It should continue autonomously unless:

- a genuine architectural decision requires user input;
- required information is unavailable;
- an action would be destructive or irreversible;
- repository ownership/scope is ambiguous;
- continuing would risk overwriting unrelated work.

---

# 2. OpenCode Runtime

Use one persistent OpenCode CLI/TUI process per substantial task.

Typical startup:

```powershell
cd C:\Github\<repository>
opencode.cmd --auto