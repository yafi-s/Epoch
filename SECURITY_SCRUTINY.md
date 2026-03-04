# Security Scrutiny Checklist

This document records the publication-safety checks performed before pushing this repository public.

## Scope

- Repository: `Epoch`
- Publish target: public GitHub repo
- Review date: 2026-03-04

## Checks Performed

- Verified `.gitignore` excludes local secrets and runtime artifacts:
  - `.env*`
  - `.modal.toml`
  - `.claude/`
  - `results/`
  - `*.log`, `*.err.log`, `*.out.log`
  - local tool binaries and IDE files
- Ran staged content pattern scans for:
  - Modal token formats (`ak-...`, `as-...`)
  - ngrok public endpoints (`*.tcp.ngrok.io:*`)
  - token key fields (`token_id`, `token_secret`)
- Ran `gitleaks` scan on commit history:
  - result: no leaks found
- Re-ran test suite before publish:
  - `poetry run pytest -q` -> all passing

## Reviewer Notes

- No secrets were identified in committed files.
- Local-only operational data (tunnels, logs, result dumps, local env files) is excluded from version control.
- Further hardening recommendation:
  - enable GitHub Advanced Security/secret scanning alerts on the repository.
