# .aider.conf.yml
model: ollama/gemma-4:26b

# Der Auto-Korrektur Loop mit Ruff
auto-lint: true
lint-cmd:
  - "ruff check --fix ."
  - "ruff format ."

# CLI / Headless Optimierungen
dark-mode: true
auto-commits: true
attribute-author: false
