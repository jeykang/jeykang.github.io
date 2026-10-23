"""Compatibility shim: the open-loop evaluator lives in evaluator.py (TERMINOLOGY.md, renamed 2026-10-06).

Re-exports every name, private ones included, so `import harness` and `from harness import ...`
keep working for code and notebooks written before the rename.
"""
import evaluator as _evaluator

globals().update({k: v for k, v in vars(_evaluator).items() if not k.startswith("__")})
