"""KCWI spectroscopy reduction pipeline (modular).

Core philosophy:
- Each major action is a *step* that reads inputs and writes outputs.
- A small state file tracks which steps completed successfully.
- Steps are implemented as plain Python functions/classes so you can add new ones
  (e.g., cosmic-ray rejection) without rewriting the whole pipeline.
"""

__all__ = []
