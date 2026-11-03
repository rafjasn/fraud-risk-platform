"""Versioned contracts shared by runtime jobs and serving (no MLflow imports).

Submodules import only pydantic and each other, so every image can load them.
"""
