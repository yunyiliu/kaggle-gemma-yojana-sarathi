"""Marks tests/ as a regular package.

Without this file, `from tests.perception_cases import ...` resolves tests/ as a
namespace portion and the import scan continues down sys.path - so on any machine that
already has a `tests` package installed (Kaggle's image does), the installed one wins and
the import fails. An empty __init__.py makes the local directory a regular package, which
takes precedence.
"""
