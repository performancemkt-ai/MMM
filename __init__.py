"""Bayesian MMM package.

This package provides tools for building Bayesian marketing mix models
for multi‑channel advertising data.  It includes helper functions
for loading datasets, transforming media inputs, specifying
hierarchical models, and optimizing budgets based on model results.

Modules
-------
transforms
    Functions for adstock and saturation transforms.
model_hierarchical
    Hierarchical Bayesian MMM using monthly intercepts.

"""

from . import transforms  # noqa: F401
from . import model_hierarchical  # noqa: F401