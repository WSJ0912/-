"""Competition pipeline for the ``cxr_first`` 10-class chest X-ray task.

This package is intentionally separate from :mod:`cxr_research`: the competition
dataset uses ten numeric class ids instead of the platform's CheXpert-14 label
contract, and the competition rules forbid medical pretraining. Sharing the
codebase would blur both contracts, so the training, evaluation, and submission
code lives here.
"""

from __future__ import annotations

__version__ = "0.1.0"
