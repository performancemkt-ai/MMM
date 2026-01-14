"""Transform utilities for the Bayesian MMM.

This module implements helper functions used by the MMM models
to transform the raw advertising spend into adstocked and saturated
inputs.  We follow the common practice of using a geometric adstock
function to account for carry‐over effects and a Hill saturation
function to model diminishing returns at higher spend levels.

Functions
---------
geometric_adstock(x, theta)
    Apply a recursive geometric adstock with decay parameter `theta`.

hill_saturation(adstocked, alpha, gamma)
    Apply a Hill saturation transform to adstocked media.

These functions are independent of PyMC and operate on PyTensor
variables when called from the model. They also work with NumPy
arrays for convenience in standalone testing.

References
----------
The implementation is adapted from open source MMM examples and
conforms to the definitions in the measured.com MMM documentation.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt


def geometric_adstock(x: pt.TensorVariable | np.ndarray, theta: pt.TensorVariable | float) -> pt.TensorVariable:
    """Compute a geometric adstock of the input time series.

    Parameters
    ----------
    x : array-like
        Raw advertising spend or exposure series (after scaling).
    theta : scalar in (0, 1)
        Decay parameter controlling the carry‑over.  A higher value
        implies longer memory.

    Returns
    -------
    adstocked : PyTensor variable
        The adstocked time series.
    """
    # When used with PyTensor variables, we rely on its scan function.
    # If input is a NumPy array, we convert to PyTensor for consistency.
    x_var = x if isinstance(x, pt.TensorVariable) else pt.as_tensor_variable(x)
    # Define the recursive update: s[t] = x[t] + theta * s[t-1]
    def _step(curr_x, prev_s, theta):
        return curr_x + theta * prev_s
    adstocked, _ = pt.scan(fn=_step,
                           sequences=x_var,
                           outputs_info=pt.as_tensor_variable(0.0),
                           non_sequences=theta)
    return adstocked


def hill_saturation(adstocked: pt.TensorVariable | np.ndarray,
                    alpha: pt.TensorVariable | float,
                    gamma: pt.TensorVariable | float) -> pt.TensorVariable:
    """Apply a Hill saturation function to an adstocked series.

    The Hill function models diminishing returns as spend increases.

    Parameters
    ----------
    adstocked : array-like
        Adstocked media input.
    alpha : positive scalar
        Controls the curvature of the response.
    gamma : positive scalar
        Controls the slope of the response.

    Returns
    -------
    saturated : PyTensor variable
        The saturated media input.
    """
    x_var = adstocked if isinstance(adstocked, pt.TensorVariable) else pt.as_tensor_variable(adstocked)
    return 1.0 / (1.0 + (alpha / (x_var + 1e-12)) ** (1.0 / gamma))