# SPDX-License-Identifier: GNU Affero General Public License v3.0
# Copyright 2023-present the Unsloth team. All rights reserved.

"""
Grouped GEMM kernels for MoE models.

This module provides TMA compatibility across Triton versions:
- Triton >= 3.6.0: uses `make_tensor_descriptor`
- Triton < 3.4.0: uses `_experimental_make_tensor_descriptor`
"""

import triton.language as tl

# TMA API compatibility shim
# The kernels use `tl._experimental_make_tensor_descriptor` which was the old API name.
# In Triton >= 3.6.0, this was renamed to `tl.make_tensor_descriptor`.
# We patch the module to ensure the old name works with newer Triton versions.

HAS_TMA = False

if hasattr(tl, "make_tensor_descriptor"):
    # Triton >= 3.6.0: new API name
    if not hasattr(tl, "_experimental_make_tensor_descriptor"):
        # Add alias for backward compatibility with kernel code
        tl._experimental_make_tensor_descriptor = tl.make_tensor_descriptor
    HAS_TMA = True
elif hasattr(tl, "_experimental_make_tensor_descriptor"):
    # Triton < 3.4.0: old experimental API
    HAS_TMA = True
else:
    # TMA not available
    HAS_TMA = False

__all__ = ["HAS_TMA"]
