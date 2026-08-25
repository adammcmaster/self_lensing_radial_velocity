"""CPU-only materialisation for stage-1 GPU prefit batches."""

import os


def initialise():
    """Select CPU before importing JAX-dependent prefit code in this process."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def materialize(payload):
    from prefit import materialize_prefit_batch

    return materialize_prefit_batch(payload)
