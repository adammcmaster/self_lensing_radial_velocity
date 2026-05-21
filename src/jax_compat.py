import logging

logger = logging.getLogger(__name__)


def install_quax_jax_stage_patch():
    """Patch quax for JAX versions that route array construction through stage()."""
    try:
        import jax.numpy as jnp
        import quax._core as quax_core
    except ImportError:
        return

    if "stage_value" in quax_core._QuaxTrace.__dict__:
        return

    def stage_value(self, val):
        return quax_core._QuaxTracer(
            self,
            quax_core._DenseArrayValue(jnp.asarray(val)),
        )

    quax_core._QuaxTrace.stage_value = stage_value
    logger.debug("Installed quax/JAX stage compatibility patch")
