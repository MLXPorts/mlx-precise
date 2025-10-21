import sys, sysconfig, platform

# Hard requirements: Python 3.14 free-threaded (cp314t), non-GIL runtime, macOS/Metal only
REQ_MAJOR, REQ_MINOR = 3, 14
if (sys.version_info.major, sys.version_info.minor) != (REQ_MAJOR, REQ_MINOR):
    raise RuntimeError(
        f"MLX-precise requires Python {REQ_MAJOR}.{REQ_MINOR} free-threaded only.")

if sysconfig.get_config_var('Py_GIL_DISABLED') != 1:
    raise RuntimeError(
        "MLX-precise must run on a free-threaded Python 3.14 build (Py_GIL_DISABLED != 1)."
        " Install python=3.14 and python-freethreading from conda-forge.")

if platform.system() != 'Darwin':
    raise RuntimeError("MLX-precise (this build) targets macOS/Metal only.")

is_gil_enabled = getattr(sys, '_is_gil_enabled', None)
if is_gil_enabled is None:
    # Free-threaded builds expose this probe; if missing, we consider this unsupported.
    raise RuntimeError("Free-threaded probe not found; ensure Python 3.14 FT is used.")
if is_gil_enabled():
    raise RuntimeError(
        "GIL is enabled at runtime. This build requires non-GIL mode."
        " Start with: python -X gil=0 ...")

# Re-export commonly used subpackages for convenience
from . import nn, optimizers  # noqa: F401
