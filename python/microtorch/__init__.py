"""microtorch: research-grade autograd + novel attention mechanisms.

Thin re-export of the pybind11 module so `import microtorch` works after
`pip install .` (see python/setup.py). The C++ extension is the source of
truth; this package adds nothing but the import path.
"""
from _microtorch import (  # noqa: F401
    Variable,
    tensor,
    backward,
    zero_grad,
    load_safetensors,
    save_safetensors,
    nn,
    ops,
)

__version__ = "0.3.1"
