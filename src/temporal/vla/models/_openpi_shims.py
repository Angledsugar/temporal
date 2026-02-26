"""Shim modules to allow importing OpenPI PyTorch code without JAX/Flax.

OpenPI's PyTorch models have transitive imports to JAX/Flax modules
(gemma.py → flax, image_tools.py → jax). These shims pre-register
stub modules in sys.modules so that Python never tries to import the
real JAX/Flax packages.

Usage:
    from temporal.vla.models._openpi_shims import install_shims
    install_shims()
    # Now OpenPI PyTorch imports work without JAX
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field


def install_shims() -> None:
    """Install all shims needed for JAX-free OpenPI PyTorch imports."""
    _install_jax_stubs()
    _install_openpi_model_shims()


def _make_stub_module(name: str) -> types.ModuleType:
    """Create a stub module with a proper __spec__ so importlib doesn't choke."""
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    mod.__path__ = []  # Make it a package
    return mod


def _install_jax_stubs() -> None:
    """Install minimal JAX/Flax stubs so Python doesn't error on import."""
    if "jax" in sys.modules:
        return  # Real JAX is available, no need for stubs

    # --- JAX stubs ---
    jax = _make_stub_module("jax")
    jax.jit = lambda *a, **kw: (lambda f: f) if not a else a[0]  # no-op decorator
    jax.numpy = _make_stub_module("jax.numpy")
    jax.image = _make_stub_module("jax.image")

    class _FakeResizeMethod:
        LINEAR = "linear"
    jax.image.ResizeMethod = _FakeResizeMethod
    jax.image.resize = lambda *a, **kw: None

    # jax._src and jax.core (used by array_typing.py)
    jax._src = _make_stub_module("jax._src")
    jax._src.tree_util = _make_stub_module("jax._src.tree_util")
    jax._src.tree_util.equality_errors = lambda *a, **kw: []
    jax.core = _make_stub_module("jax.core")
    jax.tree_util = _make_stub_module("jax.tree_util")
    jax.tree_util.keystr = lambda x: str(x)
    jax.tree_util.tree_map_with_path = lambda *a, **kw: None
    jax.typing = _make_stub_module("jax.typing")
    jax.typing.ArrayLike = object
    jax.Array = object

    sys.modules["jax"] = jax
    sys.modules["jax.numpy"] = jax.numpy
    sys.modules["jax.image"] = jax.image
    sys.modules["jax._src"] = jax._src
    sys.modules["jax._src.tree_util"] = jax._src.tree_util
    sys.modules["jax.core"] = jax.core
    sys.modules["jax.tree_util"] = jax.tree_util
    sys.modules["jax.typing"] = jax.typing

    # --- Flax stubs ---
    flax = _make_stub_module("flax")
    flax.__version__ = "0.0.0"
    flax.linen = _make_stub_module("flax.linen")
    flax.struct = _make_stub_module("flax.struct")

    # flax.linen — minimal stubs
    flax.linen.Module = object
    flax.linen.compact = lambda f: f
    flax.linen.initializers = _make_stub_module("flax.linen.initializers")
    flax.linen.initializers.normal = lambda stddev=0.01: None

    # flax.struct.dataclass — just use stdlib dataclass
    import dataclasses
    flax.struct.dataclass = dataclasses.dataclass

    sys.modules["flax"] = flax
    sys.modules["flax.linen"] = flax.linen
    sys.modules["flax.struct"] = flax.struct
    sys.modules["flax.linen.initializers"] = flax.linen.initializers

    # --- jaxtyping stub (used by openpi.shared.array_typing) ---
    jaxtyping = _make_stub_module("jaxtyping")

    class _TypeStub:
        """Stub for jaxtyping type annotations (UInt8, Float, etc.)."""
        def __class_getitem__(cls, item):
            return cls
    jaxtyping.UInt8 = _TypeStub
    jaxtyping.Float = _TypeStub
    jaxtyping.Array = _TypeStub
    jaxtyping.ArrayLike = _TypeStub
    jaxtyping.Bool = _TypeStub
    jaxtyping.DTypeLike = _TypeStub
    jaxtyping.Int = _TypeStub
    jaxtyping.Key = _TypeStub
    jaxtyping.Num = _TypeStub
    jaxtyping.Real = _TypeStub
    jaxtyping.PyTree = _TypeStub
    jaxtyping.jaxtyped = lambda *a, **kw: (lambda f: f)

    # jaxtyping.config
    class _JaxtypingConfig:
        jaxtyping_disable = False
        def update(self, key, value):
            setattr(self, key, value)
    jaxtyping.config = _JaxtypingConfig()

    # jaxtyping._decorator
    jaxtyping._decorator = _make_stub_module("jaxtyping._decorator")
    jaxtyping._decorator._check_dataclass_annotations = lambda self, tc: None

    sys.modules["jaxtyping"] = jaxtyping
    sys.modules["jaxtyping._decorator"] = jaxtyping._decorator

    # --- beartype stub ---
    beartype = _make_stub_module("beartype")
    beartype.beartype = lambda f: f  # no-op decorator
    beartype.roar = _make_stub_module("beartype.roar")
    sys.modules["beartype"] = beartype
    sys.modules["beartype.roar"] = beartype.roar


def _install_openpi_model_shims() -> None:
    """Pre-register openpi.models.gemma and openpi.models.lora with JAX-free versions."""
    if "openpi.models.gemma" in sys.modules:
        return  # Already loaded (e.g. real JAX is available)

    from temporal.vla.models._gemma_config import get_config, GemmaConfig

    # Ensure openpi.models namespace exists
    if "openpi.models" not in sys.modules:
        models_mod = types.ModuleType("openpi.models")
        models_mod.__path__ = []
        sys.modules["openpi.models"] = models_mod

    # lora shim
    lora_shim = types.ModuleType("openpi.models.lora")

    @dataclass
    class LoRAConfig:
        rank: int
        alpha: float = 1.0
    lora_shim.LoRAConfig = LoRAConfig
    sys.modules["openpi.models.lora"] = lora_shim

    # gemma shim
    gemma_shim = types.ModuleType("openpi.models.gemma")
    gemma_shim.get_config = get_config
    gemma_shim.Config = GemmaConfig
    sys.modules["openpi.models.gemma"] = gemma_shim
