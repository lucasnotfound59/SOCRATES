"""Narrow, versioned PyTensor linker compatibility for the formal runner.

Some macOS 15+ probes in the bundled PyTensor add ``-ld64`` to clang++
flags. The host clang rejects that option, so we remove only that generated
flag from PyTensor's compiler arguments. We never alter ``platform.mac_ver``.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


COMPATIBILITY_VERSION = "socrates-pytensor-ld64-v1"
_STATUS: dict[str, Any] = {
    "helper_version": COMPATIBILITY_VERSION,
    "activated": False,
    "reason": "not_configured",
    "pytensor_version": "not-installed",
    "platform": sys.platform,
    "macos_major": None,
}


def _load_compiler() -> Any:
    """Load PyTensor's compiler seam separately for controlled testing."""
    import pytensor.link.c.cmodule as cmodule
    return cmodule.GCC_compiler


def configure_pytensor_compatibility() -> dict[str, Any]:
    """Apply the narrowly scoped linker fix once and return an audit record."""
    if _STATUS["reason"] != "not_configured":
        return dict(_STATUS)
    if sys.platform != "darwin":
        _STATUS["reason"] = "non-darwin"
        return dict(_STATUS)
    try:
        version = importlib.metadata.version("pytensor")
    except importlib.metadata.PackageNotFoundError:
        _STATUS["reason"] = "pytensor-not-installed"
        return dict(_STATUS)
    _STATUS["pytensor_version"] = version
    mac_version = platform.mac_ver()[0]
    try:
        mac_major = int(mac_version.split(".", 1)[0])
    except (AttributeError, ValueError):
        mac_major = None
    _STATUS["macos_major"] = mac_major
    if mac_major is None or mac_major < 15:
        _STATUS["reason"] = "macos-version-not-affected"
        return dict(_STATUS)
    try:
        compiler = _load_compiler()
    except Exception as exc:  # pragma: no cover - incomplete runtime only
        _STATUS["reason"] = f"pytensor-import-failed:{type(exc).__name__}"
        return dict(_STATUS)
    original = compiler.compile_args
    if getattr(original, "_socrates_ld64_compat", False):
        _STATUS.update(activated=True, reason="already-active")
        return dict(_STATUS)
    probe = list(original(march_flags=False))
    if "-ld64" not in probe:
        _STATUS["reason"] = "ld64-not-present"
        return dict(_STATUS)

    def compile_args_without_ld64(march_flags: bool = True) -> list[str]:
        return [flag for flag in original(march_flags=march_flags) if flag != "-ld64"]

    setattr(compile_args_without_ld64, "_socrates_ld64_compat", True)
    compiler.compile_args = staticmethod(compile_args_without_ld64)
    _STATUS.update(activated=True, reason="removed-generated-ld64")
    return dict(_STATUS)


def compatibility_status() -> dict[str, Any]:
    """Return a copy suitable for a run manifest."""
    return dict(_STATUS)
