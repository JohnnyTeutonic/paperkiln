"""pip install for the microtorch Python bindings.

Drives the CMake build with MICROTORCH_BUILD_PYTHON=ON and drops the
resulting _microtorch extension next to the pure-python package.

    cd python && pip install .
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension

ROOT = Path(__file__).resolve().parent.parent


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_dir = ROOT / "build_py"
        build_dir.mkdir(exist_ok=True)
        subprocess.check_call(
            [
                "cmake", "..",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DMICROTORCH_BUILD_PYTHON=ON",
                f"-DPYTHON_EXECUTABLE={sys.executable}",
            ],
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "_microtorch", "-j"],
            cwd=build_dir,
        )
        # Locate the built extension (name varies by platform/ABI tag).
        built = list(build_dir.glob("_microtorch*.so")) + list(
            build_dir.glob("_microtorch*.pyd")
        )
        if not built:
            raise RuntimeError("pybind11 module not produced by CMake build")
        dest = Path(self.get_ext_fullpath(ext.name))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built[0], dest)


setup(
    name="paperkiln",
    version="0.3.1",
    description="Research-grade autograd + novel attention mechanisms",
    author="Jonathan Reich",
    packages=["microtorch"],
    ext_modules=[Extension("_microtorch", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.9",
    install_requires=["numpy"],
)
