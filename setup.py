from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cpp_engine",
        ["submission/cpp_engine.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3", "-funroll-loops"],

    ),
]

setup(
    name="cpp_engine",
    version="0.1.0",
    description="C++ High-Performance Search Engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
