from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

extra_compile = ["-O3", "-funroll-loops"]
extra_link = []
if sys.platform != "win32":
    extra_compile.append("-pthread")
    extra_link.append("-pthread")

ext_modules = [
    Pybind11Extension(
        "cpp_engine",
        [
            "code/codecs.cpp",
            "code/analysis.cpp",
            "code/corpus_io.cpp",
            "code/bindings.cpp",
        ],
        include_dirs=["code"],
        cxx_std=17,
        extra_compile_args=extra_compile,
        extra_link_args=extra_link,
    ),
]

setup(
    name="cpp_engine",
    version="0.1.0",
    description="C++ High-Performance Search Engine",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
