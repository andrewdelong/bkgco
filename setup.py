import sys
from setuptools import setup, Extension

GCO = "cpp/gco"
msvc = sys.platform == "win32"

# hidden visibility: the module's C++ symbols stay private to it
cxx = (
    ["/EHsc", "/wd4244", "/wd4267"]
    if msvc
    else ["-std=c++11", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-Wno-unused-function"]
)


DEPS = [
    GCO + "/GCoptimization.cpp",
    GCO + "/GCoptimization.h",
    GCO + "/GCoptimization.inl",
    GCO + "/LinkedBlockList.cpp",
    GCO + "/LinkedBlockList.h",
    GCO + "/graph.cpp",
    GCO + "/graph.h",
    GCO + "/maxflow.cpp",
    GCO + "/energy.h",
    GCO + "/block.h",
]

setup(
    ext_modules=[
        Extension(
            "bkgco._bkgco",
            sources=["cpp/bindings/module.cpp"],
            depends=DEPS,
            include_dirs=[GCO],
            extra_compile_args=cxx,
            language="c++",
        )
    ]
)
