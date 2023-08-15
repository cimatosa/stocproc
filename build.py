from Cython.Build import cythonize
from distutils.command.build_ext import build_ext
from setuptools import setup
import numpy
import os


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """

    extensions = ["./stocproc/stocproc_c.pyx"]

    # gcc arguments hack: enable optimizations
    os.environ["CFLAGS"] = f"-O3 -I{numpy.get_include()}"

    # Build
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                extensions,
                language_level=3,
                compiler_directives={"linetrace": True},
            ),
            "cmdclass": {"build_ext": build_ext},
        }
    )


# if __name__ == "__main__":
#     setup_kwargs = {}
#     build(setup_kwargs)
#     print(setup_kwargs)
#     setup(**setup_kwargs)
