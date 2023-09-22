"""
This script provides triggers Cython build using distutils.

Default invocation, i.e., `python3 build_ext.py` does dome distutils magic to
build the extension inplace which corresponds to former call of `setup.py build_ext --inplace`.

original source: https://github.com/richard-hartmann/unified-cythonizing/edit/main/build_ext.py

Copyright 2023 Richard Hartmann
BSD (3 clause) License
"""

from Cython.Build import cythonize
import argparse
from distutils.command.build_ext import build_ext
from distutils.extension import Extension
from distutils.dist import Distribution
from distutils.errors import *
import os
import pathlib
import shutil

# if your cython code links against numpy, uncomment
import numpy


# assume that the build_ext.py script is in the root directory of the package
root_path = pathlib.Path(__file__).absolute().parent


##############################################################
#                                                            #
# SIMPLY REGISTER YOUR EXTENSIONS HERE                       #

sp_c = Extension(
    "stocproc.stocproc_c",
    sources=["./stocproc/stocproc_c.pyx"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=['-O3'],
)
list_of_ext = [sp_c]

# no further adjustments needed                              #
#                                                            #
##############################################################


class cd:
    """
    Context manager for changing the current working directory

    taken from https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory/13197763#13197763
    """

    def __init__(self, new_path):
        self.new_path = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


class cd_dec:
    """
    a decorator which changes to current path (see 'cd' class) before calling a function
    """

    def __init__(self, new_path):
        self.new_path = new_path

    def __call__(self, fnc):
        self.fnc = fnc
        return self._cd_dec_fnc

    def _cd_dec_fnc(self, *args, **kwargs):
        with cd(self.new_path):
            self.fnc(*args, **kwargs)


# the following is adapted from https://stackoverflow.com/a/60163996
class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        try:
            with cd(root_path):
                build_ext.run(self)
        except (DistutilsPlatformError, FileNotFoundError):
            raise BuildFailed("File not found. Could not compile C extension.")

    def build_extension(self, ext):
        try:
            with cd(root_path):
                build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError, ValueError):
            raise BuildFailed("Could not compile C extension.")


@cd_dec(root_path)
def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    # NOTE that with cythonize, ext_modules must not be a list!
    # so for more than one Cython extension things will need to be adapted
    setup_kwargs.update(
        {
            "ext_modules": cythonize(list_of_ext),
            "cmdclass": {"build_ext": ExtBuilder},
        }
    )


def cmd_build_ext():
    # see https://stackoverflow.com/a/60525118
    # distutils magic. This is essentially the same as calling
    # python setup.py build_ext --inplace
    setup_kwargs = {}
    build(setup_kwargs)
    dist = Distribution(attrs=setup_kwargs)

    build_ext_cmd = dist.get_command_obj("build_ext")
    build_ext_cmd.ensure_finalized()
    build_ext_cmd.inplace = 1
    build_ext_cmd.run()


@cd_dec(root_path)
def cmd_clean(yes=False):
    dirs_to_remove = []

    d = pathlib.Path("./build")
    if d.exists():
        dirs_to_remove.append(d.absolute())
    d = pathlib.Path("./dist")
    if d.exists():
        dirs_to_remove.append(d.absolute())

    files_to_remove = []

    for e in list_of_ext:
        # remove c binaries
        full_mod_path = pathlib.Path(*e.name.split("."))
        mod_path = full_mod_path.parent
        mod_name = full_mod_path.stem
        if mod_path.exists():
            for f in mod_path.iterdir():
                f = f.name
                if f.startswith(mod_name) and f.endswith(".so"):
                    files_to_remove.append((mod_path / f).absolute())

        # remove c/cpp source files
        for s in e.sources:
            s = pathlib.Path(s)
            s_suff = s.suffix
            s_stem = s.stem
            s_parent = s.parent
            if s_suff == ".pyx":
                f = s_parent / (str(s_stem) + ".c")
                if f.exists():
                    files_to_remove.append(f.absolute())

                f = s_parent / (str(s_stem) + ".cpp")
                if f.exists():
                    files_to_remove.append(f.absolute())

    if len(dirs_to_remove) > 0:
        print("remove the following directories")
        for d in dirs_to_remove:
            print("  *", d)

    if len(files_to_remove) > 0:
        print("(and) remove the following directories files")
        for f in files_to_remove:
            print("  *", f)

    if len(dirs_to_remove) > 0 or len(files_to_remove) > 0:
        if not yes:
            answ = input("y/n [n]:")
        else:
            answ = "y"

        if answ == "y":
            for d in dirs_to_remove:
                shutil.rmtree(d)
            for f in files_to_remove:
                os.remove(f)
            print("files removed")
        else:
            print("abort!")
    else:
        print("noting to clean")


# note that upon poetry install / build, poetry actually triggers
# Command '['... .venv/bin/python', 'build_ext.py']'
# so the follows works such that this call will build the Cython extension
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        help="what to do, choose between 'build_ext' or 'clear'\n"
        + "  build_ext: triggers Cython inplace build (using distutils magic)\n"
        + "  clean: remove dirs 'build' and 'dist' as well as libraries and c/cpp files generated by cython",
        default="build_ext",
        nargs="?",
        type=str,
    )

    parser.add_argument(
        "--yes", help="assume 'yes' when running 'clean'", action="store_true"
    )

    args = parser.parse_args()

    if args.command == "build_ext":
        cmd_build_ext()
    elif args.command == "clean":
        cmd_clean(yes=args.yes)
    else:
        parser.print_help()
        raise ValueError(f"unknown command '{args.command}'")

