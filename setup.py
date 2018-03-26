from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from stocproc import version_full

author      = u"Richard Hartmann"
authors     = [author]
description = 'Generate continuous time stationary stochastic processes from a given auto correlation function.'
name        = 'stocproc'
version = version_full()

if __name__ == "__main__":
    setup(
        name=name,
        author=author,
        author_email='richard.hartmann@tu-dresden.de',
        url='https://github.com/cimatosa/stocproc',
        version=version,
        packages=[name],
        package_dir={name: name},
        license="BSD (3 clause)",
        description=description,
        long_description=description,
        keywords=["stationary", "stochastic", "process", "random"],
        classifiers= [
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'License :: OSI Approved :: BSD License',
            'Topic :: Utilities',
            'Intended Audience :: Researcher'],
        platforms=['ALL'],
        install_requires=['fcSpline>=0.1'],
        dependency_links=['https://raw.githubusercontent.com/cimatosa/fcSpline/master/egg/fcSpline-0.1-py3.4-linux-x86_64.egg'],
        ext_modules = cythonize(["stocproc/stocproc_c.pyx"]),
        include_dirs = [np.get_include()],
    )