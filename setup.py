import numpy as np
import os.path
import codecs
from setuptools import find_packages, setup
from distutils.extension import Extension
from Cython.Build import cythonize

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# ### Allows installation if cython is not yet installed
# try:
#     from Cython.Build import cythonize
# except ImportError:
#     # create closure for deferred import
#     def cythonize (*args, ** kwargs ):
#         from Cython.Build import cythonize
#         return cythonize(*args, ** kwargs)

ext_modules = [
	Extension(
		"pyblip.cython_utils._truncnorm",
		sources=["pyblip/cython_utils/_truncnorm.pyx"],
	),
	Extension(
		"pyblip.cython_utils._update_hparams",
		sources=["pyblip/cython_utils/_update_hparams.pyx"],
	),
	Extension(
		"pyblip.linear._linear",
		sources=["pyblip/linear/_linear.pyx"],
	),
	Extension(
		"pyblip.linear._linear_multi",
		sources=["pyblip/linear/_linear_multi.pyx"],
	),
	Extension(
		"pyblip.nprior._nprior",
		sources=["pyblip/nprior/_nprior.pyx"],
		# extra_compile_args = ["-ffast-math"]
	),
]

setup(
	name="pyblip",
	version=get_version('pyblip/__init__.py'),
	packages=find_packages(),
	description='Bayesian Linear Programming (BLiP) in Python',
	long_description=long_description,
	long_description_content_type="text/markdown",
	author='Asher Spector',
	author_email='amspector100@gmail.com',
	url='https://github.com/amspector100/pyblip',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	#keywords=['bayesian', 'detection', 'signals'],
	ext_modules=cythonize(
		ext_modules,
		compiler_directives={
			"language_level": 3, 
			"embedsignature": True
		},
		annotate=False,
	),
	include_dirs=[np.get_include()],
	python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.6.1",
        "cvxpy>=1.1.17", 
        "networkx>=2.4",
        "cython>=0.29.21",
        "cvxopt>=1.3.0",
    ],
    setup_requires=[
        'numpy>=1.19',
    	'setuptools>=58.0',
    	'cython>=0.29.21',
    ]
)