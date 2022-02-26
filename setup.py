from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
	Extension(
		"pyblip.cython_utils._truncnorm",
		sources=["pyblip/cython_utils/_truncnorm.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	),
	Extension(
		"pyblip.cython_utils._update_hparams",
		sources=["pyblip/cython_utils/_update_hparams.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	),
	Extension(
		"pyblip.linear._linear",
		sources=["pyblip/linear/_linear.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	),
	Extension(
		"pyblip.linear._linear_multi",
		sources=["pyblip/linear/_linear_multi.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	),
	Extension(
		"pyblip.nprior._nprior",
		sources=["pyblip/nprior/_nprior.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	),
]

setup(
	name="pyblip",
	ext_modules=cythonize(
		ext_modules,
		compiler_directives = {"language_level": 3, "embedsignature": True}
	)
)