from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
	# Extension(
	# 	"pyblip.nprior.nprior_c",
	# 	sources=["pyblip/nprior/nprior_c.pyx"],
	# 	libraries=["m"],  # Unix-like specific
	# 	extra_compile_args = ["-ffast-math"]
	# ),
	# Extension(
	# 	"pyblip.probit._probit",
	# 	sources=["pyblip/probit/_probit.pyx"],
	# 	libraries=["m"],  # Unix-like specific
	# 	extra_compile_args = ["-ffast-math"]
	# ),
	Extension(
		"pyblip.linear._linear",
		sources=["pyblip/linear/_linear.pyx"],
		libraries=["m"],  # Unix-like specific
		extra_compile_args = ["-ffast-math"]
	)
]

setup(
	name="pyblip",
	ext_modules=cythonize(
		ext_modules,
		compiler_directives = {"language_level": 3, "embedsignature": True}
	)
)