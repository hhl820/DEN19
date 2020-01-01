from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import glob

include_dirs = os.path.dirname(os.path.abspath(__file__))

source_NGENO = glob.glob(os.path.join(include_dirs, '*.cpp'))

setup(
	name='lhhngran',
	version="0.1",
	ext_modules=[CppExtension(
		name='lhhngran',
		sources=source_NGENO, include_dirs=[include_dirs]
		)],
	cmdclass={'build_ext': BuildExtension})
