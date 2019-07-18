import platform
import os
import subprocess
import sys

from setuptools import setup, find_packages, Extension

from torch.utils.cpp_extension import BuildExtension

dependency_directory = "lib"


def download_and_patch_rdkit():
    rdkit_dir = os.path.join(dependency_directory, 'rdkit')

    if not os.path.isdir(rdkit_dir):
        print("Downloading rdkit source from github.")
        subprocess.call(["git", "clone", "--branch", "Release_2019_03_1",
                         "https://github.com/rdkit/rdkit.git", rdkit_dir])

        print("Patching rdkit source")
        subprocess.check_call(["git", "apply", "../rdkit.patch"], cwd=rdkit_dir)


class CMakeExtension(Extension):
    def __init__(self, name, get_source=None, source_dir=''):
        super(CMakeExtension, self).__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)
        self.get_source = get_source


class CMakeBuild(BuildExtension):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build genric.")
        super(CMakeBuild, self).run()

    def build_extension(self, ext):
        if not isinstance(ext, CMakeExtension):
            if self.compiler.compiler_type == 'msvc':
                pass

            return super(CMakeBuild, self).build_extension(ext)

        if ext.get_source:
            ext.get_source()

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ["-G", "Ninja",
                      "-DPYTHON_EXECUTABLE=" + sys.executable,
                      "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={0}'.format(extdir)]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO={}'.format(extdir)]

        build_dir = os.path.join(self.build_temp, ext.name)
        print('build in {0}'.format(build_dir))
        os.makedirs(build_dir, exist_ok=True)

        if not os.path.exists(os.path.join(build_dir, 'CMakeFiles')):
            subprocess.check_call(["cmake", ext.source_dir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)


extensions = [
    CMakeExtension("genric.genric_extensions", download_and_patch_rdkit),
    CMakeExtension("genric.torch_extensions", source_dir="cpp/torch/"),
]

setup(
    name="induc-gen",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass=dict(build_ext=CMakeBuild),
)

rdkit_build_options = [
    ("RDK_BUILD_PYTHON_WRAPPERS", "OFF"),
    ("RDK_BUILD_CPP_TESTS", "OFF"),
    ("RDK_INSTALL_INTREE", "OFF"),
    ("RDK_BUILD_DESCRIPTORS3D", "OFF"),
    ("RDK_BUILD_COORDGEN_SUPPORT", "OFF"),
    ("RDK_BUILD_MOLINTERCHANGE_SUPPORT", "OFF")]
