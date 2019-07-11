cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)

project(induc_gen)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_subdirectory(lib/pybind11)


set(RDK_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Disable Building Rdkit Python Wrappers")
set(RDK_INSTALL_INTREE OFF CACHE BOOL "Disable Rdkit int-tree install")
set(RDK_BUILD_CPP_TESTS OFF CACHE BOOL "Disable build rdkit tests")
set(RDK_BUILD_DESCRIPTORS3D OFF CACHE BOOL "Disable descriptors3d for rdkit")
set(RDK_BUILD_COORDGEN_SUPPORT OFF CACHE BOOL "Disable build coordgen for rdkit")
set(RDK_BUILD_MOLINTERCHANGE_SUPPORT OFF CACHE BOOL "build in support for CommonChem molecule interchange")

add_subdirectory(lib/rdkit EXCLUDE_FROM_ALL)

set(SOURCES_WRAP cpp/wrap/atom.cpp cpp/wrap/bond.cpp cpp/wrap/mol.cpp cpp/wrap/molops.cpp cpp/wrap/utilities.cpp)
set(SOURCES cpp/module.cpp cpp/molecule_representation.cpp cpp/molecule_edit.cpp ${SOURCES_WRAP})
pybind11_add_module(induc_gen_extensions MODULE ${SOURCES})

set(rdkit_libraries RDGeometryLib GraphMol SmilesParse ChemTransforms)

if(NOT MSVC)
    # On non-msvc platforms, static libraries have a _static suffix.
    list(TRANSFORM rdkit_libraries APPEND _static)
endif()

target_link_libraries(induc_gen_extensions PRIVATE rdkit_base ${rdkit_libraries})

