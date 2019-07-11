cmake_minimum_required(VERSION 3.12)
project(induc_gen_torch LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
cmake_policy(SET CMP0069 NEW)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)

IF(NOT DEFINED TORCH_INCLUDE_DIRS)
    execute_process(COMMAND python -c "from torch.utils.cpp_extension import include_paths; print('{{{' + ';'.join(include_paths()) + '}}}')"
                    OUTPUT_VARIABLE TORCH_INCLUDE_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "{{{(.*)}}}" TORCH_INCLUDE_OUTPUT "${TORCH_INCLUDE_OUTPUT}")
    string(REGEX REPLACE "[{}]" "" TORCH_INCLUDE_OUTPUT "${TORCH_INCLUDE_OUTPUT}")
    set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_OUTPUT} CACHE STRING "Include directories for compiling torch extensions.")
ENDIF()

IF(NOT DEFINED TORCH_LIBRARY_DIRS)
    execute_process(COMMAND python -c "from torch.utils.cpp_extension import library_paths; print('{{{' + ';'.join(library_paths()) + '}}}')"
                    OUTPUT_VARIABLE TORCH_LIBRARY_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "{{{(.*)}}}" TORCH_LIBRARY_OUTPUT ${TORCH_LIBRARY_OUTPUT})
    string(REGEX REPLACE "[{}]" "" TORCH_LIBRARY_OUTPUT ${TORCH_LIBRARY_OUTPUT})
    set(TORCH_LIBRARY_DIRS ${TORCH_LIBRARY_OUTPUT} CACHE STRING "Library directories for compiling torch extensions.")
ENDIF()

if(WIN32)
set(TORCH_LIBRARIES c10 c10_cuda caffe2 torch torch_python _C caffe2_gpu)
endif()

link_directories(${TORCH_LIBRARY_DIRS})

add_library(pytorch_imported INTERFACE)
target_compile_definitions(pytorch_imported INTERFACE -D_GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(pytorch_imported INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(pytorch_imported INTERFACE ${TORCH_LIBRARIES})

add_library(cub INTERFACE)
target_include_directories(cub INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/cub-1.8.0/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/lib/pybind11)

add_library(torch_extensions_kernels STATIC
    segment_logsumexp.cpp segment_logsumexp_cuda.cu segment_logsumexp_backward.cu
    segment_pool.cpp segment_pool_cuda.cu
    repeat_interleave.cpp repeat_interleave_cuda.cu)
target_include_directories(torch_extensions_kernels PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set_target_properties(torch_extensions_kernels PROPERTIES CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden")
set_target_properties(torch_extensions_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(torch_extensions_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(torch_extensions_kernels PUBLIC -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=torch_extensions)
target_link_libraries(torch_extensions_kernels PUBLIC pytorch_imported cub)

pybind11_add_module(torch_extensions MODULE NO_EXTRAS module.cpp)
target_link_libraries(torch_extensions PRIVATE pytorch_imported torch_extensions_kernels)

if(IPO_SUPPORTED)
set_target_properties(torch_extensions PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(MSVC)
    set(disabled_warnings /wd4244 /wd4251 /wd4275 /wd4267 /wd4522 /wd4273 /wd4305)
    list(TRANSFORM disabled_warnings_cuda PREPEND -Xcompiler)
    target_compile_definitions(torch_extensions_kernels PRIVATE _CRT_SECURE_NO_DEPRECATE)
    target_compile_definitions(torch_extensions PRIVATE _CRT_SECURE_NO_DEPRECATE)
    target_compile_options(torch_extensions PRIVATE ${disabled_warnings})
    target_compile_options(torch_extensions_kernels PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${disabled_warnings}>)
    target_compile_options(torch_extensions_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "/wd4244 /wd4251 /wd4275 /wd4267 /wd4522 /wd4273 /wd4305">)
endif()
