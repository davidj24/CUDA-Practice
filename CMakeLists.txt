# Specify the minimum version of CMake required.
# Newer versions like 3.20+ offer more features for CUDA integration.
cmake_minimum_required(VERSION 3.20)

# Define your project.
# LANGUAGES CXX, C, and CUDA tell CMake to enable support for these languages.
project(CUDA_Practice LANGUAGES CXX CUDA) # C is often implied by CXX or not needed explicitly

# Set C++ standard. Modern C++ features are beneficial.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer standard compliance

# Set CUDA C++ standard. CUDA 11+ often supports C++17 features.
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF) # Prefer standard compliance

# Find the CUDA toolkit. This sets up variables like CUDA_TOOLKIT_ROOT_DIR
# and targets like CUDA::cudart.
find_package(CUDAToolkit REQUIRED)

# Define your executable target.
# It's generally cleaner to list source files directly here.
# Assuming your main CUDA code is in 'main.cu' and potentially other C++ files.
add_executable(${CMAKE_PROJECT_NAME}
    main.cu
    # If you have other C++ files, list them here:
    # src/some_other_file.cpp
)

# Link the CUDA runtime library. This is crucial for using CUDA functions.
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE CUDA::cudart)

# --- Optional: Add compile options for debugging and optimization ---

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4> # C++ warnings for MSVC (equivalent to -Wall/-Wextra on GCC/Clang)

    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75> # Your GPU architecture (e.g., sm_75)
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets> # Suppress warnings about old GPU target
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-g> 
    $<$<COMPILE_LANGUAGE:CXX>:-g> 
)




add_executable(vecadd
    vecadd.cu
    
)

# Link the CUDA runtime library. This is crucial for using CUDA functions.
target_link_libraries(vecadd PRIVATE CUDA::cudart)

# --- Optional: Add compile options for debugging and optimization ---

target_compile_options(vecadd PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4> # C++ warnings for MSVC (equivalent to -Wall/-Wextra on GCC/Clang)

    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_75> # Your GPU architecture (e.g., sm_75)
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets> # Suppress warnings about old GPU target
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-g> 
    $<$<COMPILE_LANGUAGE:CXX>:-g> 
)