# ---------------------------------------------------------------------------
#  cuda_engine - native CUDA compute/raster addon for the GeoSwarm demo
#
#  Built by cmake-js against Electron's Node ABI (never the system Node).
#  Output lands at native/build/Release/cuda_engine.node.
#
#  IMPORTANT ORDERING NOTE
#  -----------------------
#  Everything the CUDA language enablement depends on has to be in the cache
#  BEFORE project(). The Visual Studio generator picks its CUDA integration
#  from CMAKE_GENERATOR_TOOLSET, while a Ninja build would pick it from
#  CMAKE_CUDA_COMPILER - we set both so neither path can silently fall back
#  to whatever nvcc happens to be first on PATH (which on this class of
#  machine is often an ancient toolkit that cannot emit sm_120).
#
#  If you change any of the four cache variables below, delete the build
#  directory first (npm run clean:native). A stale CMakeCache.txt pins the
#  previously detected compiler and produces errors that make no sense.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# --- CUDA toolchain pin (must precede project()) ---------------------------
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc.exe" CACHE FILEPATH "")
set(CMAKE_GENERATOR_TOOLSET "cuda=12.9" CACHE STRING "")  # VS generator selects CUDA via toolset, not CMAKE_CUDA_COMPILER

# Real + virtual arch for Blackwell consumer parts. The plain "120" entry emits
# SASS for sm_120; "120-virtual" keeps PTX in the fatbin so newer drivers can
# JIT the module on architectures that do not exist yet.
set(CMAKE_CUDA_ARCHITECTURES "120;120-virtual" CACHE STRING "")

project(cuda_engine LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# Default to Release when a single-config generator is used; the VS generator
# ignores this and takes --config from cmake-js (Release).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
#  node-addon-api headers
#
#  cmake-js hands us CMAKE_JS_INC (node + uv + v8 headers) but not the C++
#  wrapper. Ask node for the installed package's include directory rather than
#  hardcoding a node_modules layout that npm is free to change.
# ---------------------------------------------------------------------------
execute_process(COMMAND node -p "require('node-addon-api').include_dir"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT NODE_ADDON_API_DIR)
  message(FATAL_ERROR "node-addon-api include_dir came back empty - run npm install at the repo root.")
endif()

# node -p prints a Windows path with backslashes; CMake wants forward slashes
# or it treats the separators as escape sequences further down.
string(REPLACE "\\" "/" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")
if(NOT IS_ABSOLUTE "${NODE_ADDON_API_DIR}")
  set(NODE_ADDON_API_DIR "${CMAKE_SOURCE_DIR}/../${NODE_ADDON_API_DIR}")
endif()
message(STATUS "node-addon-api include dir: ${NODE_ADDON_API_DIR}")

# ---------------------------------------------------------------------------
#  Sources
# ---------------------------------------------------------------------------
set(SOURCES
    src/addon.cc
    src/engine.cc
    src/native_view.cc
    src/kernels/swarm.cu
    src/kernels/weather.cu
    src/kernels/storm.cu
    src/kernels/raster.cu
)

add_library(cuda_engine SHARED ${SOURCES} ${CMAKE_JS_SRC})

set_target_properties(cuda_engine PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    # Device code lives in four translation units and calls shared __device__
    # helpers, so we need relocatable device code + a device link step.
    CUDA_SEPARABLE_COMPILATION ON
    # Static cudart keeps cudart64_*.dll out of the shipping tree - the .node
    # ends up with zero CUDA DLL dependencies. Also why there is no curand or
    # cufft anywhere: the kernels use hash RNG and analytic noise instead.
    CUDA_RUNTIME_LIBRARY Static
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_features(cuda_engine PRIVATE cxx_std_17)

target_include_directories(cuda_engine PRIVATE
    ${CMAKE_JS_INC}
    ${NODE_ADDON_API_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(cuda_engine PRIVATE
    NAPI_VERSION=8               # Electron 43 / Node 24 ABI floor we target
    NAPI_DISABLE_CPP_EXCEPTIONS  # no C++ exceptions across the N-API boundary
    NOMINMAX                     # windows.h min/max macros wreck std::min/max
    WIN32_LEAN_AND_MEAN
    UNICODE
    _UNICODE
)

# -use_fast_math trades a few ULP for much cheaper transcendentals, which is
# the right call for a visual demo dominated by sin/cos/rsqrt.
# --extended-lambda lets us pass __device__ lambdas into helper templates.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math --extended-lambda")

target_link_libraries(cuda_engine PRIVATE
    ${CMAKE_JS_LIB}
    CUDA::cudart_static
    d3d11
    dxgi
)

# ---------------------------------------------------------------------------
#  Generate node.lib on MSVC.
#
#  Mandatory for cmake-js on Windows: the delay-load stub library the addon
#  links against is produced from a .def file at configure time. Without this
#  block every N-API symbol comes back unresolved at link.
# ---------------------------------------------------------------------------
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
  execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
