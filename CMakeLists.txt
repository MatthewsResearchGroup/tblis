# Keep empty arguments instead of undefining the variable
set(CMAKE_POLICY_DEFAULT_CMP0174 NEW)

cmake_minimum_required(VERSION 3.23)

file(STRINGS tblis-version TBLIS_VERSION_RAW)
set(TBLIS_VERSION "${TBLIS_VERSION_RAW}")
string(REPLACE "." ";" TBLIS_VERSION_RAW ${TBLIS_VERSION_RAW})
list(GET TBLIS_VERSION_RAW 0 TBLIS_VERSION_MAJOR)
list(GET TBLIS_VERSION_RAW 1 TBLIS_VERSION_MINOR)

project(TBLIS
    VERSION "${TBLIS_VERSION}"
    LANGUAGES C CXX
    HOMEPAGE_URL "http://www.github.com/MatthewsResearchGroup/tblis"
    DESCRIPTION "The Tensor-Based Library Instantiation Software"
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CheckCompilerFlag)
include(ConfigureWrapper)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CMakePackageConfigHelpers)
include(CheckCCompilerFlag)
include(FetchContent)
include(ExternalProject)
include(CheckLibraryExists)

set(TBLIS_PREFIX ${INSTALL_PREFIX})

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/marray/CMakeLists.txt)
    message(FATAL_ERROR "MArray not found -- have you checkout out the git submodules?")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/tci/CMakeLists.txt)
    message(FATAL_ERROR "TCI not found -- have you checkout out the git submodules?")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/stl_ext/stl_ext/complex.hpp)
    message(FATAL_ERROR "stl_ext not found -- have you checkout out the git submodules?")
endif()

###############################################################################
#
# Set up options
#
###############################################################################

option(ENABLE_TESTS "Build tests" OFF)

option(ENABLE_SHARED "Build a shared library" ON)

option(ENABLE_STATIC "Build a static library" ON)

option(ENABLE_COMPAT "Enable compatibility with the TBLIS 1.x interface" OFF)
set(TBLIS_ENABLE_COMPAT ${ENABLE_COMPAT})

option(ENABLE_MEMKIND "Enable use of the memkind library for MCDRAM allocation if supported." OFF)

option(ENABLE_HWLOC "Enable use of the hwloc library for hardware locality awareness." ON)

set(BLIS_CONFIG_FAMILY auto CACHE STRING "BLIS sub-configuration to use.")

set(LENGTH_TYPE ptrdiff_t CACHE STRING "The type to use for lengths of tensor dimensions. Must be a signed ISO C integer type.")

set(STRIDE_TYPE ptrdiff_t CACHE STRING "The type to use for strides of tensor dimensions. Must be a signed ISO C integer type.")

set(LABEL_TYPE char CACHE STRING "The type to use to label tensor dimensions. Must be a standard ISO C integer type.")

# Check for valid types
set(SIGNED_INT_TYPES
    "signed char"
    "signed short"
    "short"
    "signed short int"
    "short int"
    "signed int"
    "signed"
    "int"
    "signed long"
    "long"
    "signed long int"
    "long int"
    "signed long long"
    "long long"
    "signed long long int"
    "long long int"
    "int8_t"
    "int16_t"
    "int32_t"
    "int64_t"
    "int_least8_t"
    "int_least16_t"
    "int_least32_t"
    "int_least64_t"
    "int_fast8_t"
    "int_fast16_t"
    "int_fast32_t"
    "int_fast64_t"
    "intmax_t"
    "intptr_t"
    "ptrdiff_t"
)

set(INT_TYPES
    ${SIGNED_INT_TYPES}
    "unsigned char"
    "char"
    "unsigned short"
    "unsigned short int"
    "unsigned int"
    "unsigned"
    "unsigned long"
    "unsigned long int"
    "unsigned long long"
    "unsigned long long int"
    "uint8_t"
    "uint16_t"
    "uint32_t"
    "uint64_t"
    "uint_least8_t"
    "uint_least16_t"
    "uint_least32_t"
    "uint_least64_t"
    "uint_fast8_t"
    "uint_fast16_t"
    "uint_fast32_t"
    "uint_fast64_t"
    "uintmax_t"
    "uintptr_t"
    "size_t"
)

if(NOT LENGTH_TYPE IN_LIST SIGNED_INT_TYPES)
    message(FATAL_ERROR "Length type ${LENGTH_TYPE} is not a signed ISO C integer type.")
endif()

if(NOT STRIDE_TYPE IN_LIST SIGNED_INT_TYPES)
    message(FATAL_ERROR "Stride type ${STRIDE_TYPE} is not a signed ISO C integer type.")
endif()

if(NOT LABEL_TYPE IN_LIST INT_TYPES)
    message(FATAL_ERROR "Label type ${LABEL_TYPE} is not an ISO C integer type.")
endif()

# Apple's sys/cdefs.h redefines __restrict to blank in C++ mode (!!!!!!)
if(APPLE)
    set(RESTRICT "__restrict__")
else()
    set(RESTRICT "__restrict")
endif()

# Check for bit operations
check_cxx_source_compiles("
    int main()
    {
        // Test bitset builtin functions
        auto x = 42u;
        auto xl = 42ul;
        auto xll = 42ull;
        __builtin_popcount(x);
        __builtin_popcountl(xl);
        __builtin_popcountll(xll);
        __builtin_clz(x);
        __builtin_clzl(xl);
        __builtin_clzll(xll);
        __builtin_ctz(x);
        __builtin_ctzl(xl);
        __builtin_ctzll(xll);
        return 0;
    }"
    TBLIS_HAVE_GCC_BITSET_BUILTINS
)
set(CMAKE_REQUIRED_FLAGS -std=c++20)
check_cxx_source_compiles("
    #include <bit>
    int main()
    {
        // Test C++20 bitset functions
        auto x = 42u;
        auto xl = 42ul;
        auto xll = 42ull;
        std::popcount(x);
        std::popcount(xl);
        std::popcount(xll);
        std::countl_zero(x);
        std::countl_zero(xl);
        std::countl_zero(xll);
        std::countr_zero(x);
        std::countr_zero(xl);
        std::countr_zero(xll);
        return 0;
    }"
    TBLIS_HAVE_CXX20_BITSET
)
unset(CMAKE_REQUIRED_FLAGS)

# Check if we need -lm
check_symbol_exists(sqrt math.h HAVE_SQRT)
find_library(HAVE_LIBM m)
if(NOT HAVE_SQRT)
    if (NOT HAVE_LIBM)
        message(FATAL_ERROR "no math library found.")
    else()
        list(APPEND PRIVATE_LIBS ${HAVE_LIBM})
    endif()
endif()

# Check if we need -lrt
check_symbol_exists(clock_gettime time.h HAVE_CLOCK_GETTIME)
find_library(HAVE_LIBRT rt)
if(NOT HAVE_CLOCK_GETTIME)
    if (NOT HAVE_LIBRT)
        message(FATAL_ERROR "clock_gettime not found.")
    else()
        list(APPEND PRIVATE_LIBS ${HAVE_LIBRT})
    endif()
endif()

# Check for openmp simd flag
set(OMP_SIMD_FLAG "")
check_compiler_flag(CXX -fopenmp-simd HAVE_FOPENMP_SIMD)
if(HAVE_FOPENMP_SIMD)
    set(OMP_SIMD_FLAG -fopenmp-simd)
endif()

check_compiler_flag(CXX -qopenmp-simd HAVE_QOPENMP_SIMD)
if(HAVE_QOPENMP_SIMD AND NOT HAVE_FOPENMP_SIMD)
    set(OMP_SIMD_FLAG -qopenmp-simd)
endif()

# Check for memkind
if(ENABLE_MEMKIND)
    check_include_file(hbwmalloc.h TBLIS_HAVE_HBWMALLOC_H)
    check_library_exists(memkind hbw_malloc "" TBLIS_HAVE_LIBMEMKIND)
    if(TBLIS_HAVE_HBWMALLOC_H AND TBLIS_HAVE_LIBMEMKIND)
        list(APPEND PRIVATE_LIBS -lmemkind)
    else()
        set(TBLIS_HAVE_HBWMALLOC_H OFF)
    endif()
endif()

# Check for hwloc
if(ENABLE_HWLOC)
    check_include_file(hwloc.h TBLIS_HAVE_HWLOC_H)
    check_library_exists(hwloc hwloc_topology_init "" TBLIS_HAVE_LIBHWLOC)
    if(TBLIS_HAVE_HWLOC_H AND TBLIS_HAVE_LIBHWLOC)
        list(APPEND PRIVATE_LIBS -lhwloc)
    else()
        set(TBLIS_HAVE_HWLOC_H OFF)
    endif()
endif()

# Check for lscpu
find_program(TBLIS_HAVE_LSCPU lscpu)

# Check for sysctl
check_symbol_exists(sysctl sys/sysctl.h TBLIS_HAVE_SYSCTL)
check_symbol_exists(sysctlbyname sys/sysctl.h TBLIS_HAVE_SYSCTLBYNAME)

# Check for sysconf
check_symbol_exists(sysconf unistd.h TBLIS_HAVE_SYSCONF)
check_symbol_exists(_SC_NPROCESSORS_CONF unistd.h TBLIS_HAVE__SC_NPROCESSORS_CONF)
check_symbol_exists(_SC_NPROCESSORS_ONLN unistd.h TBLIS_HAVE__SC_NPROCESSORS_ONLN)

# Find make
find_program(MAKE_EXECUTABLE NAMES make gmake mingw32-make REQUIRED)

set(ENABLE_SHARED_OLD ${ENABLE_SHARED})
set(ENABLE_STATIC_OLD ${ENABLE_STATIC})
set(ENABLE_SHARED OFF)
set(ENABLE_STATIC ON)
add_subdirectory(tblis/external/tci)
add_subdirectory(tblis/external/marray)
set(ENABLE_SHARED ${ENABLE_SHARED_OLD})
set(ENABLE_STATIC ${ENABLE_STATIC_OLD})

# Check for known bug in nvc 25.7
if(CMAKE_CXX_COMPILER_ID STREQUAL NVHPC AND
   CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 25.7.0)
    set(TBLIS_LARGE_ATOMIC_WORKAROUND ON)
endif()

###############################################################################
#
# Find and/or set up BLIS
#
###############################################################################

file(STRINGS blis-git-tag BLIS_GIT_TAG)
set(BLIS_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/blis")

message(CHECK_START "Looking for BLIS")

find_package(BLIS QUIET)

if(BLIS_FOUND)
    set(BLIS_TARGET BLIS::BLIS)
    message(CHECK_PASS "found via cmake")
    #not sure what to do here
    #set(BLIS_BINARY_DIR ...)
    add_custom_target(blis-install)
else()
    find_package(PkgConfig)
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
    pkg_check_modules(BLIS IMPORTED_TARGET blis>=2.0)
endif()

if(BLIS_FOUND)
    set(BLIS_TARGET PkgConfig::BLIS)
    message(CHECK_PASS "found via pkgconfig")
    set(BLIS_BINARY_DIR "${BLIS_PREFIX}/share/blis")
    set(BLIS_INSTALL_DIR ${BLIS_PREFIX})
    add_custom_target(blis-install)
    target_include_directories(${BLIS_TARGET} INTERFACE
        $<BUILD_INTERFACE:${BLIS_INSTALL_DIR}/include>
    )
else()
    FetchContent_Declare(blis
        GIT_REPOSITORY https://github.com/flame/blis.git
        GIT_TAG ${BLIS_GIT_TAG}
    )

    FetchContent_MakeAvailable(blis)
    FetchContent_GetProperties(
      blis
      SOURCE_DIR BLIS_SOURCE_DIR
      BINARY_DIR BLIS_BINARY_DIR
      POPULATED BLIS_POPULATED
    )
    set(BLIS_INSTALL_DIR "${BLIS_BINARY_DIR}/usr")

    set(BLIS_DEBUG_FLAG "")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(BLIS_DEBUG_FLAG "--enable-debug=noopt")
    endif()

    message(CHECK_PASS "downloaded")

    if (NOT EXISTS ${BLIS_BINARY_DIR}/config.mk)
        # configure BLIS at CMake configure time
        message(STATUS "Configuring BLIS")
        execute_process(
            WORKING_DIRECTORY ${BLIS_BINARY_DIR}
            COMMAND ${BLIS_SOURCE_DIR}/configure
                --prefix=${BLIS_INSTALL_DIR}
                --disable-shared
                --enable-static
                ${BLIS_DEBUG_FLAG}
                --disable-blas
                --disable-cblas
                "--enable-threading=pthread"
                ${BLIS_CONFIG_FAMILY}
        )
        file(MAKE_DIRECTORY ${BLIS_INSTALL_DIR}/include/blis)
    endif()

    # Add + to the Make recipe line if CMake is new enough.
    # This enables parallel compilation of the plugin.
    # It does nothing if the generator isn't Make.
    # Tested with CMake<3.28 and >=3.28
    # See the CMake documentation for "add_custom_command"
    # (Since you need Make anyway, it's a good idea to use it as the generator.)
    set(TBLIS_JOBSERVER_AWARE_ARG "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0")
      set(TBLIS_JOBSERVER_AWARE_ARG JOB_SERVER_AWARE TRUE)
    endif()

    add_custom_command(
        COMMAND ${MAKE_EXECUTABLE} install
        OUTPUT
            ${BLIS_INSTALL_DIR}/lib/libblis.a
            ${BLIS_INSTALL_DIR}/include/blis/blis.h
        ${TBLIS_JOBSERVER_AWARE_ARG}
        COMMENT "Building BLIS"
        WORKING_DIRECTORY ${BLIS_BINARY_DIR}
    )
    add_custom_target(blis-install
        DEPENDS
            ${BLIS_INSTALL_DIR}/lib/libblis.a
            ${BLIS_INSTALL_DIR}/include/blis/blis.h
    )

    # extract the blis configuration name (at CMake configure time) into BLIS_CONFIG_FAMILY
    file(READ "${BLIS_BINARY_DIR}/config.mk" CONFIG_MK_CONTENTS)
    string(REGEX MATCH "CONFIG_NAME[^\n]*" CONFIG_LINE "${CONFIG_MK_CONTENTS}")
    string(REGEX MATCHALL "[^ \t]+" CONFIG_TOKENS "${CONFIG_LINE}")
    list(GET CONFIG_TOKENS 2 BLIS_CONFIG_FAMILY)
    message(STATUS "BLIS configuration: ${BLIS_CONFIG_FAMILY}")

    # make an interface library target for BLIS
    add_library(tblis-blis INTERFACE)
    target_link_libraries(tblis-blis INTERFACE
        $<BUILD_INTERFACE:${BLIS_INSTALL_DIR}/lib/libblis.a>
        $<INSTALL_INTERFACE:${INSTALL_LIBDIR}/tblis/libblis_core.a>
    )
    target_include_directories(tblis-blis INTERFACE
        $<BUILD_INTERFACE:${BLIS_INSTALL_DIR}/include>
    )
    add_dependencies(tblis-blis blis-install)
    set(BLIS_TARGET tblis-blis)
endif()

###############################################################################
#
# Set up BLIS plugin
#
###############################################################################

# extract the blis configuration name (at CMake configure time) into BLIS_CONFIG_FAMILY
file(READ "${BLIS_BINARY_DIR}/config.mk" CONFIG_MK_CONTENTS)
string(REGEX MATCH "CONFIG_NAME[^\n]*" CONFIG_LINE "${CONFIG_MK_CONTENTS}")
string(REGEX MATCHALL "[^ \t]+" CONFIG_TOKENS "${CONFIG_LINE}")
list(GET CONFIG_TOKENS 2 BLIS_CONFIG_FAMILY)
message(STATUS "BLIS configuration: ${BLIS_CONFIG_FAMILY}")

set(TBLIS_PLUGIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/tblis/plugin)
file(MAKE_DIRECTORY ${TBLIS_PLUGIN_DIR})

get_target_property(TCI_INCLUDES tci-static INTERFACE_INCLUDE_DIRECTORIES)
string(REGEX REPLACE ";+" ";" TCI_INCLUDES ${TCI_INCLUDES})
list(TRANSFORM TCI_INCLUDES PREPEND -I)
set(PLUGIN_FLAGS
    -I${BLIS_INSTALL_DIR}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/marray
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/tci
    -I${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/tci/include
    -I${CMAKE_CURRENT_BINARY_DIR}
	-I${CMAKE_CURRENT_BINARY_DIR}/tblis/external/tci
	-I${CMAKE_CURRENT_BINARY_DIR}/tblis/external/tci/include
    ${TCI_INCLUDES}
)

add_custom_command(
    OUTPUT ${TBLIS_PLUGIN_DIR}/config.mk
    COMMAND ${BLIS_INSTALL_DIR}/share/blis/configure-plugin
        CC=${CMAKE_C_COMPILER}
        CXX=${CMAKE_CXX_COMPILER}
        CFLAGS="${PLUGIN_FLAGS} ${OMP_SIMD_FLAG}"
        CXXFLAGS="${PLUGIN_FLAGS} ${OMP_SIMD_FLAG} -std=c++20"
        ${BLIS_DEBUG_FLAG}
        -f --build
        --path=${CMAKE_CURRENT_SOURCE_DIR}/tblis/plugin
        tblis
    DEPENDS blis-install
    COMMENT "Configuring BLIS plugin"
    WORKING_DIRECTORY ${TBLIS_PLUGIN_DIR}
    COMMAND_EXPAND_LISTS
)
add_custom_target(plugin-configure
    DEPENDS ${TBLIS_PLUGIN_DIR}/config.mk
)

add_custom_command(
    OUTPUT ${TBLIS_PLUGIN_DIR}/lib/${BLIS_CONFIG_FAMILY}/libblis_tblis.a
    COMMAND ${MAKE_EXECUTABLE}
    ${TBLIS_JOBSERVER_AWARE_ARG}
    DEPENDS plugin-configure
    COMMENT "Building BLIS plugin"
    WORKING_DIRECTORY ${TBLIS_PLUGIN_DIR}
)
add_custom_target(plugin-build
    DEPENDS ${TBLIS_PLUGIN_DIR}/lib/${BLIS_CONFIG_FAMILY}/libblis_tblis.a
)

add_library(tblis-plugin INTERFACE)
target_link_libraries(tblis-plugin INTERFACE
    $<BUILD_INTERFACE:${TBLIS_PLUGIN_DIR}/lib/${BLIS_CONFIG_FAMILY}/libblis_tblis.a>
    $<INSTALL_INTERFACE:${INSTALL_LIBDIR}/tblis/libblis_tblis.a>
)
add_dependencies(tblis-plugin plugin-build)

###############################################################################
#
# Configure TBLIS
#
###############################################################################

set(PKGCONFIG_LIBS "-L${libdir}/tblis -lblis_tblis -lblis_core")
set(PKGCONFIG_REQUIRES "tci >= 1.0")

configure_file(tblis/tblis_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/tblis/tblis_config.h @ONLY)
configure_file(tblis.pc.in ${CMAKE_CURRENT_BINARY_DIR}/tblis.pc @ONLY)
configure_file(TBLISConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/TBLISConfig.cmake @ONLY)

set(TBLIS_SOURCES
    tblis/frame/0/add.cxx
    tblis/frame/0/mult.cxx
    tblis/frame/0/reduce.cxx
    tblis/frame/1m/packm/packm_blk_bsmtc.cxx
    tblis/frame/1m/packm/packm_blk_dpd.cxx
    tblis/frame/1t/dense/add.cxx
    tblis/frame/1t/dense/dot.cxx
    tblis/frame/1t/dense/reduce.cxx
    tblis/frame/1t/dense/scale.cxx
    tblis/frame/1t/dense/set.cxx
    tblis/frame/1t/dense/shift.cxx
    tblis/frame/1t/dpd/add.cxx
    tblis/frame/1t/dpd/dot.cxx
    tblis/frame/1t/dpd/reduce.cxx
    tblis/frame/1t/dpd/scale.cxx
    tblis/frame/1t/dpd/set.cxx
    tblis/frame/1t/dpd/shift.cxx
    tblis/frame/1t/indexed/add.cxx
    tblis/frame/1t/indexed/dot.cxx
    tblis/frame/1t/indexed/reduce.cxx
    tblis/frame/1t/indexed/scale.cxx
    tblis/frame/1t/indexed/set.cxx
    tblis/frame/1t/indexed/shift.cxx
    tblis/frame/1t/indexed_dpd/add.cxx
    tblis/frame/1t/indexed_dpd/dot.cxx
    tblis/frame/1t/indexed_dpd/reduce.cxx
    tblis/frame/1t/indexed_dpd/scale.cxx
    tblis/frame/1t/indexed_dpd/set.cxx
    tblis/frame/1t/indexed_dpd/shift.cxx
    tblis/frame/1t/add.cxx
    tblis/frame/1t/dot.cxx
    tblis/frame/1t/reduce.cxx
    tblis/frame/1t/scale.cxx
    tblis/frame/1t/set.cxx
    tblis/frame/1t/shift.cxx
    tblis/frame/3m/gemm/gemm_ker_bsmtc.cxx
    tblis/frame/3m/gemm/gemm_ker_dpd.cxx
    tblis/frame/3t/dense/mult.cxx
    tblis/frame/3t/dpd/mult.cxx
    tblis/frame/3t/indexed/mult.cxx
    tblis/frame/3t/indexed_dpd/mult.cxx
    tblis/frame/3t/mult.cxx
    tblis/frame/base/basic_types.cxx
    tblis/frame/base/block_scatter.cxx
    tblis/frame/base/dpd_block_scatter.cxx
    tblis/frame/base/env.cxx
    tblis/frame/base/tensor.cxx
    tblis/frame/base/thread.cxx
)

add_library(tblis-objects OBJECT ${TBLIS_SOURCES})
target_link_libraries(tblis-objects PUBLIC tblis-plugin ${BLIS_TARGET} marray tci-static)
target_include_directories(tblis-objects PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/stl_ext>
    $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
)
set_target_properties(tblis-objects PROPERTIES
    POSITION_INDEPENDENT_CODE 1
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_compile_options(tblis-objects
  PRIVATE -DMARRAY_ENABLE_ASSERTS -DTBLIS_DEBUG ${OMP_SIMD_FLAG} $<$<CONFIG:Debug>:-O0>
)
add_dependencies(tblis-objects plugin-build)

set(TBLIS_TARGET_SPECS "")

if (ENABLE_SHARED)
    list(APPEND TBLIS_TARGET_SPECS "tblis SHARED")
endif()

if (ENABLE_STATIC)
    list(APPEND TBLIS_TARGET_SPECS "tblis-static STATIC")
endif()

foreach(TGT_SPEC IN LISTS TBLIS_TARGET_SPECS)
    string(REPLACE " " ";"  TGT_SPEC "${TGT_SPEC}")
    list(GET TGT_SPEC 0 TARGET_NAME)
    list(GET TGT_SPEC 1 LIB_TYPE)

    add_library(${TARGET_NAME} ${LIB_TYPE} $<TARGET_OBJECTS:tblis-objects>)

    if(LIB_TYPE STREQUAL "SHARED")
        target_link_libraries(${TARGET_NAME}
            PRIVATE ${PRIVATE_LIBS} tblis-plugin ${BLIS_TARGET}
            PUBLIC ${PUBLIC_LIBS} marray tci-static
        )
    else()
        target_link_libraries(${TARGET_NAME}
            PUBLIC ${PUBLIC_LIBS} tblis-plugin ${BLIS_TARGET} marray tci-static
            PRIVATE ${PRIVATE_LIBS}
        )
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME tblis
    )

    if(LIB_TYPE STREQUAL "SHARED")
        set_target_properties(${TARGET_NAME} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
            INSTALL_NAME_DIR ${INSTALL_LIBDIR}
        )
    else()
        set_target_properties(${TARGET_NAME} PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
        )
    endif()

    list(APPEND TBLIS_TARGETS ${TARGET_NAME})

    target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${INSTALL_INCLUDEDIR}>
    )
endforeach()

list(GET TBLIS_TARGETS 0 TBLIS_MAIN_TARGET)
list(REVERSE TBLIS_TARGETS)
list(GET TBLIS_TARGETS 0 TBLIS_STATIC_IF_POSSIBLE)

# Select header files to be installed
target_sources(${TBLIS_MAIN_TARGET}
    INTERFACE FILE_SET tblis_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/tblis ${CMAKE_CURRENT_BINARY_DIR}/tblis
    FILES
        tblis/frame/base/aligned_allocator.hpp
        tblis/frame/base/alignment.hpp
        tblis/frame/base/basic_types.h
        tblis/frame/base/thread.h
        tblis/frame/1t/add.h
        tblis/frame/1t/dot.h
        tblis/frame/1t/reduce.h
        tblis/frame/1t/scale.h
        tblis/frame/1t/set.h
        tblis/frame/1t/shift.h
        tblis/frame/3t/mult.h
        tblis/tblis.h
)

target_sources(${TBLIS_MAIN_TARGET}
    INTERFACE FILE_SET config_header TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/tblis
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tblis/tblis_config.h
)

target_sources(${TBLIS_MAIN_TARGET}
    INTERFACE FILE_SET public_headers TYPE HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES tblis.h
)

# If building a shared tblis library, the user will also
# need to link the static BLIS and plugin libraries
if(ENABLE_STATIC)
    install(FILES
        ${TBLIS_PLUGIN_DIR}/lib/${BLIS_CONFIG_FAMILY}/libblis_tblis.a
        DESTINATION ${INSTALL_LIBDIR}/tblis
    )
    if (BLIS_POPULATED)
        install(FILES
            ${BLIS_INSTALL_DIR}/lib/libblis.a
            DESTINATION ${INSTALL_LIBDIR}/tblis
            RENAME libblis_core.a
        )
    endif()

    target_link_directories(tblis-static BEFORE
        PUBLIC ${INSTALL_LIBDIR}/tblis
    )

    list(APPEND TBLIS_TARGETS tblis-plugin)
    if (BLIS_POPULATED)
        list(APPEND TBLIS_TARGETS tblis-blis)
    endif()
endif()

install(TARGETS ${TBLIS_TARGETS}
    EXPORT tblis-targets
    FILE_SET tblis_headers DESTINATION "${INSTALL_INCLUDEDIR}/tblis"
    FILE_SET config_header DESTINATION "${INSTALL_INCLUDEDIR}/tblis"
    FILE_SET public_headers DESTINATION "${INSTALL_INCLUDEDIR}"
)

install(EXPORT tblis-targets
    FILE TBLISTargets.cmake
        NAMESPACE TBLIS::
        DESTINATION ${INSTALL_LIBDIR}/cmake/TBLIS
)

write_basic_package_version_file(
    "TBLISConfigVersion.cmake"
    VERSION ${CMAKE_PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/TBLISConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/TBLISConfigVersion.cmake"
    DESTINATION ${INSTALL_LIBDIR}/cmake/TBLIS)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tblis.pc" DESTINATION ${INSTALL_LIBDIR}/pkgconfig)

###############################################################################
#
# Configure tests
#
###############################################################################

if(ENABLE_TESTS)

    set(TEST_SOURCES
        test/test.cxx
        test/random.cxx
        test/1t/dot.cxx
        test/1t/reduce.cxx
        test/1t/replicate.cxx
        test/1t/scale.cxx
        test/1t/trace.cxx
        test/1t/transpose.cxx
        test/3m/gemm.cxx
        test/3m/gemv.cxx
        test/3m/ger.cxx
        test/3t/contract.cxx
        test/3t/mult.cxx
        test/3t/outer_prod.cxx
        test/3t/weight.cxx
    )

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG 2b60af89e23d28eefc081bc930831ee9d45ea58b # v3.8.1
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(tblis-test ${TEST_SOURCES})
    target_link_libraries(tblis-test
        PUBLIC ${TBLIS_STATIC_IF_POSSIBLE} Catch2::Catch2
    )
    target_include_directories(tblis-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tblis/external/stl_ext>
    )
    set_target_properties(tblis-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
        RUNTIME_OUTPUT_NAME test
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
    target_compile_options(tblis-test
        PRIVATE -DMARRAY_ENABLE_ASSERTS -DTBLIS_DEBUG ${OMP_SIMD_FLAG} $<$<CONFIG:Debug>:-O0>
    )

endif()
