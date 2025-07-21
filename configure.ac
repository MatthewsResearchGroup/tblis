AC_PREREQ([2.70])
AC_INIT([tblis],[1.3.0],[damatthews@smu.edu],[tblis],[http://www.github.com/devinamatthews/tblis])
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS(config.h)

: ${CFLAGS="-O3"}
: ${CXXFLAGS="-O3"}

AC_PROG_CC([icc gcc clang])
if test "x$ac_cv_prog_cc_c11" = "xno" && \
   test "x$ac_cv_prog_cc_c99" = "xno"; then
    AC_MSG_ERROR(A C99/C11 compiler is required)
fi
AM_PROG_AS
AC_PROG_EGREP
AC_PROG_AWK
AX_BLAS([blas_found=yes])
AC_PROG_CXX([icpc g++ clang++])
AX_CXX_COMPILE_STDCXX([17], [noext])

#
# Apple's sys/cdefs.h redefines __restrict to null in C++ mode (!!!!!!)
# Stick with __restrict__ for now, as it works with everything except MSVC
#
#AC_C_RESTRICT

AC_DEFINE([RESTRICT], [__restrict__], [More convenient macro for restrict.])
AM_CONDITIONAL([ENABLE_BLAS], [test x"$blas_found" = xyes])

AC_SEARCH_LIBS([clock_gettime], [rt])

topdir="\"`(cd $srcdir && pwd)`\""
AC_DEFINE_UNQUOTED([TOPDIR], [$topdir], [The top source directory.])

AM_CONDITIONAL(IS_CLANG, [$CXX --version 2>/dev/null | grep -q clang])

#
# Check for openmp simd flag
#

AC_MSG_CHECKING([for OpenMP SIMD pragma])
omp_simd_flag="not supported"
for flag in -fopenmp-simd -qopenmp-simd; do
    CXXFLAGS_save="$CXXFLAGS"
    CXXFLAGS="$CXXFLAGS $flag -Werror"
    AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <cstdio>]], [[
        auto f = [](int* x){
        #pragma omp simd
        for (int i = 0;i < 256;i++)
            x[i] = i;
        }; printf("%p", &f);
        ]])],[omp_simd_flag="$flag"],[])
    CXXFLAGS="$CXXFLAGS_save"
done
AC_MSG_RESULT([$omp_simd_flag])
if test "x$omp_simd_flag" != "xnot supported"; then
    CXXFLAGS="$CXXFLAGS $omp_simd_flag"
fi

#
# Check for memkind
#

AC_ARG_WITH([memkind], AS_HELP_STRING([--without-memkind],
    [do not attempt to link with the memkind library]))

if test "x$with_memkind" != "xno"; then
    AC_CHECK_HEADERS([hbwmalloc.h memkind.h])
    AC_SEARCH_LIBS([hbw_malloc], [memkind])
fi

#
# Check for hwloc
#

AC_ARG_WITH([hwloc], AS_HELP_STRING([--without-hwloc],
    [do not attempt to link with the hwloc library]))

if test "x$with_hwloc" != "xno"; then
    AC_CHECK_HEADERS([hwloc.h])
    AC_SEARCH_LIBS([hwloc_topology_init], [hwloc])
fi

#
# Check for lscpu
#

AC_CHECK_PROGS([LSCPU], [lscpu], [:])
if test "x$LSCPU" != "x:"; then
    AC_DEFINE([HAVE_LSCPU], [1], [Define if the system has the lscpu command.])
fi

#
# Check for sysctl
#

AC_CHECK_FUNCS([sysctl sysctlbyname])

#
# Check for sysconf
#

AC_CHECK_FUNCS([sysconf])
AX_CHECK_DEFINE([unistd.h], [_SC_NPROCESSORS_CONF],
                [AC_DEFINE([HAVE__SC_NPROCESSORS_CONF], [1],
                           [sysconf(_SC_NPROCESSORS_CONF) is valid.])])
AX_CHECK_DEFINE([unistd.h], [_SC_NPROCESSORS_ONLN],
                [AC_DEFINE([HAVE__SC_NPROCESSORS_ONLN], [1],
                           [sysconf(_SC_NPROCESSORS_ONLN) is valid.])])

#
# Determine basic types
#

AC_ARG_WITH([length-type], AS_HELP_STRING([--with-length-type@<:@=type@:>@],
    [use 'type' for the default index length type @<:@default=ptrdiff_t@:>@]),
    [len_type=$with_length_type], [len_type=ptrdiff_t])

AC_CHECK_TYPE([$len_type],,
              [AC_MSG_ERROR([$len_type must be a standard ISO C type])])
AX_CHECK_SIGN([$len_type],,
              [AC_MSG_ERROR([$len_type must be signed])])
AC_MSG_CHECKING([for index length type])
AC_MSG_RESULT([$len_type])
AC_DEFINE_UNQUOTED([LEN_TYPE], [$len_type], [len_type])

AC_ARG_WITH([stride-type], AS_HELP_STRING([--with-stride-type@<:@=type@:>@],
    [use 'type' for the default index stride type @<:@default=ptrdiff_t@:>@]),
    [stride_type=$with_stride_type], [stride_type=ptrdiff_t])

AC_CHECK_TYPE([$stride_type],,
              [AC_MSG_ERROR([$stride_type must be a standard ISO C type])])
AX_CHECK_SIGN([$stride_type],,
              [AC_MSG_ERROR([$stride_type must be signed])])
AC_MSG_CHECKING([for index stride type])
AC_MSG_RESULT([$stride_type])
AC_DEFINE_UNQUOTED([STRIDE_TYPE], [$stride_type], [stride_type])

AC_ARG_WITH([label-type], AS_HELP_STRING([--with-label-type@<:@=type@:>@],
    [use 'type' for the default index label type @<:@default=char@:>@]),
    [label_type=$with_label_type], [label_type=char])

AC_CHECK_TYPE([$label_type],,
              [AC_MSG_ERROR([$label_type must be a standard ISO C type])])
AC_MSG_CHECKING([for index label type])
AC_MSG_RESULT([$label_type])
AC_DEFINE_UNQUOTED([LABEL_TYPE], [$label_type], [label_type])

#
# Determine configurations to build
#
# Supported:
#
# reference, bulldozer, piledriver, excavator, zen, core2, sandybridge,
# haswell, skx1, skx2, knl, auto
#
# Possible with porting from BLIS:
#
# armv7a, armv8a, bgq, cortex-a15, cortex-a9, power7, loongson3a, mic
#

AC_ARG_ENABLE([config],
[  --enable-config=...    a comma-separated list of configurations
                          @<:@default=auto@:>@

                          possible values are:

                          reference, bulldozer, piledriver, excavator, zen,
                          core2, sandybridge, haswell, skx1, skx2, knl,
                          firestorm, auto

                          the following meta-configurations are also available:

                          skx = skx1,skx2
                          intel = core2,sandybridge,haswell,skx,knl
                          amd = bulldozer,piledriver,excavator,zen
                          x86 = intel,amd
                          arm64 = firestorm],
              [], [enable_config=auto])

if echo $enable_config | grep -q auto; then
    enable_config=
    using_default_configs=yes
#    AC_CHECK_DEFINE([__MIC__], [enable_config=mic])
    AC_CHECK_DEFINE([__x86_64__], [enable_config=x86])
    AC_CHECK_DEFINE([_M_X64], [enable_config=x86])
    AC_CHECK_DEFINE([__i386], [enable_config=x86])
    AC_CHECK_DEFINE([_M_IX86], [enable_config=x86])
    AC_CHECK_DEFINE([__aarch64__], [enable_config=arm64])
#    AC_CHECK_DEFINE([__arm__], [enable_config=arm])
#    AC_CHECK_DEFINE([_M_ARM], [enable_config=arm])
#    AC_CHECK_DEFINE([__powerpc64__], [enable_config=power7])
#    AC_CHECK_DEFINE([__ppc64__], [enable_config=power7])
#    AC_CHECK_DEFINE([__PPC64__], [enable_config=power7])
#    AC_CHECK_DEFINE([__powerpc__], [enable_config=power7])
#    AC_CHECK_DEFINE([__ppc__], [enable_config=power7])
#    AC_CHECK_DEFINE([__PPC__], [enable_config=power7])
#    AC_CHECK_DEFINE([__bgq__], [enable_config=bgq])
#    AC_CHECK_DEFINE([__mips], [enable_config=loongson3a])
    AC_CHECK_DEFINE([__INTEL_COMPILER], [enable_config=intel])
fi

if ( echo $enable_config | grep -qv reference ) && test x$using_default_configs = xyes; then
    enable_config="${enable_config},reference"
fi

configs=`echo $enable_config | sed 's/x86/intel,amd/' \
                             | sed 's/intel/core2,sandybridge,haswell,skx,knl/' \
                             | sed 's/amd/bulldozer,piledriver,excavator,zen/' \
                             | sed 's/skx,/skx1,skx2,/' \
                             | sed 's/skx$/skx1,skx2/' \
                             | sed 's/arm64/firestorm/' \
                             | sed 's/,/ /g'`

for config in $configs; do
    if ( echo $config | $EGREP -qv '^(reference|core2|sandybridge|haswell|skx1|skx2|knl|bulldozer|piledriver|excavator|zen|firestorm)$' ); then
        AC_MSG_ERROR([Unknown configuration $config])
    fi
done

AC_MSG_CHECKING([compiler vendor])

vendor_string=`$CC --version 2>/dev/null`
if test x"$vendor_string" = x; then
    vendor_string=`$CC -qversion 2>/dev/null`
fi
if test x"$vendor_string" = x; then
    AC_MSG_ERROR([Unable to determine compiler vendor.])
fi

cc_vendor=`echo $vendor_string | $EGREP -oi 'icc|gcc|clang|emcc|pnacl|IBM|gnu-cc' | tr 'A-Z' 'a-z' | { read first rest ; echo $first ; }`
if test x"$cc_vendor" = "xgnu-cc"; then
    cc_vendor=gcc
fi
if test x"$cc_vendor" = x; then
    AC_MSG_ERROR([Unable to determine compiler vendor.])
fi

AC_MSG_RESULT([$cc_vendor])

if test x"$cc_vendor" = xicc || \
   test x"$cc_vendor" = xgcc || \
   test x"$cc_vendor" = xclang; then
    cc_version=`echo $vendor_string | $EGREP -o '@<:@0-9@:>@+\.@<:@0-9@:>@+\.?@<:@0-9@:>@*' | { read first rest ; echo $first ; }`
else
    AC_MSG_ERROR([Unsupported compiler.])
fi

cc_major=`echo $cc_version | cut -d. -f1`
cc_minor=`echo $cc_version | cut -d. -f2`
cc_revision=`echo $cc_version | cut -d. -f3`

AC_MSG_CHECKING([compiler version])
if test x"$cc_revision" = x; then
    AC_MSG_RESULT([$cc_major.$cc_minor])
else
    AC_MSG_RESULT([$cc_major.$cc_minor.$cc_revision])
fi

#
# Compiler requirements:
#
# General: icc 15+, gcc 4.7+, clang 3.3+
#
# bulldozer: any
# piledriver: any
# excavator: gcc 4.9+, clang 3.5+
# zen: gcc 6.0+, clang 4.0+
#
# core2: any
# sandybridge: any
# haswell: any
# knl: icc 14.1+, gcc 5.0+, clang 3.5+
# skx[12]: icc 15.1+, gcc 6.0+, clang 3.9+
#
if test x"$cc_vendor" = xicc; then
    if test $cc_major -lt 15; then
        AC_MSG_ERROR([Unsupported compiler version.])
    fi
    if test $cc_major -eq 15; then
        if test $cc_revision -lt 1; then
            configs=`echo $configs | sed 's/ *skx1 */ /' \
                                   | sed 's/ *skx2 */ /'`
        fi
    fi
elif test x"$cc_vendor" = xgcc; then
    if test $cc_major -lt 4; then
        AC_MSG_ERROR([Unsupported compiler version.])
    fi
    if test $cc_major -eq 4; then
        if test $cc_minor -lt 7; then
            AC_MSG_ERROR([Unsupported compiler version.])
        fi
        if test $cc_minor -lt 9; then
            configs=`echo $configs |  sed 's/ *excavator */ /'`
        fi
        configs=`echo $configs | sed 's/ *knl */ /'`
    fi
    if test $cc_major -lt 6; then
        configs=`echo $configs | sed 's/ *skx1 */ /' \
                               | sed 's/ *skx2 */ /' \
                               | sed 's/ *zen */ /'`
    fi
elif test x"$cc_vendor" = xclang; then
    if test $cc_major -lt 3; then
        AC_MSG_ERROR([Unsupported compiler version.])
    fi
    if test $cc_major -eq 3; then
        if test $cc_minor -lt 3; then
            AC_MSG_ERROR([Unsupported compiler version.])
        fi
        if test $cc_minor -lt 5; then
            configs=`echo $configs | sed 's/ *knl */ /' \
                                   | sed 's/ *excavator */ /'`
        fi
        if test $cc_minor -lt 9; then
            configs=`echo $configs | sed 's/ *skx1 */ /' \
                                   | sed 's/ *skx2 */ /'`
        fi
    fi
    if test $cc_major -lt 4; then
        configs=`echo $configs | sed 's/ *zen */ /'`
    fi
fi

if ( echo $configs | grep -q sandybridge ); then
    AC_MSG_CHECKING([whether inline AVX can be compiled])
    AC_TRY_ASSEMBLE([
        vzeroall
        vmovapd %ymm0, %ymm1
        vmulpd %ymm0, %ymm0, %ymm1
        ], [enable_avx=yes], [enable_avx=no])
    AC_MSG_RESULT([$enable_avx])
    if test "x$enable_avx" = xno; then
        AC_MSG_WARN([AVX cannot be compiled; disabling sandybridge configuration])
        configs=`echo $configs | sed 's/ *sandybridge */ /'`
    fi
fi

if ( echo $configs | $EGREP -q 'haswell|piledriver|excavator|skx' ); then
    AC_MSG_CHECKING([whether inline FMA3 can be compiled])
    AC_TRY_ASSEMBLE([
        vzeroall
        vfmadd213pd %ymm0, %ymm1, %ymm2
        ], [enable_fma3=yes], [enable_fma3=no])
    AC_MSG_RESULT([$enable_fma3])
    if test "x$enable_fma3" = xno; then
        AC_MSG_WARN([FMA3 cannot be compiled; disabling haswell, piledriver, excavator, skx, and zen configurations])
        configs=`echo $configs | sed 's/ *haswell */ /' \
                               | sed 's/ *piledriver */ /' \
                               | sed 's/ *excavator */ /' \
                               | sed 's/ *skx1 */ /' \
                               | sed 's/ *skx2 */ /' \
                               | sed 's/ *zen */ /'`
    fi
fi

if ( echo $configs | grep -q bulldozer ); then
    AC_MSG_CHECKING([whether inline FMA4 can be compiled])
    AC_TRY_ASSEMBLE([
        vzeroall
        vfmaddpd %ymm0, %ymm1, %ymm2, %ymm3
        ], [enable_fma4=yes], [enable_fma4=no])
    AC_MSG_RESULT([$enable_fma4])
    if test "x$enable_fma4" = xno; then
        AC_MSG_WARN([FMA4 cannot be compiled; disabling bulldozer configuration])
        configs=`echo $configs | sed 's/ *bulldozer */ /'`
    fi
fi

if ( echo $configs | $EGREP -q 'knl|skx' ); then
    CPPFLAGS_save="$CPPFLAGS"
    if test `uname -s` = Darwin; then
        CPPFLAGS="$CPPFLAGS -Wa,-march=knl"
    fi
    AC_MSG_CHECKING([whether inline AVX-512F can be compiled])
    AC_TRY_ASSEMBLE([
        vmovapd %zmm0, %zmm1
        vmulpd %zmm0, %zmm0, %zmm1
        vfmadd213pd 0x400(%rax,%rsi,8) {1to8}, %zmm1, %zmm2
        ], [enable_avx512f=yes], [enable_avx512f=no])
    AC_MSG_RESULT([$enable_avx512f])
    if test "x$enable_avx512f" = xno; then
        AC_MSG_WARN([AVX-512F cannot be compiled; disabling knl and skx configurations])
        configs=`echo $configs | sed 's/ *knl */ /' \
                               | sed 's/ *skx1 */ /' \
                               | sed 's/ *skx2 */ /'`
    fi
    CPPFLAGS="$CPPFLAGS_save"
fi

if ( echo $configs | $EGREP -q 'skx' ); then
    CPPFLAGS_save="$CPPFLAGS"
    if test `uname -s` = Darwin; then
        CPPFLAGS="$CPPFLAGS -Wa,-march=skylake-avx512"
    fi
    AC_MSG_CHECKING([whether inline AVX-512DQ can be compiled])
    AC_TRY_ASSEMBLE([
        vpmullq %zmm0, %zmm0, %zmm1
        ], [enable_avx512dq=yes], [enable_avx512dq=no])
    AC_MSG_RESULT([$enable_avx512dq])
    if test "x$enable_avx512dq" = xno; then
        AC_MSG_WARN([AVX-512DQ cannot be compiled; disabling skx configuration])
        configs=`echo $configs | sed 's/ *skx1 */ /' \
                               | sed 's/ *skx2 */ /'`
    fi
    CPPFLAGS="$CPPFLAGS_save"
fi

AM_CONDITIONAL(ENABLE_INTEL_COMPILER, [test x"$cc_vendor" = xicc])

AM_CONDITIONAL(IS_OSX, [test `uname -s` = Darwin])

if test -z `echo $configs | tr -d "$IFS"`; then
    AC_MSG_ERROR([no configurations enabled])
fi

AC_MSG_CHECKING([enabled configurations])
AC_MSG_RESULT([$configs])

AM_CONDITIONAL(ENABLE_ARMV7A, [echo $configs | grep -q armv7a])
AM_CONDITIONAL(ENABLE_ARMV8A, [echo $configs | grep -q armv8a])
AM_CONDITIONAL(ENABLE_BGQ, [echo $configs | grep -q bgq])
AM_CONDITIONAL(ENABLE_BULLDOZER, [echo $configs | grep -q bulldozer])
AM_CONDITIONAL(ENABLE_EXCAVATOR, [echo $configs | grep -q excavator])
AM_CONDITIONAL(ENABLE_CORTEX_A15, [echo $configs | grep -q cortex-a15])
AM_CONDITIONAL(ENABLE_CORTEX_A9, [echo $configs | grep -q cortex-a9])
AM_CONDITIONAL(ENABLE_CORE2, [echo $configs | grep -q core2])
AM_CONDITIONAL(ENABLE_HASWELL, [echo $configs | grep -q haswell])
AM_CONDITIONAL(ENABLE_SKX1, [echo $configs | grep -q skx1])
AM_CONDITIONAL(ENABLE_SKX2, [echo $configs | grep -q skx2])
AM_CONDITIONAL(ENABLE_KNL, [echo $configs | grep -q knl])
AM_CONDITIONAL(ENABLE_LOONGSON3A, [echo $configs | grep -q loongson3a])
AM_CONDITIONAL(ENABLE_MIC, [echo $configs | grep -q mic])
AM_CONDITIONAL(ENABLE_PILEDRIVER, [echo $configs | grep -q piledriver])
AM_CONDITIONAL(ENABLE_POWER7, [echo $configs | grep -q power7])
AM_CONDITIONAL(ENABLE_REFERENCE, [echo $configs | grep -q reference])
AM_CONDITIONAL(ENABLE_SANDYBRIDGE, [echo $configs | grep -q sandybridge])
AM_CONDITIONAL(ENABLE_ZEN, [echo $configs | grep -q zen])
AM_CONDITIONAL(ENABLE_FIRESTORM, [echo $configs | grep -q firestorm])

include_configs=""
for config in $configs; do
include_configs="$include_configs
#include \"configs/$config/config.hpp\""
done

AC_SUBST([INCLUDE_CONFIGS], [$include_configs])
AM_SUBST_NOTMAKE([INCLUDE_CONFIGS])

foreach_config=""
for config in $configs; do
foreach_config="$foreach_config
FOREACH_CONFIG(${config}_config)"
done

AC_SUBST([FOREACH_CONFIG], [$foreach_config])
AM_SUBST_NOTMAKE([FOREACH_CONFIG])

foreach_config_and_type=""
for config in $configs; do
foreach_config_and_type="$foreach_config_and_type
#define FOREACH_TYPE(type) \\
FOREACH_CONFIG_AND_TYPE(type, ${config}_config)
#include \"foreach_type.h\""
done

AC_SUBST([FOREACH_CONFIG_AND_TYPE], [$foreach_config_and_type])
AM_SUBST_NOTMAKE([FOREACH_CONFIG_AND_TYPE])

types="float double scomplex dcomplex"
foreach_type=""
for type in $types; do
foreach_type="$foreach_type
FOREACH_TYPE($type)"
done

AC_SUBST([FOREACH_TYPE], [$foreach_type])
AM_SUBST_NOTMAKE([FOREACH_TYPE])

LT_INIT

mkdir -p bin

AC_CONFIG_FILES([Makefile
                 src/configs/foreach_type.h
                 src/configs/foreach_config.h
                 src/configs/foreach_config_and_type.h
                 src/configs/include_configs.hpp])
AX_PREFIX_CONFIG_H([src/tblis_config.h], [TBLIS])
AC_OUTPUT

TBLIS_CONFIG_SUBDIR([src/external/tci])
