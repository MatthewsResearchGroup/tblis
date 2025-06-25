AC_PREREQ([2.70])
AC_INIT([tblis],[2.0],[damatthews@smu.edu],[tblis],[http://www.github.com/devinamatthews/tblis])
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

VERSION_MAJOR=${PACKAGE_VERSION%%.*}
VERSION_MINOR=${PACKAGE_VERSION#*.}
VERSION_PATCH=${VERSION_MINOR#*.}
if test "x$VERSION_PATCH" = x"$VERSION_MINOR"; then
    VERSION_PATCH=0
else
    VERSION_MINOR=${VERSION_MINOR%%.*}
fi

AC_DEFINE_UNQUOTED([VERSION_MAJOR], [$VERSION_MAJOR], [Major version number.])
AC_DEFINE_UNQUOTED([VERSION_MINOR], [$VERSION_MINOR], [Minor version number.])
AC_DEFINE_UNQUOTED([VERSION_PATCH], [$VERSION_PATCH], [Patch version number.])

AC_CONFIG_MACRO_DIR([m4])

AC_CONFIG_HEADERS(config.h)

: ${CFLAGS="-O2 -g"}
: ${CXXFLAGS="-O2 -g"}
CXXFLAGS="$CXXFLAGS -Wno-return-type-c-linkage"

AC_PROG_CC([icc gcc clang])
if test "x$ac_cv_prog_cc_c11" = "xno" && \
   test "x$ac_cv_prog_cc_c99" = "xno"; then
    AC_MSG_ERROR(A C99/C11 compiler is required)
fi
AM_PROG_AS
AC_PROG_EGREP
AC_PROG_AWK
AC_PROG_CXX([icpc g++ clang++])
AX_CXX_COMPILE_STDCXX([20], [noext])

#
# Apple's sys/cdefs.h redefines __restrict to null in C++ mode (!!!!!!)
# Stick with __restrict__ for now, as it works with everything except MSVC
#
#AC_C_RESTRICT

AC_DEFINE([RESTRICT], [__restrict__], [More convenient macro for restrict.])

AC_SEARCH_LIBS([clock_gettime], [rt])

topdir="\"`(cd $srcdir && pwd)`\""
AC_DEFINE_UNQUOTED([TOPDIR], [$topdir], [The top source directory.])
AC_SUBST([TOPDIR], [$topdir])

AM_CONDITIONAL(IS_CLANG, [$CXX --version 2>/dev/null | grep -q clang])

AM_CONDITIONAL(IS_OSX, [test `uname -s` = Darwin])

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
# Find BLIS and set up plugin
#

AC_ARG_WITH([blis-dir], AS_HELP_STRING([--with-blis-dir=dir],
    [search for BLIS in the indicated directory, or 'download' to
     force download of an internal copy. The default value 'auto'
     first searches using pkg-config and then downloads an internal
     copy if not found.]),
    [blis_dir=$with_blis_dir], [blis_dir=auto])
AC_ARG_WITH([blis-confname], AS_HELP_STRING([--with-blis-confname=confname],
    [BLIS configuration to be used if BLIS is downloaded and built.
     The default configuration is 'auto'.]),
    [blis_confname=$with_blis_confname], [blis_confname=auto])

ac_cv_search_libs_blis=no
BUILD_BLIS=no

PKG_CONFIG_PATH_save="$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="$blis_dir/share/pkgconfig:$PKG_CONFIG_PATH"

PKG_CHECK_MODULES([BLIS], [blis >= 2], [
    PKG_CHECK_VAR([BLIS_PREFIX], [blis], [prefix])
], [
    BLIS_PREFIX=none
])

PKG_CONFIG_PATH="$PKG_CONFIG_PATH_save"

AC_MSG_CHECKING([for blis])

if test "x$blis_dir" != "xdownload"; then

    if test "x$BLIS_PREFIX" = "xnone"; then
        if test "x$blis_dir" != "xauto"; then
            AC_MSG_RESULT([not found])
            AC_MSG_ERROR([Could not find BLIS in the specified location])
        fi

        blis_dir=download
    else
        AC_MSG_RESULT([in $BLIS_PREFIX])
        blis_dir="$BLIS_PREFIX"
    fi
fi

AC_SUBST([CONFIG_STATUS_DEPENDENCIES], ['$(top_srcdir)/blis-git-tag'])

if test "x$blis_dir" = "xdownload"; then
    BLIS_REPO=https://github.com/flame/blis
    BLIS_TAG=$(cat $srcdir/blis-git-tag)

    blis_downloaded=no
    if test -d download/blis; then
        cd download/blis
        if test "x$(git rev-parse --short HEAD)" = "x$BLIS_TAG"; then
            blis_downloaded=yes
        fi
        cd ../..
    fi

    if test "x$blis_downloaded" = "xyes"; then
        AC_MSG_RESULT([already downloaded])
    else
        AC_MSG_RESULT([downloading BLIS $BLIS_TAG from $BLIS_REPO])
        mkdir -p download
        rm -rf download/blis
        if ! ( cd download && git clone $BLIS_REPO blis && cd blis && git checkout -q $BLIS_TAG 2>/dev/null ); then
            AC_MSG_ERROR([an error occurred downloading BLIS])
        fi
    fi

    BLIS_PREFIX=$(pwd)/download/blis/installed
    BUILD_BLIS=yes
fi

AC_SUBST([BUILD_BLIS], [$BUILD_BLIS])
AM_CONDITIONAL(BUILD_BLIS, [test "x$BUILD_BLIS" = "xyes"])

LT_INIT

mkdir -p bin

AC_CONFIG_FILES([Makefile])
AX_PREFIX_CONFIG_H([src/tblis_config.h], [TBLIS])
AC_OUTPUT

TBLIS_CONFIG_SUBDIR([src/external/tci])

if test "x$BUILD_BLIS" = "xyes"; then
    ( cd download/blis && ./configure -tomp,pthreads -p "$BLIS_PREFIX" --disable-blas --disable-cblas --enable-static --disable-shared "$blis_confname" )
fi
