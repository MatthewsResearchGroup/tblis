AC_INIT(marray, 1.0, damatthews@smu.edu, marray, http://www.github.com/devinamatthews/marray)
AC_CONFIG_AUX_DIR([.])
AM_INIT_AUTOMAKE([foreign silent-rules subdir-objects])
AM_SILENT_RULES([yes])
AM_MAINTAINER_MODE([disable])

AC_CONFIG_MACRO_DIR([m4])

AC_PROG_CXX([clang++])
AX_CXX_COMPILE_STDCXX([17], [noext])
AC_LANG([C++])
LT_INIT

CXXFLAGS="$CXXFLAGS -DMARRAY_ENABLE_ASSERTS"

AC_ARG_ENABLE([code-coverage], AS_HELP_STRING([--enable-code-coverage], [Enable code coverage testing with llvm-cov]))
AM_CONDITIONAL([coverage], [test "x$enable_code_coverage" = "xyes"])

mkdir -p bin

AC_CONFIG_FILES([Makefile])
AC_CONFIG_COMMANDS([quickfix], [sed -i' ' 's/marray\\\ 1.0/marray-1.0/g' Makefile])
AC_OUTPUT
