AC_INIT(TBLIS,
    m4_normalize(m4_include([tblis-version])),
    damatthews@smu.edu,
    tblis,
    http://www.github.com/MatthewsResearchGroup/tblis)

AC_CONFIG_MACRO_DIR([cmake])

AX_CMAKE_OPTION([tests], AS_HELP_STRING([--enable-tests], [Enable building tests.]))

AX_CMAKE_OPTION([static], AS_HELP_STRING([--enable-static], [Enable building a static library.]))

AX_CMAKE_OPTION([shared], AS_HELP_STRING([--disable-static], [Disable building a shared library.]))

AX_CMAKE_OPTION([compat], AS_HELP_STRING([--enable-compat], [Maintain compatibility with the TBLIS 1.x interface.]))

AX_CMAKE_OPTION([memkind], AS_HELP_STRING([--enable-memkind], [Enable use of the memkind library for MCDRAM allocation if supported.]))

AX_CMAKE_OPTION([hwloc], AS_HELP_STRING([--disable-hwloc], [Disable use of the hwloc library.]))

AX_CMAKE_ENABLE([threading], AS_HELP_STRING([--enable-threading=<type>],
    [enable threading with the specific threading model, as supported by TCI]))

AX_CMAKE_WITH([blis-config-family], AS_HELP_STRING([--with-blis-config-family=config],
    [use the specific BLIS sub-configuration @<:@default=auto@:>@]))

AX_CMAKE_WITH([blis-thread-model], AS_HELP_STRING([--with-blis-thread-model=model],
    [use the specific BLIS sub-configuration @<:@default=pthread@:>@]))

AX_CMAKE_WITH([length-type], AS_HELP_STRING([--with-length-type=type],
    [use 'type' for the default index length type @<:@default=ptrdiff_t@:>@]))

AX_CMAKE_WITH([stride-type], AS_HELP_STRING([--with-stride-type=type],
    [use 'type' for the default index stride type @<:@default=ptrdiff_t@:>@]))

AX_CMAKE_WITH([label-type], AS_HELP_STRING([--with-label-type=type],
    [use 'type' for the default index label type @<:@default=char@:>@]))

AC_ARG_WITH([blis-prefix], AS_HELP_STRING([--with-blis-prefix=dir],
    [Search for BLIS also in the indicated directory.]),
    [BLIS_PREFIX="$with_blis_prefix:"])

BUILD_TYPE=Release
AC_ARG_ENABLE([debug], AS_HELP_STRING([--enable-debug],
    [Enable debugging mode.]), [BUILD_TYPE=Debug])

AX_CMAKE([-DCMAKE_BUILD_TYPE=${BUILD_TYPE}],
         [PKG_CONFIG_PATH="${BLIS_PREFIX}${PKG_CONFIG_PATH}" CMAKE_PREFIX_PATH="${BLIS_PREFIX}${CMAKE_PREFIX_PATH}"])