AC_INIT([tnn],[0.1],[xiang.zhang@nyu.edu])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([-Wall -Werror])

AC_PROG_CC
AC_PROG_RANLIB

AM_PROG_CC_C_O

AC_CONFIG_HEADERS([tnn/tnn_config.h])
AC_CONFIG_FILES([Makefile tnn/Makefile])

AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

AC_OUTPUT