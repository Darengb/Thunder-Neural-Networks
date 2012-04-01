AC_INIT([tnn],[0.1],[xiang.zhang@nyu.edu])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([-Wall -Werror])
LT_INIT([disable-static])

AC_PROG_CC

AC_CONFIG_HEADERS([tnn/tnn_config.h])
AC_CONFIG_FILES([Makefile tnn/Makefile])

AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

AC_CHECK_HEADERS([stdio.h stddef.h stdlib.h stdbool.h])
AC_CHECK_HEADERS([gsl/gsl_vector.h gsl/gsl_matrix.h gsl/gsl_blas.h])

AC_OUTPUT