/* Thunder Neural Networks Numerical Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This header defines the following functions:
 * tnn_error tnn_numeric_v2m(gsl_vector *v, gsl_matrix *m, int size1, int size2);
 */

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <tnn_error.h>

#ifndef TNN_NUMERIC_H
#define TNN_NUMERIC_H

//Vector to matrix conversion
tnn_error tnn_numeric_v2m(gsl_vector *v, gsl_matrix *m, int size1, int size2);

#endif //TNN_NUMERIC_H
