/* Thunder Neural Networks Numerical Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_numeric_v2m(gsl_vector *v, gsl_matrix *m, size_t size1, size_t size2);
 */

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <tnn_numeric.h>
#include <tnn_error.h>

//Vector to matrix conversion
tnn_error tnn_numeric_v2m(gsl_vector *v, gsl_matrix *m, size_t size1, size_t size2){
  if(v->stride != 1 || size1*size2 != v->size){
    return TNN_ERROR_NUMERIC_INCOMP;
  }

  m->size1 = size1;
  m->size2 = size2;
  m->tda = size2;
  m->data = v->data;
  m->block = v->block;
  m->owner = 0; //Does not own

  return TNN_ERROR_SUCCESS;
}
