/* Thunder Neural Networks Regularizer - l1 Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The source implements the following functions:
 * tnn_error tnn_reg_init_l1(tnn_reg *r);
 * tnn_error tnn_reg_l_l1(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_d_l1(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_debug_l1(tnn_reg *r);
 * tnn_error tnn_reg_destroy_l1(tnn_reg *r);
 */

#include <stddef.h>
#include <tnn_error.h>
#include <tnn_reg.h>
#include <tnn_reg_l1.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

tnn_error tnn_reg_init_l1(tnn_reg *r){
  //Defined type
  r->t = TNN_REG_TYPE_L1;

  //No constant paramters
  r->c = NULL;

  //Store the functions
  r->l = &tnn_reg_l_l1;
  r->d = &tnn_reg_d_l1;
  r->debug = &tnn_reg_debug_l1;
  r->destroy = &tnn_reg_destroy_l1;

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_l_l1(tnn_reg *r, gsl_vector *w, double *l){
  //Get the sum of absolute values
  *l = gsl_blas_dasum(w);

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_d_l1(tnn_reg *r, gsl_vector *w, gsl_vector *d){
  size_t i;

  //Routine check
  if(w->size != d->size){
    return TNN_ERROR_REG_INCOMP;
  }

  for(i = 0; i < w->size; i = i + 1){
    if(gsl_vector_get(w, i) > 0){
      gsl_vector_set(d, i, 1.0);
    } else if(gsl_vector_get(w, i) < 0){
      gsl_vector_set(d, i, -1.0);
    } else {
      gsl_vector_set(d, i, 0.0);
    }
  }

  return TNN_ERROR_SUCCESS;

}

tnn_error tnn_reg_debug_l1(tnn_reg *r){
  printf("regularizer (l1) = %p, type = %d, l = %p, d = %p, debug = %p\n",
	 r, r->t, r->l, r->d, r-> debug);
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_destroy_l1(tnn_reg *r){
  //Do nothing
  return TNN_ERROR_SUCCESS;
}
