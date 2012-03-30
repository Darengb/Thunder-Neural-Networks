/* Thunder Neural Networks Regularizer - l2 Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The source implements the following functions:
 * tnn_error tnn_reg_init_l2(tnn_reg *r);
 * tnn_error tnn_reg_l_l2(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_d_l2(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_debug_l2(tnn_reg *r);
 * tnn_error tnn_reg_destroy_l2(tnn_reg *r);
 */

#include <tnn_error.h>
#include <tnn_reg.h>
#include <tnn_reg_l2.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

tnn_error tnn_reg_init_l2(tnn_reg *r){
  //Defined type
  r->t = TNN_REG_TYPE_L2;

  //No constant paramters
  r->c = NULL;

  //Store the functions
  r->l = &tnn_reg_l_l2;
  r->d = &tnn_reg_d_l2;
  r->debug = &tnn_reg_debug_l2;
  r->destroy = &tnn_reg_destroy_l2;

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_l_l2(tnn_reg *r, gsl_vector *w, double *l){
  //Get the sum of absolute values
  *l = gsl_blas_dnrm2(w);
  *l = (*l)*(*l);
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_d_l2(tnn_reg *r, gsl_vector *w, gsl_vector *d){
  //Routine check
  if(w->size != d->size){
    return TNN_ERROR_REG_INCOMP;
  }

  //Compute the derivatives
  gsl_vector_set_zero(d);
  if(gsl_blas_daxpy(2.0, w, d) != 0){
    return TNN_ERROR_GSL;
  }

  return TNN_ERROR_SUCCESS;

}

tnn_error tnn_reg_debug_l2(tnn_reg *r){
  printf("regularizer (l2) = %p, type = %d, l = %p, d = %p, debug = %p\n",
	 r, r->t, r->l, r->d, r-> debug);
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_reg_destroy_l2(tnn_reg *r){
  //Do nothing
  return TNN_ERROR_SUCCESS;
}
