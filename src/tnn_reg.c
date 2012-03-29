/* Thunder Neural Network Regularizer Utility Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The source implements the following functions:
 * tnn_error tnn_reg_l(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_add_l(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_add_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_debug(tnn_reg *r);
 * tnn_error tnn_reg_destroy(tnn_reg *r);
 */

#include <tnn_error.h>
#include <tnn_reg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

//Polymorphically compute the loss of the regularizer
tnn_error tnn_reg_l(tnn_reg *r, gsl_vector *w, double *l){
  if(r->l != NULL){
    return (*r->l)(r, w, l);
  }
  return TNN_ERROR_REG_FUNCNDEF;
}

//Polymorphically compute the derivatives of the regularizer
tnn_error tnn_reg_d(tnn_reg *r, gsl_vector *w, gsl_vector *d){
  if(r->d != NULL){
    return (*r->d)(r, w, d);
  }
  return TNN_ERROR_REG_FUNCNDEF;
}

//Add the loss of the regularizer to the value l
tnn_error tnn_reg_add_l(tnn_reg *r, gsl_vector *w, double *l){
  tnn_error ret;
  double regl;
  if(r->l != NULL){
    //Check whether the execution is successful
    if((ret = (*r->l)(r,w,&regl)) != TNN_ERROR_SUCCESS){
      return ret;
    }
    *l = *l + regl;
    return TNN_ERROR_SUCCESS;
  }
  return TNN_ERROR_REG_FUNCNDEF;
}

//Add the derivatives of the regularizer to the vector d
tnn_error tnn_reg_add_d(tnn_reg *r, gsl_vector *w, gsl_vector *d){
  tnn_error ret;
  gsl_vector *regd;
  if(r->d != NULL){
    regd = gsl_vector_alloc(d->size);
    //Check whether the execution is successful
    if((ret = (*r->d)(r, w, regd)) != TNN_ERROR_SUCCESS){
      gsl_vector_free(regd);
      return ret;
    }
    if(gsl_blas_daxpy(1.0, regd, d) != 0){
      gsl_vector_free(regd);
      return TNN_ERROR_GSL;
    }
    gsl_vector_free(regd);
    return TNN_ERROR_SUCCESS;
  }
  return TNN_ERROR_REG_FUNCNDEF;
}

//Polymorphically debug the regularizer
tnn_error tnn_reg_debug(tnn_reg *r){
  if(r->debug != NULL){
    return (*r->debug)(r);
  }
  printf("regularizer (unknown) = %p, type = %d, l = %p, d = %p, debug = %p\n",
	 r, r->t, r->l, r->d, r-> debug);
  return TNN_ERROR_REG_FUNCNDEF;
}

//Polymorphically destory the regularizer
tnn_error tnn_reg_destroy(tnn_reg *r){
  if(r->destroy != NULL){
    return (*r->destroy)(r);
  }
  return TNN_ERROR_REG_FUNCNDEF;
}
