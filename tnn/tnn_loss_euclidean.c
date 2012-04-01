/* Thunder Neural Networks Loss - Euclidean Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/19/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_loss_init_euclidean(tnn_loss *l, tnn_state *input1, tnn_state *input2, tnn_state *output);
 * tnn_error tnn_loss_bprop_euclidean(tnn_loss *l);
 * tnn_error tnn_loss_fprop_euclidean(tnn_loss *l);
 * tnn_error tnn_loss_randomize_euclidean(tnn_loss *l, double k);
 * tnn_error tnn_loss_destroy_euclidean(tnn_loss *l);
 * tnn_error tnn_loss_debug_euclidean(tnn_loss *l);
 */

#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_macro.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_loss.h>
#include <tnn/tnn_loss_euclidean.h>

tnn_error tnn_loss_init_euclidean(tnn_loss *l, tnn_state *input1, tnn_state *input2, tnn_state *output){
  //Check inputs
  if(input1->size != input2->size || output->size !=1){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Defined type
  l->t = TNN_LOSS_TYPE_EUCLIDEAN;

  //No constant paramters
  l-> c = NULL;

  //No paramters
  l->w.valid = false;

  //Link the inputs and outputs
  l->input1 = input1;
  l->input2 = input2;
  l->output = output;

  //Store the functions
  l->bprop = &tnn_loss_bprop_euclidean;
  l->fprop = &tnn_loss_fprop_euclidean;
  l->randomize = &tnn_loss_randomize_euclidean;
  l->destroy = &tnn_loss_destroy_euclidean;
  l->debug = &tnn_loss_debug_euclidean;

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_loss_bprop_euclidean(tnn_loss *l){
  //Routine check
  if(l->t != TNN_LOSS_TYPE_EUCLIDEAN){
    return TNN_ERROR_LOSS_MISTYPE;
  }
  if(l->input1->valid != true || l->input2->valid != true || l->output->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //bprop to input1 and input2 dx = dl 2 (x-y); dy = dl 2 (y-x)
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&l->input1->x, &l->input1->dx));
  TNN_MACRO_GSLTEST(gsl_blas_daxpy(-1.0, &l->input2->x, &l->input1->dx));
  gsl_blas_dscal(2.0*gsl_vector_get(&l->output->dx, 0), &l->input1->dx);
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&l->input1->dx, &l->input2->dx));
  gsl_blas_dscal(-1.0, &l->input2->dx);

  return TNN_ERROR_SUCCESS;
}


tnn_error tnn_loss_fprop_euclidean(tnn_loss *l){
  gsl_vector *diff;
  double loss;

  //Routine check                                                                                                                                   
  if(l->t != TNN_LOSS_TYPE_EUCLIDEAN){
    return TNN_ERROR_LOSS_MISTYPE;
  }
  if(l->input1->valid != true || l->input2->valid != true || l->output->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Do the forward propagation
  if((diff = gsl_vector_alloc(l->input1->size)) == NULL){
    return TNN_ERROR_GSL;
  }
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&l->input1->x, diff));
  TNN_MACRO_GSLTEST(gsl_blas_daxpy(-1.0, &l->input2->x, diff));
  loss = gsl_blas_dnrm2(diff);
  gsl_vector_set(&l->output->x, 0, loss*loss);
  gsl_vector_free(diff);

  return TNN_ERROR_SUCCESS;
}


tnn_error tnn_loss_randomize_euclidean(tnn_loss *l, double k){
  //Do nothing
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_loss_destroy_euclidean(tnn_loss *l){
  //Do nothing
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_loss_debug_euclidean(tnn_loss *l){
  tnn_error ret;

  //Routine check
  if(l->t != TNN_LOSS_TYPE_EUCLIDEAN){
    return TNN_ERROR_LOSS_MISTYPE;
  }

  printf("loss (euclidean) = %p, type = %d, const = %p\n", l, l->t, l->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", l->bprop, l->fprop, l->randomize, l-> destroy, l->debug);
  printf("paramters: ");
  if((ret = tnn_state_debug(&l->w))!=TNN_ERROR_SUCCESS){
    printf("loss (euclidean) paramter state debug error\n");
    return ret;
  }
  printf("input 1: ");
  if((ret = tnn_state_debug(l->input1)) != TNN_ERROR_SUCCESS){
    printf("loss (euclidean) input1 state debug error\n");
    return ret;
  }
  printf("input 2: ");
  if((ret = tnn_state_debug(l->input2)) != TNN_ERROR_SUCCESS){
    printf("loss (euclidean) input2 state debug error\n");
    return ret;
  }
  printf("output: ");
  if((ret = tnn_state_debug(l->output)) != TNN_ERROR_SUCCESS){
    printf("loss (euclidean) output state debug error\n");
    return ret;
  }
  return TNN_ERROR_SUCCESS;
}
