/* Thunder Neural Networks Module - Bias Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/23/2012
 *
 * This header implements the following functions:
 * tnn_error tnn_module_init_bias(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p);
 * tnn_error tnn_module_bprop_bias(tnn_module *m);
 * tnn_error tnn_module_fprop_bias(tnn_module *m);
 * tnn_error tnn_module_randomize_bias(tnn_module *m, double k);
 * tnn_error tnn_module_destroy_bias(tnn_module *m);
 * tnn_error tnn_module_debug_bias(tnn_module *m);
 */

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_macro.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_bias.h>

tnn_error tnn_module_init_bias(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p){
  tnn_error ret;

  //Check whether the output state has the same size with the input
  if(input->size != output->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Define type
  m->t = TNN_MODULE_TYPE_BIAS;

  //No consant parameters
  m->c = NULL;

  //Allocate the paramter states
  tnn_state_init(&m->w, input->size);
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(p, &m->w), ret);

  //Link the inputs and outputs
  m->input = input;
  m->output = output;

  //Store the functions
  m->bprop = &tnn_module_bprop_bias;
  m->fprop = &tnn_module_fprop_bias;
  m->randomize = &tnn_module_randomize_bias;
  m->destroy = &tnn_module_destroy_bias;
  m->clone = &tnn_module_clone_bias;
  m->debug = &tnn_module_debug_bias;

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_bprop_bias(tnn_module *m){
  //Routine check
  if(m->t != TNN_MODULE_TYPE_BIAS){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true || m->w.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //bprop to input
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&m->output->dx,  &m->input->dx));

  //bprop to dw
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&m->output->dx,  &m->w.dx));

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_fprop_bias(tnn_module *m){
  //Routine check
  if(m->t != TNN_MODULE_TYPE_BIAS){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true || m->w.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //fprop to output
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&m->input->x, &m->output->x));
  TNN_MACRO_GSLTEST(gsl_blas_daxpy(1.0, &m->w.x, &m->output->x));

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_randomize_bias(tnn_module *m, double k){
  double z;
  size_t i;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_BIAS){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true || m->w.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Initialize
  srand(time(NULL));
  z = k/sqrt((double)m->input->size);

  //Set every element
  for(i = 0; i < m->w.size; i = i + 1){
    gsl_vector_set(&m->w.x, i, 2.0*z*((double)rand()/(double)RAND_MAX) - z);
  }

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_destroy_bias(tnn_module *m){
  //Do nothing...
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_clone_bias(tnn_module *m1, tnn_module *m2, tnn_param *p, tnn_pstable *t){
  tnn_error ret;

  //Routine check
  if(m1->t != TNN_MODULE_TYPE_BIAS){
    return TNN_ERROR_MODULE_MISTYPE;
  }

  //Retrieve input and output
  TNN_MACRO_ERRORTEST(tnn_pstable_find(t, m1->input, &m2->input), ret);
  TNN_MACRO_ERRORTEST(tnn_pstable_find(t, m1->output, &m2->output), ret);
  if(m1->input->size != m2->input->size || m1->output->size != m2->output->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Defined type
  m2->t = TNN_MODULE_TYPE_BIAS;

  //No constant paramters
  m2->c = NULL;
  
  //Allocate the parameter states
  tnn_state_init(&m2->w, m2->input->size);
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(p,&m2->w), ret);

  //Store the functions
  m2->bprop = &tnn_module_bprop_bias;
  m2->fprop = &tnn_module_fprop_bias;
  m2->randomize = &tnn_module_randomize_bias;
  m2->destroy = &tnn_module_destroy_bias;
  m2->debug = &tnn_module_debug_bias;
  m2->clone = &tnn_module_clone_bias;

  //Copy the state
  TNN_MACRO_ERRORTEST(tnn_state_copy(&m1->w, &m2->w), ret);

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_debug_bias(tnn_module *m){
  tnn_error ret;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_BIAS){
    printf("module (bias) mistype\n");
    return TNN_ERROR_MODULE_MISTYPE;
  }

  printf("module (bias) = %p, prev = %p, next = %p, type = %d, constant = %p\n", m, m->prev, m->next, m->t, m->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", m->bprop, m->fprop, m->randomize, m->destroy, m->debug);
  printf("paramter: ");
  if((ret = tnn_state_debug(&m->w)) != TNN_ERROR_SUCCESS){
    printf("module (bias) parameter state debug error\n");
    return ret;
  }
  printf("input: ");
  if((ret = tnn_state_debug(m->input)) != TNN_ERROR_SUCCESS){
    printf("module (bias) input state debug error\n");
    return ret;
  }
  printf("output: ");
  if((tnn_state_debug(m->output)) != TNN_ERROR_SUCCESS){
    printf("module (bias) output state debug error\n");
    return ret;
  }

  return TNN_ERROR_SUCCESS;
}
