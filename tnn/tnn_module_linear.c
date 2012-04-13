/* Thunder Neural Networks Module - Linear Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_module_init_linear(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p);
 * tnn_error tnn_module_bprop_linear(tnn_module *m);
 * tnn_error tnn_module_fprop_linear(tnn_module *m);
 * tnn_error tnn_module_randomize_linear(tnn_module *m, double k);
 * tnn_error tnn_module_destroy_linear(tnn_module *m);
 * tnn_error tnn_module_debug(tnn_module *m);
 */

#include <stddef.h>
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
#include <tnn/tnn_module.h>
#include <tnn/tnn_numeric.h>
#include <tnn/tnn_module_linear.h>

tnn_error tnn_module_init_linear(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p){
  tnn_error ret;

  //Defined type
  m->t = TNN_MODULE_TYPE_LINEAR;

  //No constant paramters
  m->c = NULL;
  
  //Allocate the parameter states
  tnn_state_init(&m->w, input->size*output->size);
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(p,&m->w), ret);

  //Link the inputs and outputs
  m->input = input;
  m->output = output;

  //Store the functions
  m->bprop = &tnn_module_bprop_linear;
  m->fprop = &tnn_module_fprop_linear;
  m->randomize = &tnn_module_randomize_linear;
  m->destroy = &tnn_module_destroy_linear;
  m->debug = &tnn_module_debug_linear;
  m->clone = &tnn_module_clone_linear;

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_bprop_linear(tnn_module *m){
  tnn_error ret;
  gsl_matrix w;
  gsl_matrix dw;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_LINEAR){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true || m->w.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Transform the matrix
  TNN_MACRO_ERRORTEST(tnn_numeric_v2m(&m->w.x, &w, m->output->size, m->input->size),ret);
  TNN_MACRO_ERRORTEST(tnn_numeric_v2m(&m->w.dx, &dw, m->output->size, m->input->size), ret);

  //bprop to input
  TNN_MACRO_GSLTEST(gsl_blas_dgemv(CblasTrans, 1.0, &w, &m->output->dx, 0.0, &m->input->dx));

  //bprop to dw
  gsl_matrix_set_zero(&dw);
  TNN_MACRO_GSLTEST(gsl_blas_dger(1.0, &m->output->dx, &m->input->x, &dw));

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_fprop_linear(tnn_module *m){
  tnn_error ret;
  gsl_matrix w;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_LINEAR){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true || m->w.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Transform the matrix
  TNN_MACRO_ERRORTEST(tnn_numeric_v2m(&m->w.x, &w, m->output->size, m->input->size),ret);

  //Compute the result using BLAS
  TNN_MACRO_GSLTEST(gsl_blas_dgemv(CblasNoTrans, 1.0, &w, &m->input->x, 0.0, &m->output->x));

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_randomize_linear(tnn_module *m, double k){
  double z;
  size_t i;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_LINEAR){
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

tnn_error tnn_module_destroy_linear(tnn_module *m){
  //Do nothing...
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_clone_linear(tnn_module *m1, tnn_module *m2, tnn_param *p, tnn_pstable *t){
  tnn_error ret;

  //Routine check
  if(m1->t != TNN_MODULE_TYPE_LINEAR){
    return TNN_ERROR_MODULE_MISTYPE;
  }

  //Retrieve input and output
  TNN_MACRO_ERRORTEST(tnn_pstable_find(t, m1->input, &m2->input), ret);
  TNN_MACRO_ERRORTEST(tnn_pstable_find(t, m1->output, &m2->output), ret);
  if(m1->input->size != m2->input->size || m1->output->size != m2->output->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Defined type
  m2->t = TNN_MODULE_TYPE_LINEAR;

  //No constant paramters
  m2->c = NULL;
  
  //Allocate the parameter states
  tnn_state_init(&m2->w, m2->input->size*m2->output->size);
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(p,&m2->w), ret);

  //Store the functions
  m2->bprop = &tnn_module_bprop_linear;
  m2->fprop = &tnn_module_fprop_linear;
  m2->randomize = &tnn_module_randomize_linear;
  m2->destroy = &tnn_module_destroy_linear;
  m2->debug = &tnn_module_debug_linear;
  m2->clone = &tnn_module_clone_linear;

  //Copy the state
  TNN_MACRO_ERRORTEST(tnn_state_copy(&m1->w, &m2->w), ret);

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_debug_linear(tnn_module *m){
  tnn_error ret;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_LINEAR){
    printf("module (linear) mistype\n");
    return TNN_ERROR_MODULE_MISTYPE;
  }

  printf("module (linear) = %p, prev = %p, next = %p, type = %d, constant = %p\n", m, m->prev, m->next, m->t, m->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", m->bprop, m->fprop, m->randomize, m->destroy, m->debug);
  printf("paramter: ");
  if((ret = tnn_state_debug(&m->w)) != TNN_ERROR_SUCCESS){
    printf("module (linear) debug error\n");
    return ret;
  }
  printf("input: ");
  if((ret = tnn_state_debug(m->input)) != TNN_ERROR_SUCCESS){
    printf("module (linear) debug error\n");
    return ret;
  }
  printf("output: ");
  if((ret = tnn_state_debug(m->output)) != TNN_ERROR_SUCCESS){
    printf("module (linear) debug error\n");
    return ret;
  }
  return TNN_ERROR_SUCCESS;
}
