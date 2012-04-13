/* Thunder Neural Networks Module - Sum Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/10/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_module_init_sum(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *io);
 * tnn_error tnn_module_bprop_sum(tnn_module *m);
 * tnn_error tnn_module_fprop_sum(tnn_module *m);
 * tnn_error tnn_module_randomize_sum(tnn_module *m, double k);
 * tnn_error tnn_module_destroy_sum(tnn_module *m);
 * tnn_error tnn_module_debug_sum(tnn_module *m);
 * tnn_error tnn_module_sum_get(tnn_module *m, tnn_state **t, size_t ind);
 */

#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_macro.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_sum.h>
#include <tnn/utarray.h>
#include <gsl/gsl_vector.h>

tnn_error tnn_module_init_sum(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *io){
  tnn_error ret;
  tnn_module_sum *c;
  size_t i;
  UT_icd sarray_icd = {sizeof(tnn_state*), NULL, NULL, NULL};
  tnn_state *t;

  //Check the sizes and validness
  if(input->size % output->size != 0){
    return TNN_ERROR_STATE_INCOMP;
  }
  if(input->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Defined type
  m->t = TNN_MODULE_TYPE_SUM;

  //Constant parameter is a new tnn_module_sum
  c = (tnn_module_sum *)malloc(sizeof(tnn_module_sum));
  m->c = c;

  //Allocate sub-states
  utarray_new(c->sarray, &sarray_icd);
  if(c->sarray == NULL){
    TNN_ERROR_ALLOC;
  }
  for(i = 0; i < input->size; i = i + output->size){
    //Alloc and initialize the state
    t = (tnn_state *)malloc(sizeof(tnn_state));
    if(t == NULL){
      return TNN_ERROR_ALLOC;
    }
    t->size = output->size;

    //Get the substate and store it
    TNN_MACRO_ERRORTEST(tnn_param_state_sub(io, input, t, i), ret);
    utarray_push_back(c->sarray, &t);
  }

  //Init the state
  tnn_state_init(&m->w, 0L);

  //Link the inputs and outputs
  m->input = input;
  m->output = output;

  //Store the functions
  m->bprop = &tnn_module_bprop_sum;
  m->fprop = &tnn_module_fprop_sum;
  m->randomize = &tnn_module_randomize_sum;
  m->destroy = &tnn_module_destroy_sum;
  m->debug = &tnn_module_debug_sum;
  
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_bprop_sum(tnn_module *m){
  tnn_state **t;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_SUM){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //bprop to each input
  for(t = (tnn_state **)utarray_front(((tnn_module_sum*)m->c)->sarray);
      t != NULL;
      t = (tnn_state **)utarray_next(((tnn_module_sum*)m->c)->sarray, t)){
    TNN_MACRO_GSLTEST(gsl_blas_dcopy(&m->output->dx, &(*t)->dx));
  }

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_fprop_sum(tnn_module *m){
  tnn_state **t;

   //Routine check
  if(m->t != TNN_MODULE_TYPE_SUM){
    return TNN_ERROR_MODULE_MISTYPE;
  }
  if(m->input->valid != true || m->output->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //fprop to output
  TNN_MACRO_GSLTEST(gsl_blas_dscal(0.0, &m->output->x));
  for(t = (tnn_state **)utarray_front(((tnn_module_sum*)m->c)->sarray);
      t != NULL;
      t = (tnn_state **)utarray_next(((tnn_module_sum*)m->c)->sarray, t)){
    TNN_MACRO_GSLTEST(gsl_blas_daxpy(1.0, &(*t)->x, &m->output->x));
  }

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_randomize_sum(tnn_module *m, double k){
  //Do nothing...
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_destroy_sum(tnn_module *m){
  //Note: Allocated sub-states will be destroyed outside

  //Destroy the array
  utarray_free(((tnn_module_sum*)m->c)->sarray);
  //Destroy the constant
  free((tnn_module_sum*)m->c);

  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_debug_sum(tnn_module *m){
  tnn_error ret;
  tnn_state **t;
  size_t i;

  //Routine check
  if(m->t != TNN_MODULE_TYPE_SUM){
    printf("module (sum) mistype\n");
    return TNN_ERROR_MODULE_MISTYPE;
  }

  printf("module (sum) = %p, prev = %p, next = %p, type = %d, constant = %p\n", m, m->prev, m->next, m->t, m->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", m->bprop, m->fprop, m->randomize, m->destroy, m->debug);
  printf("paramter: ");
  if((ret = tnn_state_debug(&m->w)) != TNN_ERROR_SUCCESS){
    printf("module (sum) debug error\n");
    return ret;
  }
  printf("input: ");
  if((ret = tnn_state_debug(m->input)) != TNN_ERROR_SUCCESS){
    printf("module (sum) debug error\n");
    return ret;
  }
  printf("output: ");
  if((ret = tnn_state_debug(m->output)) != TNN_ERROR_SUCCESS){
    printf("module (sum) debug error\n");
    return ret;
  }
  for(t = (tnn_state **)utarray_front(((tnn_module_sum*)m->c)->sarray), i = 0L;
      t != NULL;
      t = (tnn_state **)utarray_next(((tnn_module_sum*)m->c)->sarray, t), i = i + 1L){
    printf("subinput #%ld: ", i);
    if((ret = tnn_state_debug(*t))!= TNN_ERROR_SUCCESS){
      printf("module (sum) debug error\n");
      return ret;
    }
  }
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_module_sum_get(tnn_module *m, tnn_state **t, size_t ind){
  tnn_state **s;

  if(ind < utarray_len(((tnn_module_sum*)m->c)->sarray)){
    s = (tnn_state **)utarray_eltptr(((tnn_module_sum*)m->c)->sarray, ind);
    *t = *s;
  } else {
    TNN_ERROR_PARAM_NEXIST;
  }

  return TNN_ERROR_SUCCESS;
}
