/* Thunder Neural Networks Module Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_module_bprop(tnn_module *m);
 * tnn_error tnn_module_fprop(tnn_module *m);
 * tnn_error tnn_module_randomize(tnn_module *m);
 * tnn_error tnn_module_destroy(tnn_module *m);
 * tnn_error tnn_module_debug(tnn_module *m);
 */

#include <tnn/tnn_error.h>
#include <tnn/tnn_module.h>

//Polymorphic back-propagation method
tnn_error tnn_module_bprop(tnn_module *m){
  if(m->bprop != NULL){
    return (*m->bprop)(m);
  }
  return TNN_ERROR_MODULE_FUNCNDEF;
}

//Polymorphic forward-propagtion method
tnn_error tnn_module_fprop(tnn_module *m){
  if(m->fprop != NULL){
    return (*m->fprop)(m);
  }
  return TNN_ERROR_MODULE_FUNCNDEF;
}

//Polymorphic randomize method
tnn_error tnn_module_randomize(tnn_module *m, double k){
  if(m->randomize != NULL){
    return (*m->randomize)(m, k);
  }
  return TNN_ERROR_MODULE_FUNCNDEF;
}

//Polymorphic destroy method
tnn_error tnn_module_destroy(tnn_module *m){
  if(m->destroy != NULL){
    return (*m->destroy)(m);
  }
  return TNN_ERROR_MODULE_FUNCNDEF;
}

//Polymorphic debug helper
tnn_error tnn_module_debug(tnn_module *m){
  tnn_error ret;

  if(m->debug != NULL){
    printf("polymorphic module: ");
    return (*m->debug)(m);
  }
  printf("polymphic module: debug function undefined.\n");
  printf("module (unknown) = %p, prev = %p, next = %p, type = %d, constant = %p\n", m, m->prev, m->next, m->t, m->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", m->bprop, m->fprop, m->randomize, m->destroy, m->debug);
  printf("paramter: ");
  if((ret = tnn_state_debug(&m->w)) != TNN_ERROR_SUCCESS){
    printf("module (unknown) paramter state debug error\n");
    return ret;
  }
  printf("input: ");
  if((ret = tnn_state_debug(m->input)) != TNN_ERROR_SUCCESS){
    printf("module (unknown) input state debug error\n");
    return ret;
  }
  printf("output: ");
  if((tnn_state_debug(m->output)) != TNN_ERROR_SUCCESS){
    printf("module (unknown) output state debug error\n");
    return ret;
  }
  return TNN_ERROR_MODULE_FUNCNDEF;
}
