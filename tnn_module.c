/* Thunder Neural Networks Module Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_module_bprop(tnn_module *m);
 * tnn_error tnn_module_fprop(tnn_module *m);
 * tnn_error tnn_module_randomize(tnn_module *m);
 * tnn_error tnn_module_destroy(tnn_module *m);
 */

#include "tnn_error.h"
#include "tnn_module.h"

//Polymorphic back-propagation method
tnn_error tnn_module_bprop(tnn_module *m){
  if(m->bprop != NULL){
    return (*m->bprop)(m);
  }
  return TNN_ERROR_SUCCESS;
}

//Polymorphic forward-propagtion method
tnn_error tnn_module_fprop(tnn_module *m){
  if(m->fprop != NULL){
    return (*m->fprop)(m);
  }
  return TNN_ERROR_SUCCESS;
}

//Polymorphic randomize method
tnn_error tnn_module_randomize(tnn_module *m, double k){
  if(m->randomize != NULL){
    return (*m->randomize)(m, k);
  }
  return TNN_ERROR_SUCCESS;
}

//Polymorphic destroy method
tnn_error tnn_module_destroy(tnn_module *m){
  if(m->destroy != NULL){
    return (*m->destroy)(m);
  }
  return TNN_ERROR_SUCCESS;
}
