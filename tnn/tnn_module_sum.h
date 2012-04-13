/* Thunder Neural Networks Module - Sum Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/10/2012
 *
 * This header defines the following structure:
 * tnn_module_sum(int n, UT_array *sarray)
 *
 * This header defines the following functions:
 * tnn_error tnn_module_init_sum(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *io);
 * tnn_error tnn_module_bprop_sum(tnn_module *m);
 * tnn_error tnn_module_fprop_sum(tnn_module *m);
 * tnn_error tnn_module_randomize_sum(tnn_module *m, double k);
 * tnn_error tnn_module_destroy_sum(tnn_module *m);
 * tnn_error tnn_module_clone_sum(tnn_module *m1, tnn_module *m2, tnn_param *p, tnn_pstable *io);
 * tnn_error tnn_module_debug_sum(tnn_module *m);
 * tnn_error tnn_module_sum_get(tnn_module *m, tnn_state **t, size_t ind); 
 */

#include <tnn/tnn_error.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_param.h>
#include <tnn/utarray.h>

#ifndef TNN_MODULE_SUM_H
#define TNN_MODULE_SUM_H

//The structure
typedef struct __STRUCT_tnn_module_sum{
  //Number of input states
  int n;
  //Array of input states
  UT_array *sarray;
} tnn_module_sum;

//Function definitions
tnn_error tnn_module_init_sum(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *io);
tnn_error tnn_module_bprop_sum(tnn_module *m);
tnn_error tnn_module_fprop_sum(tnn_module *m);
tnn_error tnn_module_randomize_sum(tnn_module *m, double k);
tnn_error tnn_module_destroy_sum(tnn_module *m);
tnn_error tnn_module_clone_sum(tnn_module *m1, tnn_module *m2, tnn_param *p, tnn_pstable *io);
tnn_error tnn_module_debug_sum(tnn_module *m);
tnn_error tnn_module_sum_get(tnn_module *m, tnn_state **t, size_t ind); 

#endif //TNN_MODULE_SUM_H
