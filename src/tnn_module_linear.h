/* Thunder Neural Networks Module - Linear Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This header defines the following functions:
 * tnn_error tnn_module_init_linear(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p);
 * tnn_error tnn_module_bprop_linear(tnn_module *m);
 * tnn_error tnn_module_fprop_linear(tnn_module *m);
 * tnn_error tnn_module_randomize_linear(tnn_module *m, double k);
 * tnn_error tnn_module_destroy_linear(tnn_module *m);
 * tnn_error tnn_module_debug_linear(tnn_module *m);
 */

#include <tnn_error.h>
#include <tnn_state.h>
#include <tnn_param.h>
#include <tnn_module.h>

#ifndef TNN_MODULE_LINEAR_H
#define TNN_MODULE_LINEAR_H

//Function definitions
tnn_error tnn_module_init_linear(tnn_module *m, tnn_state *input, tnn_state *output, tnn_param *p);
tnn_error tnn_module_bprop_linear(tnn_module *m);
tnn_error tnn_module_fprop_linear(tnn_module *m);
tnn_error tnn_module_randomize_linear(tnn_module *m, double k);
tnn_error tnn_module_destroy_linear(tnn_module *m);
tnn_error tnn_module_debug_linear(tnn_module *m);

#endif //TNN_MODULE_LINEAR_H
