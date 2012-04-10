/* Thunder Neural Networks Module Utilities Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header defines the following structure:
 * tnn_module(tnn_module_type t, void *c, tnn_state w, tnn_state *input, tnn_state *output,
 *            TNN_MODULE_FUNC_BPROP bprop,
 *            TNN_MODULE_FUNC_FPROP fprop,
 *            TNN_MODULE_FUNC_RANDOMIZE randomize,
 *            TNN_MODULE_FUNC_DESTROY destroy)
 *
 * This header defines the following functions:
 * tnn_error tnn_module_bprop(tnn_module *m);
 * tnn_error tnn_module_fprop(tnn_module *m);
 * tnn_error tnn_module_randomize(tnn_module *m, double k);
 * tnn_error tnn_module_destroy(tnn_module *m);
 * tnn_error tnn_module_debug(tnn_module *m);
 */

#include <tnn/tnn_error.h>
#include <tnn/tnn_state.h>
#ifndef TNN_MODULE_H
#define TNN_MODULE_H

//Module types
typedef enum __ENUM_tnn_module_type{
  TNN_MODULE_TYPE_NONE, //Undefined (or user-defined) type
  TNN_MODULE_TYPE_LINEAR, //Linear module
  TNN_MODULE_TYPE_BIAS, //Bias module
  TNN_MODULE_TYPE_TANH, //Hyperbolic tangent module
  TNN_MODULE_TYPE_SOFTMAX, //Softmax module
  TNN_MODULE_TYPE_NEGEXP, //Exponential of each negated component
  TNN_MODULE_TYPE_SUM, //Sum module
  TNN_MODULE_TYPE_BRANCH, //Branch module
  TNN_MODULE_TYPE_CONV1, //1-D convolutional module
  TNN_MODULE_TYPE_CONV2, //2-D convolutional module

  TNN_MODULE_TYPE_SIZE //Size indicator (if you want to define your own polymorph-safe module, do it above this size.)
} tnn_module_type;

//Function types
struct __STRUCT_tnn_module;
typedef tnn_error (*TNN_MODULE_FUNC_BPROP) (struct __STRUCT_tnn_module *module);
typedef tnn_error (*TNN_MODULE_FUNC_FPROP) (struct __STRUCT_tnn_module *module);
typedef tnn_error (*TNN_MODULE_FUNC_RANDOMIZE) (struct __STRUCT_tnn_module *module, double k);
typedef tnn_error (*TNN_MODULE_FUNC_DESTROY) (struct __STRUCT_tnn_module *module);
typedef tnn_error (*TNN_MODULE_FUNC_DEBUG) (struct __STRUCT_tnn_module *module);

//The structure
typedef struct __STRUCT_tnn_module{
  //Module type
  tnn_module_type t;
  //Constant parameters struct
  void *c;
  //Variable parameters
  tnn_state w;
  //Input states
  tnn_state *input;
  //Output states
  tnn_state *output;
  //Back-propagation method
  TNN_MODULE_FUNC_BPROP bprop;
  //Forward-propagation method
  TNN_MODULE_FUNC_FPROP fprop;
  //Randomization method
  TNN_MODULE_FUNC_RANDOMIZE randomize;
  //Destroy method
  TNN_MODULE_FUNC_DESTROY destroy;
  //Debug method
  TNN_MODULE_FUNC_DEBUG debug;

  //UTList operation support
  struct __STRUCT_tnn_module *prev;
  struct __STRUCT_tnn_module *next;
} tnn_module;

//Polymorphic back-propagation method
tnn_error tnn_module_bprop(tnn_module *m);
//Polymorphic forward-propagtion method
tnn_error tnn_module_fprop(tnn_module *m);
//Polymorphic randomize method
tnn_error tnn_module_randomize(tnn_module *m, double k);
//Polymorphic destroy method
tnn_error tnn_module_destroy(tnn_module *m);
//Polymorphic Debug helper
tnn_error tnn_module_debug(tnn_module *m);

#endif //TNN_MODULE_H
