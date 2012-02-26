/* Thunder Neural Networks Loss Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * This header defines the following structure:
 * tnn_loss(tnn_loss_type t, void *c, tnn_state w, tnn_state *input1, tnn_state *input2, tnn_state *output,
 *           TNN_LOSS_FUNC_BPROP bprop,
 *           TNN_LOSS_FUNC_FPROP fprop,
 *           TNN_LOSS_FUNC_RANDOMIZE randomize,
 *           TNN_LOSS_FUNC_DESTROY destroy)
 *
 * This header defines the following polymorphic functions:
 * tnn_error tnn_loss_bprop(tnn_loss *l);
 * tnn_error tnn_loss_fprop(tnn_loss *l);
 * tnn_error tnn_loss_randomize(tnn_loss *l);
 * tnn_error tnn_loss_destroy(tnn_module *m);
 */

#include "tnn_error.h"
#include "tnn_state.h"

#ifndef TNN_LOSS_H
#define TNN_LOSS_H

//Module types
typedef enum __ENUM_tnn_loss_type{
  TNN_LOSS_TYPE_NONE, //Undefined (or user-defined) type
  TNN_LOSS_TYPE_EUCL, //Euclidean loss
  TNN_LOSS_TYPE_CENT, //Cross-entropy loss

  TNN_LOSS_TYPE_SIZE //Size indicator (if you want to define your own polymorph-safe module, do it above this size.)
} tnn_loss_type;

//Function types
struct __STRUCT_tnn_loss;
typedef tnn_error (*TNN_LOSS_FUNC_BPROP) (struct __STRUCT_tnn_loss *loss);
typedef tnn_error (*TNN_LOSS_FUNC_FPROP) (struct __STRUCT_tnn_loss *loss);
typedef tnn_error (*TNN_LOSS_FUNC_RANDOMIZE) (struct __STRUCT_tnn_loss *loss);
typedef tnn_error (*TNN_LOSS_FUNC_DESTROY) (struct __STRUCT_tnn_loss *loss);

//The structure
typedef struct __STRUCT_tnn_loss{
  //Loss type
  tnn_loss_type t;
  //Constant paramters struct
  void *c;
  //Variable paramters
  tnn_state w;
  //Input states
  tnn_state *input1;
  tnn_state *input2;
  //Output states
  tnn_state *output;
  //Back-propagation method
  TNN_LOSS_FUNC_BPROP bprop;
  //Forward-propagation method
  TNN_LOSS_FUNC_FPROP fprop;
  //Randomization method
  TNN_LOSS_FUNC_RANDOMIZE randomize;
  //Destroy method
  TNN_LOSS_FUNC_DESTROY destroy;

  //UTList operation support
  struct __STRUCT_tnn_loss *prev;
  struct __STRUCT_tnn_loss *next;
} tnn_loss;

//Polymorphic back-propagation method
tnn_error tnn_loss_bprop(tnn_loss *l);
//Polymorphic forward-propagation method
tnn_error tnn_loss_fprop(tnn_loss *l);
//Polymorphic randomize method
tnn_error tnn_loss_randomize(tnn_loss *l);
//Polymorphic destroy method
tnn_error tnn_loss_destroy(tnn_loss *l);

#endif //TNN_LOSS_H
