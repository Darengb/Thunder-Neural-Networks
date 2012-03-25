/* Thunder Neural Networks Machine Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_machine_init(tnn_machine *m, int ninput, int noutput);
 * tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param *p);
 * tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param *io);
 * tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_get_sou(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_bprop(tnn_machine *m);
 * tnn_error tnn_machine_fprop(tnn_machine *m);
 * tnn_error tnn_machine_randomize(tnn_machine *m, double k);
 * tnn_error tnn_machine_destroy(tnn_machine *m);
 * tnn_error tnn_machine_debug(tnn_machine *m);
 */

#include <stdio.h>
#include <tnn_error.h>
#include <tnn_param.h>
#include <tnn_module.h>
#include <tnn_machine.h>
#include <utlist.h>

//Initialize the machine with designated input and output size
tnn_error tnn_machine_init(tnn_machine *m, int ninput, int noutput){
  return TNN_ERROR_SUCCESS;
}

//Get the parameter of this machine
tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param *p){
  return TNN_ERROR_SUCCESS;
}

//Get the io paramter of this machine
tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param *io){
  return TNN_ERROR_SUCCESS;
}

//Allocate state in io for this machine
tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s){
  return TNN_ERROR_SUCCESS;
}

//Allocate state in io for this machine, initialized to 0
tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s){
  return TNN_ERROR_SUCCESS;
}

//Get the input state of this machine
tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state *s){
  return TNN_ERROR_SUCCESS;
}

//Get the output state of this machine
tnn_error tnn_machine_get_sout(tnn_machine *m, tnn_state *s){
  return TNN_ERROR_SUCCESS;
}

//Append a module to the machine
tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod){
  return TNN_ERROR_SUCCESS;
}

//Prepend a module to the machine
tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod){
  return TNN_ERROR_SUCCESS;
}

//Get the input module of this machine
tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module *mod){
  return TNN_ERROR_SUCCESS;
}

//Get the output module of this machine
tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module *mod){
  return TNN_ERROR_SUCCESS;
}

//Run back propagation backward with respect to modules
tnn_error tnn_machine_bprop(tnn_machine *m){
  return TNN_ERROR_SUCCESS;
}

//Run forward propagation forward with respect to modules
tnn_error tnn_machine_fprop(tnn_machine *m){
  return TNN_ERROR_SUCCESS;
}

//Run randomize for all of the modules
tnn_error tnn_machine_randomize(tnn_machine *m, double k){
  return TNN_ERROR_SUCCESS;
}

//Destroy this machine
//Note: states in io parameter will be freed!
tnn_error tnn_machine_destroy(tnn_machine *m){
  return TNN_ERROR_SUCCESS;
}

//Debug this machine
//Run debug for all of the components.
tnn_error tnn_machine_debug(tnn_machine *m){
  return TNN_ERROR_SUCCESS;
}
