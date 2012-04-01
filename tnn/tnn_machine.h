/* Thunder Neural Networks Machine Utilities Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/25/2012
 *
 * This header defines the following structure:
 * tnn_machine(tnn_state sin, tnn_state sout, tnn_param io, tnn_module *m, tnn_param p,
 *             tnn_module min, tnn_module mout)
 *
 * This header defines the following functions:
 * tnn_error tnn_machine_init(tnn_machine *m, size_t ninput, size_t noutput);
 * tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param **p);
 * tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param **io);
 * tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state **s);
 * tnn_error tnn_machine_get_sout(tnn_machine *m, tnn_state **s);
 * tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module **mod);
 * tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module **mod);
 * tnn_error tnn_machine_bprop(tnn_machine *m);
 * tnn_error tnn_machine_fprop(tnn_machine *m);
 * tnn_error tnn_machine_randomize(tnn_machine *m, double k);
 * tnn_error tnn_machine_destroy(tnn_machine *m);
 * tnn_error tnn_machine_debug(tnn_machine *m);
 */

#include <stddef.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_module.h>
#include <tnn/utlist.h>

#ifndef TNN_MACHINE_H
#define TNN_MACHINE_H

//A backward DL linked list operation.
//Tail is at head->prev
#define DL_FOREACH_BACKWARD(head, el) \
  for(el=(head? head->prev : 0L); el; el=(el==head ? 0L : el->prev))

//The structure
typedef struct __STRUCT_tnn_machine{
  //Input state: make min use it!
  tnn_state sin;
  //Output state: make mout use it!
  tnn_state sout;
  //Input and output parameters: all states stored in it will be destroyed when destroy this machine!
  tnn_param io;
  //Store all the modules: in sequence forward and backward propagation!
  //Use non-circular double linked list!
  tnn_module *m;
  //Store all parameters in the modules
  tnn_param p;
  //Input module
  tnn_module min;
  //Output module
  tnn_module mout;
} tnn_machine;

//Initialize the machine with designated input and output size
tnn_error tnn_machine_init(tnn_machine *m, size_t ninput, size_t noutput);

//Get the parameter of this machine
tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param **p);

//Get the io paramter of this machine
tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param **io);

//Allocate state in io for this machine
tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s);

//Allocate state in io for this machine, initialized to 0
tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s);

//Get the input state of this machine
tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state **s);

//Get the output state of this machine
tnn_error tnn_machine_get_sout(tnn_machine *m, tnn_state **s);

//Append a module to the machine
tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod);

//Prepend a module to the machine
tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod);

//Get the input module of this machine
tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module **mod);

//Get the output module of this machine
tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module **mod);

//Run back propagation backward with respect to modules
tnn_error tnn_machine_bprop(tnn_machine *m);

//Run forward propagation forward with respect to modules
tnn_error tnn_machine_fprop(tnn_machine *m);

//Run randomize for all of the modules
tnn_error tnn_machine_randomize(tnn_machine *m, double k);

//Destroy this machine
//Note: states in io parameter will be freed!
tnn_error tnn_machine_destroy(tnn_machine *m);

//Debug this machine
//Run debug for all of the components.
tnn_error tnn_machine_debug(tnn_machine *m);

#endif //TNN_MACHINE_H
