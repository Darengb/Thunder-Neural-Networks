/* Dummy Test 5 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/26/2012
 *
 * Tests for the following utilities were performed:
 * tnn_machine
 *
 * Results:
 * 1. Initialization error in conditional expression
 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_machine.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_linear.h>
#include <tnn/tnn_module_bias.h>

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

#define A 10
#define B 5
#define C 22
#define D 3

int main(){
  int i;
  tnn_machine m;
  tnn_module *mod;
  tnn_param *p;
  tnn_module *min;
  tnn_module *mout;
  tnn_state *sin;
  tnn_state *sout;
  tnn_state *in;
  tnn_state *out;

  //Initialize the machine
  printf("Initializing machine m: %s\n", TEST_FUNC(tnn_machine_init(&m, A, A)));
  printf("Get machine paramters: %s\n", TEST_FUNC(tnn_machine_get_param(&m, &p)));
  printf("Get machine input module: %s\n", TEST_FUNC(tnn_machine_get_min(&m, &min)));
  printf("Get machine output module: %s\n", TEST_FUNC(tnn_machine_get_mout(&m, &mout)));
  printf("Get machine input state: %s\n", TEST_FUNC(tnn_machine_get_sin(&m, &sin)));
  printf("Get machine output state: %s\n", TEST_FUNC(tnn_machine_get_sout(&m, &sout)));

  //Initializing modules
  out = malloc(sizeof(tnn_state));
  printf("Initializing first hidden state: %s\n", TEST_FUNC(tnn_state_init(out, A)));
  printf("Allocating first hidden state: %s\n", TEST_FUNC(tnn_machine_state_alloc(&m, out)));
  printf("Debugging first hidden state: %s\n", TEST_FUNC(tnn_state_debug(out)));
  printf("Initializing min: %s\n", TEST_FUNC(tnn_module_init_bias(min, sin, out, p)));
  printf("Debugging min: %s\n", TEST_FUNC(tnn_module_debug(min)));
  in = out;
  out = malloc(sizeof(tnn_state));
  printf("Initializing second hidden state: %s\n", TEST_FUNC(tnn_state_init(out, B)));
  printf("Allocating second hidden state %s\n", TEST_FUNC(tnn_machine_state_alloc(&m, out)));
  printf("Debugging second hidden state %s\n", TEST_FUNC(tnn_state_debug(out)));
  mod = malloc(sizeof(tnn_module));
  printf("Initializing first hidden layer: %s\n", TEST_FUNC(tnn_module_init_linear(mod, in, out, p)));
  printf("Append first hidden layer: %s\n", TEST_FUNC(tnn_machine_module_append(&m, mod)));
  printf("Debugging first hidden layer: %s\n", TEST_FUNC(tnn_module_debug(mod)));
  in = out;
  out = malloc(sizeof(tnn_state));
  printf("Initializing third hidden state: %s\n", TEST_FUNC(tnn_state_init(out, B)));
  printf("Allocating third hidden state %s\n", TEST_FUNC(tnn_machine_state_alloc(&m, out)));
  printf("Debugging third hidden state %s\n", TEST_FUNC(tnn_state_debug(out)));
  mod = malloc(sizeof(tnn_module));
  printf("Initializing second hidden layer: %s\n", TEST_FUNC(tnn_module_init_linear(mod, in, out, p)));
  printf("Append second hidden layer: %s\n", TEST_FUNC(tnn_machine_module_append(&m, mod)));
  printf("Debugging second hidden layer: %s\n", TEST_FUNC(tnn_module_debug(mod)));
  in = out;
  printf("Initializing mout: %s\n", TEST_FUNC(tnn_module_init_linear(mout, in, sout, p)));
  printf("Debugging mout: %s\n", TEST_FUNC(tnn_module_debug(mout)));

  printf("Debugging machine: %s\n", TEST_FUNC(tnn_machine_debug(&m)));

  printf("Randomize machine: %s\n", TEST_FUNC(tnn_machine_randomize(&m, 1.0)));
  printf("Debugging machine: %s\n", TEST_FUNC(tnn_machine_debug(&m)));

  printf("Initializing input data\n");
  for(i = 0; i < A; i = i + 1){
    gsl_vector_set(&sin->x, i, (double)(C + i));
    gsl_vector_set(&sout->dx, i, (double)(D + 2*i));
  }
  printf("Debugging sin: %s\n", TEST_FUNC(tnn_state_debug(sin)));
  printf("Debugging sout: %s\n", TEST_FUNC(tnn_state_debug(sout)));


  printf("Running forward propagation: %s\n", TEST_FUNC(tnn_machine_fprop(&m)));
  printf("Debugging machine: %s\n", TEST_FUNC(tnn_machine_debug(&m)));

  printf("Running backward propagation: %s\n", TEST_FUNC(tnn_machine_bprop(&m)));
  printf("Debugging machine: %s\n", TEST_FUNC(tnn_machine_debug(&m)));
  return 0;
  
}
