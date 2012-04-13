/* Dummy Test 4 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/12/2012
 *
 * Tests for the following utilities were performed:
 * tnn_module_sum
 *
 * Results:
 * Perfect Coding!
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_sum.h>

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

#define A 40
#define B 10
#define C 2
#define D 7

int main(){
  tnn_param p, io;
  tnn_state in, out;
  tnn_module m;
  int i;

  printf("Initializing state in: %s\n", TEST_FUNC(tnn_state_init(&in, A)));
  printf("Initializing state out: %s\n", TEST_FUNC(tnn_state_init(&out, B)));
  printf("Initializing paramter p: %s\n", TEST_FUNC(tnn_param_init(&p)));
  printf("Initializing paramter io: %s\n", TEST_FUNC(tnn_param_init(&io)));
  printf("Allocating state in: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &in)));
  printf("Allocating state out: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &out)));
  printf("Initializing module m: %s\n", TEST_FUNC(tnn_module_init_sum(&m, &in, &out, &io)));
  printf("Debugging module m: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Setting input\n");
  for(i = 0; i < in.size; i = i + 1){
    gsl_vector_set(&in.x, i, (double)(C+i));
  }

  printf("Setting output\n");
  for(i = 0; i < out.size; i = i + 1){
    gsl_vector_set(&out.dx, i, 2.0*(double)(D+i));
  }

  printf("Radomizing weights using polymorphic call: %s\n", TEST_FUNC(tnn_module_randomize(&m, 1.0)));
  printf("Debugging module m: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Executing fprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_fprop(&m)));
  printf("Debugging module m: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Executing bprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_bprop(&m)));
  printf("Debugging module m: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Debugging parameter p: %s\n", TEST_FUNC(tnn_param_debug(&p)));
  printf("Debugging parameter io: %s\n", TEST_FUNC(tnn_param_debug(&io)));

  printf("Destroying paramter p: %s\n", TEST_FUNC(tnn_param_destroy(&p)));
  printf("Destroying paramter io: %s\n", TEST_FUNC(tnn_param_destroy(&io)));
  printf("Destroy module m using polymorphic call: %s\n", TEST_FUNC(tnn_module_destroy(&m)));
  printf("Debugging module m: %s\n", TEST_FUNC(tnn_module_debug(&m)));
  
  return 0;
}
