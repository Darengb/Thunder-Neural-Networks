/* Dummy Test 2 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/25/2012
 *
 * Tests for the following utilities were performed:
 * tnn_module, tnn_module_linear, tnn_numeric_v2m
 *
 * Results:
 * Perfect Programming!
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_linear.h>
#include <tnn/tnn_numeric.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_state.h>

#define A 4
#define B 3
#define C 5
#define D 9
#define E 0.9

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

void printState(tnn_state *s);

int main(){
  tnn_param p, io;
  tnn_state in, out;
  tnn_module m;
  int i,j;

  printf("Initializing state in: %s\n", TEST_FUNC(tnn_state_init(&in, A)));
  printf("Initializing state out: %s\n", TEST_FUNC(tnn_state_init(&out, B)));
  printf("Initializing paramter p: %s\n", TEST_FUNC(tnn_param_init(&p)));
  printf("Initializing paramter io: %s\n", TEST_FUNC(tnn_param_init(&io)));
  printf("Allocating state in: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &in)));
  printf("Allocating state out: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &out)));
  printf("Allocating state out again: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &out)));
  printf("Initializing module m: %s\n", TEST_FUNC(tnn_module_init_linear(&m, &in, &out, &p)));
  printf("Initialized module t = %ld, c = %ld, w.size = %d, input = %ld, output = %ld, bprop = %ld, fprop = %ld, randomize = %ld, destroy = %ld\n",
	 (long)m.t, (long)m.c, (int)m.w.size, (long)m.input, (long)m.output, (long)m.bprop, (long)m.fprop, (long)m.randomize, (long)m.destroy);

  printf("Setting input\n");
  for(i = 0; i < in.size; i = i + 1){
    gsl_vector_set(&in.x, i, (double)(C+i));
  }
  printf("Debugging using polymorphic call: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Setting output\n");
  for(i = 0; i < out.size; i = i + 1){
    gsl_vector_set(&out.dx, i, E + (double)i/10.0);
  }
  printf("Debugging using polymorphic call: %s\n", TEST_FUNC(tnn_module_debug(&m)));

  printf("Radomizing weights using polymorphic call: %s\n", TEST_FUNC(tnn_module_randomize(&m, 1.0)));
  printState(&m.w);
  printf("Executing fprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_fprop(&m)));
  printState(&out);
  printf("Executing bprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_bprop(&m)));
  printf("Debugging using polymorphic call: %s\n", TEST_FUNC(tnn_module_debug(&m)));
  printf("Destroying paramter p: %s\n", TEST_FUNC(tnn_param_destroy(&p)));
  printf("Destroying paramter io: %s\n", TEST_FUNC(tnn_param_destroy(&io)));
  printf("Radomizing weights using polymorphic call: %s\n", TEST_FUNC(tnn_module_randomize(&m, 1.0)));
  printf("Executing fprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_fprop(&m)));
  printf("Executing bprop using polymorphic call: %s\n", TEST_FUNC(tnn_module_bprop(&m)));
  printf("Destroy module m using polymorphic call: %s\n", TEST_FUNC(tnn_module_destroy(&m)));
  return 0;
}

void printState(tnn_state *s){
  int i;
  for(i = 0; i < s->size - 1; i = i + 1){
    printf("%.5g ", gsl_vector_get(&s->x, i));
  }
  printf("%.5g\n", gsl_vector_get(&s->x, s->size - 1));
  for(i = 0; i < s->size - 1; i = i + 1){
    printf("%.5g ", gsl_vector_get(&s->dx, i));
  }
  printf("%.5g\n", gsl_vector_get(&s->dx, s->size - 1));
}
