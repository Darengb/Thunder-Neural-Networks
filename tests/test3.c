/* Dummy Test 3 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/20/2012
 *
 * Tests for the following utilities were performed:
 * tnn_loss, tnn_loss_euclidean
 *
 * Results:
 * 1. Forgot to square the norm in computing of loss!
 *
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_loss.h>
#include <tnn/tnn_loss_euclidean.h>

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

#define A 10
#define B 22
#define C 7
#define D 9

int main(){
  tnn_param io;
  tnn_state in1, in2, out;
  tnn_loss l;
  int i;

  printf("Initializing state in1: %s\n", TEST_FUNC(tnn_state_init(&in1, A)));
  printf("Initializing state in2: %s\n", TEST_FUNC(tnn_state_init(&in2, A)));
  printf("Initializing state out: %s\n", TEST_FUNC(tnn_state_init(&out, 1)));
  printf("Initializing parameter io: %s\n", TEST_FUNC(tnn_param_init(&io)));
  printf("Allocating state in1 %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &in1)));
  printf("Allocating state in2 %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &in2)));
  printf("Allocating state out: %s\n", TEST_FUNC(tnn_param_state_alloc(&io, &out)));
  printf("Initializing euclidean loss: %s\n", TEST_FUNC(tnn_loss_init_euclidean(&l, &in1, &in2, &out)));
  printf("Debugging loss: %s\n", TEST_FUNC(tnn_loss_debug(&l)));

  printf("Setting state in1 and in2\n");
  for(i = 0; i < in1.size; i = i + 1){
    gsl_vector_set(&in1.x, i, (double)(B+i));
    gsl_vector_set(&in2.x, i, (double)(2*(C+i)));
  }
  printf("Setting state out\n");
  gsl_vector_set(&out.dx, 0, (double)D);
  printf("Debugging loss: %s\n", TEST_FUNC(tnn_loss_debug(&l)));

  printf("Running forward propagation: %s\n", TEST_FUNC(tnn_loss_fprop(&l)));
  printf("Debugging loss: %s\n", TEST_FUNC(tnn_loss_debug(&l)));

  printf("Running backward propagation: %s\n", TEST_FUNC(tnn_loss_bprop(&l)));
  printf("Debugging loss: %s\n", TEST_FUNC(tnn_loss_debug(&l)));

  printf("Destroying paramber io: %s\n", TEST_FUNC(tnn_param_destroy(&io)));
  printf("Debugging loss: %s\n", TEST_FUNC(tnn_loss_debug(&l)));

  return 0;
}
