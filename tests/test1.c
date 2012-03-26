/* Dummy Test 1 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/20/2012
 *
 * Tests for the following utilities were performed;
 * tnn_state and tnn_param
 *
 * Results:
 * 
 */

#include <stdio.h>
#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>

#define N 5
#define M 3
#define I 6
#define J 2
#define A 4
#define B 7
#define C 8

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

int main(){
  tnn_state s[N];
  tnn_state t, u;
  tnn_param p;
  int i,j;

  //Initialize all the states
  for(i = 0; i < N; i = i + 1){
    printf("Initializing state %d: %s\n", i, TEST_FUNC(tnn_state_init(&s[i], i + M)));
    printf("Initialized state %d to be size %d, valid %d.\n", i, s[i].size, s[i].valid);
  }

  //Initialize the parameter
  printf("Initializing parameter: %s\n", TEST_FUNC(tnn_param_init(&p)));
  tnn_param_debug(&p);

  //Allocate states in the paramter
  for(i = 0; i < N; i = i + 1){
    printf("Allocating state %d: %s\n", i, TEST_FUNC(tnn_param_state_alloc(&p, &s[i])));
    printf("Allocated state %d to be valid %d, vector owner %d.\n", i, s[i].valid, s[i].x.owner);
    tnn_param_debug(&p);
  }

  //Initialize the values in the vector
  for(i = 0; i < p.x->size; i = i + 1){
    gsl_vector_set(p.x, i, i + A);
    gsl_vector_set(p.dx, i, i + B);
  }
  tnn_param_debug(&p);

  //Initialize values for t using calloc
  printf("Initializing state t: %s\n", TEST_FUNC(tnn_state_init(&t, I)));
  printf("Initialized state t to be size %d, valid %d.\n", t.size, t.valid);
  printf("Allocating state t: %s\n", TEST_FUNC(tnn_param_state_calloc(&p, &t)));
  printf("Allocated state t to be valid %d, vector owner %d.\n", t.valid, t.x.owner);
  tnn_param_debug(&p);

  printf("Initializing state u: %s\n", TEST_FUNC(tnn_state_init(&u, J)));
  printf("Getting subvector u: %s\n", TEST_FUNC(tnn_param_state_sub(&p, &s[N-1], &u, 1)));
  tnn_param_debug(&p);

  //Destroy the parameter
  printf("Destroying the paramter: %s.\n", TEST_FUNC(tnn_param_destroy(&p)));
  printf("Destroyed paramter x = %d, dx = %d, states = %d, size = %d.\n", p.x, p.dx, p.states, p.size);
  for(i = 0; i < N; i = i + 1){
    printf("Destroyed state %d, valid: %d\n", i, s[i].valid);
  }
  printf("Destroyed state t, valid: %d\n", t.valid);
  printf("Destroyed state u, valid: %d\n", u.valid);
  tnn_param_debug(&p);
}
