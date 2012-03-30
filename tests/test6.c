/* Dummy Test 6 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * Tests for the following utilities were performed:
 * tnn_reg, tnn_reg_l1, tnn_reg_l2
 *
 * Results:
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_reg.h>
#include <tnn/tnn_reg_l1.h>
#include <tnn/tnn_reg_l2.h>

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

#define A 10
#define B 5

void printVector(gsl_vector *v);

int main(){
  double l;
  tnn_reg l1;
  tnn_reg l2;
  gsl_vector *w;
  gsl_vector *d;
  int i;

  //Initializing regularizers
  printf("Initializing l1 regularizer: %s\n", TEST_FUNC(tnn_reg_init_l1(&l1)));
  printf("Initializing l2 regularizer: %s\n", TEST_FUNC(tnn_reg_init_l2(&l2)));
  printf("Debugging l1 regularizer: %s\n", TEST_FUNC(tnn_reg_debug(&l1)));
  printf("Debugging l2 regularizer: %s\n", TEST_FUNC(tnn_reg_debug(&l2)));

  //Allocating vectors
  printf("Allocating vectors\n");
  w = gsl_vector_alloc(A);
  d = gsl_vector_alloc(A);
  for(i = 0; i < w->size; i = i + 1){
    gsl_vector_set(w, i, (double)(B - i));
  }
  printf("w:");
  printVector(w);
  printf("d:");
  printVector(d);

  printf("Test l1 loss: %s\n", TEST_FUNC(tnn_reg_l(&l1, w, &l)));
  printf("%g\n", l);
  printf("Test l2 loss: %s\n", TEST_FUNC(tnn_reg_l(&l2, w, &l)));
  printf("%g\n", l);

  printf("TEST l1 derivatives: %s\n", TEST_FUNC(tnn_reg_d(&l1, w, d)));
  printVector(d);
  printf("TEST l2 derivatives: %s\n", TEST_FUNC(tnn_reg_d(&l2, w, d)));
  printVector(d);

  printf("Test l1 add loss: %s\n", TEST_FUNC(tnn_reg_addl(&l1, w, &l)));
  printf("%g\n", l);
  printf("Test l2 add loss: %s\n", TEST_FUNC(tnn_reg_addl(&l2, w, &l)));
  printf("%g\n", l);

  printf("Test l1 add derivatives: %s\n", TEST_FUNC(tnn_reg_addd(&l1, w, d)));
  printVector(d);
  printf("Test l2 add derivatives: %s\n", TEST_FUNC(tnn_reg_addd(&l2, w, d)));
  printVector(d);

  printf("Destroying l1 regularizer: %s\n", TEST_FUNC(tnn_reg_destroy(&l1)));
  printf("Destroying l2 regularizer: %s\n", TEST_FUNC(tnn_reg_destroy(&l2)));

  return 0;
}

void printVector(gsl_vector *v){
  size_t i;
  for(i = 0; i < v->size; i = i + 1){
    printf(" %g", gsl_vector_get(v,i));
  }
  printf("\n");
}
