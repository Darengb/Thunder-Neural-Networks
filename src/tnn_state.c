/* Thunder Neural Networks State Utilities Implementation
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header implements the following functions:
 * tnn_error tnn_state_init(tnn_state *s, size_t n);
 * tnn_error tnn_state_debug(tnn_state *s);
 */

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <tnn_state.h>
#include <tnn_error.h>

tnn_error tnn_state_init(tnn_state *s, size_t n){
  s->valid = false;
  s->size = n;
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_state_debug(tnn_state *s){
  size_t i;
  printf("state = %p, size = %ld, valid = %c, prev = %p, next = %p\n", s, s->size, s->valid == true?'T':'F', s->prev, s->next);
  if(s->valid == true){
    printf("x:");
    for(i = 0; i < s->size; i = i + 1){
      printf(" %g", gsl_vector_get(&s->x, i));
    }
    printf("\n");
    printf("dx:");
    for(i = 0; i < s->size; i = i + 1){
      printf(" %g", gsl_vector_get(&s->dx, i));
    }
    printf("\n");
  }
  return TNN_ERROR_SUCCESS;
}
