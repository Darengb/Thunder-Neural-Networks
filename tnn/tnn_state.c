/* Thunder Neural Networks State Utilities Implementation
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header implements the following functions:
 * tnn_error tnn_state_init(tnn_state *s, size_t n);
 * tnn_error tnn_state_debug(tnn_state *s);
 * tnn_error tnn_state_copy(tnn_state *s, tnn_state *t);
 */

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_macro.h>

tnn_error tnn_state_init(tnn_state *s, size_t n){
  s->valid = false;
  s->size = n;
  s->parent = NULL;
  s->offset = 0L;
  return TNN_ERROR_SUCCESS;
}

tnn_error tnn_state_debug(tnn_state *s){
  size_t i;
  printf("state = %p, size = %ld, valid = %c, parent = %p, offset = %ld, prev = %p, next = %p\n",
	 s, s->size, s->valid == true?'T':'F', s->parent, s->offset, s->prev, s->next);
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

tnn_error tnn_state_copy(tnn_state *s, tnn_state *t){
  //Check validity
  if(s->valid != true || t->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Check dimensions
  if(s->size != t->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&s->x, &t->x));
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&s->dx, &t->dx));

  return TNN_ERROR_SUCCESS;
}
