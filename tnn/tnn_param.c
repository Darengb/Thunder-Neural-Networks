/* Thunder Nueral Networks Parameter Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/20/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_param_init(tnn_param *p);
 * tnn_error tnn_param_state_alloc(tnn_param *p, tnn_state *s);
 * tnn_error tnn_param_state_calloc(tnn_param *p, tnn_state *s);
 * tnn_error tnn_param_destroy(tnn_param p);
 * tnn_error tnn_param_state_sub(tnn_param *p, tnn_state *s, tnn_state *t, size_t offset);
 * tnn_error tnn_param_debug(tnn_param *p);
 */

#include <stddef.h>
#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include <tnn/utlist.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_error.h>

//Initialize size to 0, pointers to NULL
tnn_error tnn_param_init(tnn_param *p){
  p->x = NULL;
  p->dx = NULL;
  p->states = NULL;
  p->size = 0;
  return TNN_ERROR_SUCCESS;
}

//Allocate a state in s, using s's size.
tnn_error tnn_param_state_alloc(tnn_param *p, tnn_state *s){
  gsl_vector *x;
  gsl_vector *dx;
  gsl_vector_view xv;
  gsl_vector_view dxv;
  tnn_state *elt;
  tnn_state *tmp;
  size_t i, size;

  //Routine check
  if(s->valid == true){
    return TNN_ERROR_PARAM_VALID;
  }

  //Allocate new vectors
  size = p->size + s->size;
  x = gsl_vector_alloc(size);
  dx = gsl_vector_alloc(size);

  if(x == NULL || dx == NULL){
    return TNN_ERROR_ALLOC;
  }

  //Copy to original vectors
  if(p->size > 0){
    xv = gsl_vector_subvector(x, 0, p->size);
    dxv = gsl_vector_subvector(dx, 0, p->size);
    gsl_vector_memcpy(&xv.vector, p->x);
    gsl_vector_memcpy(&dxv.vector, p->dx);
    gsl_vector_free(p->x);
    gsl_vector_free(p->dx);
  }
  p->x = x;
  p->dx = dx;
  p->size = size;

  //Add this state to the list
  DL_APPEND(p->states, s);

  //Renew the information stored in all lists
  i = 0;
  DL_FOREACH_SAFE(p->states, elt, tmp){
    xv = gsl_vector_subvector(p->x, i, elt->size);
    dxv = gsl_vector_subvector(p->dx, i, elt->size);
    elt->x = xv.vector;
    elt->dx = dxv.vector;
    elt->valid = true;
    i = i + elt->size;
  }

  return TNN_ERROR_SUCCESS;
}

//Allocate a state in s, using s's size.
tnn_error tnn_param_state_calloc(tnn_param *p, tnn_state *s){
  gsl_vector *x;
  gsl_vector *dx;
  gsl_vector_view xv;
  gsl_vector_view dxv;
  tnn_state *elt;
  tnn_state *tmp;
  size_t i, size;

  //Routine check
  if(s->valid == true){
    return TNN_ERROR_PARAM_VALID;
  }

  //Allocate new vectors
  size = p->size + s->size;
  x = gsl_vector_calloc(size);
  dx = gsl_vector_calloc(size);

  if(x == NULL || dx == NULL){
    return TNN_ERROR_ALLOC;
  }

  //Copy to original vectors
  if(p->size > 0){
    xv = gsl_vector_subvector(x, 0, p->size);
    dxv = gsl_vector_subvector(dx, 0, p->size);
    gsl_vector_memcpy(&xv.vector, p->x);
    gsl_vector_memcpy(&dxv.vector, p->dx);
    gsl_vector_free(p->x);
    gsl_vector_free(p->dx);
  }
  p->x = x;
  p->dx = dx;
  p->size = size;

  //Add this state to the list
  DL_APPEND(p->states, s);

  //Renew the information stored in all lists
  i = 0;
  DL_FOREACH_SAFE(p->states, elt, tmp){
    xv = gsl_vector_subvector(p->x, i, elt->size);
    dxv = gsl_vector_subvector(p->dx, i, elt->size);
    elt->x = xv.vector;
    elt->dx = dxv.vector;
    elt->valid = true;
    i = i + elt->size;
  }

  return TNN_ERROR_SUCCESS;
}

//Destroy the parameter object
//It sets all the states stored in this parameter invalid, free the space of x and dx
tnn_error tnn_param_destroy(tnn_param *p){
  tnn_state *elt;
  tnn_state *tmp;

  if(p->size > 0){
    //Set all of the states to invalid
    DL_FOREACH_SAFE(p->states, elt, tmp){
      elt->valid = false;
    }
    p->states = NULL;
    //Free the vectors
    gsl_vector_free(p->x);
    gsl_vector_free(p->dx);
    p->x = NULL;
    p->dx = NULL;
    //Set the size to be 0
    p->size = 0;
  }

  return TNN_ERROR_SUCCESS;
}

//Get sub state vectors, using t's size.
tnn_error tnn_param_state_sub(tnn_param *p, tnn_state *s, tnn_state *t, size_t offset){
  gsl_vector_view xv;
  gsl_vector_view dxv;
  tnn_state *elt, *tmp;
  bool found;

  //Routine check
  if(t->valid == true){
    return TNN_ERROR_PARAM_VALID;
  }

  //Check whether s is in the list
  found = false;
  DL_FOREACH_SAFE(p->states, elt, tmp){
    if(elt == s){
      found = true;
    }
  }
  if(found == false){
    return TNN_ERROR_PARAM_NEXIST;
  }

  //Add this state to the list
  DL_APPEND(p->states, t);

  //Renew the information in state t
  xv = gsl_vector_subvector(&s->x, offset, t->size);
  dxv = gsl_vector_subvector(&s->dx, offset, t->size);
  t->x = xv.vector;
  t->dx = dxv.vector;
  t->valid = true;

  return TNN_ERROR_SUCCESS;
}

//Debug info from paramters
tnn_error tnn_param_debug(tnn_param *p){
  size_t i;
  tnn_state *elt, *tmp;
  printf("paramter = %p, size = %ld, x = %p, dx = %p, states = %p\n", p, p->size, p->x, p->dx, p->states);
  if(p->size > 0){
    printf("x:");
    for(i = 0; i < p->size; i = i + 1){
      printf(" %g", gsl_vector_get(p->x, i));
    }
    printf("\n");
    printf("dx:");
    for(i = 0; i < p->size; i = i + 1){
      printf(" %g", gsl_vector_get(p->dx, i));
    }
    printf("\n");
    //Debug for each states
    DL_FOREACH_SAFE(p->states, elt, tmp){
      tnn_state_debug(elt);
    }
  }
  return TNN_ERROR_SUCCESS;
}
