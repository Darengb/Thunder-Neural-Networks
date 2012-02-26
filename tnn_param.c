/* Thunder Nueral Networks Parameter Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/20/2012
 *
 * This source implements the following functions:
 * int tnn_param_init(tnn_param *p);
 * int tnn_param_salloc(tnn_param p, tnn_state *s);
 * int tnn_param_destroy(tnn_param p);
 */

#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include "utlist.h"
#include "tnn_state.h"
#include "tnn_param.h"
#include "tnn_error.h"

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
  int i, size;

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
  int i, size;

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
