/* Thunder Neural Networks Parameter Utilities Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header defines the following structure:
 * tnn_param(gsl_vector *x, gsl_vector *dx, tnn_state *states, size_t size)
 *
 * This header defines the following functions:
 * tnn_error tnn_param_init(tnn_param *p);
 * tnn_error tnn_param_state_alloc(tnn_param *p, tnn_state *s);
 * tnn_error tnn_param_state_calloc(tnn_param *p, tnn_state *s);
 * tnn_error tnn_param_destroy(tnn_param p);
 * tnn_error tnn_param_state_sub(tnn_param *p, tnn_state *s, tnn_state *t, size_t offset);
 * tnn_error tnn_param_debug(tnn_param *p);
 */

#include <stddef.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_error.h>

#ifndef TNN_PARAM_H
#define TNN_PARAM_H

typedef struct __STRUCT_tnn_param{
  gsl_vector *x;
  gsl_vector *dx;
  tnn_state *states;
  size_t size;
} tnn_param;

//Initialize size to 0, pointers to NULL
tnn_error tnn_param_init(tnn_param *p);

//Allocate a state in s, using s's size.
tnn_error tnn_param_state_alloc(tnn_param *p, tnn_state *s);
tnn_error tnn_param_state_calloc(tnn_param *p, tnn_state *s);

//Destroy the parameter objects
//It set all the states stored in this parameter invalid, free the space of x and dx.
tnn_error tnn_param_destroy(tnn_param *p);

//Get sub state vectors, using t's size.
tnn_error tnn_param_state_sub(tnn_param *p, tnn_state *s, tnn_state *t, size_t offset);

//Debug info from paramters
tnn_error tnn_param_debug(tnn_param *p);

#endif //TNN_PARAM_H
