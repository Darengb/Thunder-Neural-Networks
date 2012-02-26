/* Thunder Neural Networks Parameter Utilities Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header defines the following structure:
 * tnn_param(gsl_vector *x, gsl_vector *dx, tnn_state *states, int size)
 *
 * This header defines the following functions:
 * int tnn_param_init(tnn_param *p);
 * int tnn_param_salloc(tnn_param p, tnn_state *s);
 * int tnn_param_destroy(tnn_param p);
 */

#include <gsl/gsl_vector.h>
#include "tnn_state.h"
#include "tnn_error.h"

#ifndef TNN_PARAM_H
#define TNN_PARAM_H

typedef struct __STRUCT_tnn_param{
  gsl_vector *x;
  gsl_vector *dx;
  tnn_state *states;
  int size;
} tnn_param;

//Initialize size to 0, pointers to NULL
tnn_error tnn_param_init(tnn_param *p);

//Allocate a state in s, using s's size.
tnn_error tnn_param_state_alloc(tnn_param *p, tnn_state *s);
tnn_error tnn_param_state_calloc(tnn_param *p, tnn_state *s);

//Destroy the parameter objects
//It set all the states stored in this parameter invalid, free the space of x and dx.
tnn_error tnn_param_destroy(tnn_param *p);

#endif //TNN_PARAM_H
