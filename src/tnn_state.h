/* Thunder Neural Networks State Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header defines the following structure:
 * tnn_state(gsl_vector w, gsl_vector dw, int size, bool valid)
 *
 * This header also declares the following functions:
 * tnn_error tnn_state_init(tnn_state *s, size_t n);
 * tnn_error tnn_state_debug(tnn_state *s);
 */

#include <stddef.h>
#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include <tnn_error.h>

#ifndef TNN_STATE_H
#define TNN_STATE_H

typedef struct __STRUCT_tnn_state{
  //The state values
  gsl_vector x;
  //The state gradients
  gsl_vector dx;
  //The size of this state
  size_t size;
  //Validity flag
  bool valid;

  //utlist support
  struct __STRUCT_tnn_state *next;
  struct __STRUCT_tnn_state *prev;
} tnn_state;

//Initialize the state s into size n and invalid state
//State can only be allocated to tnn_param
tnn_error tnn_state_init(tnn_state *s, size_t n);

//Print debug information
tnn_error tnn_state_debug(tnn_state *s);

#endif //TNN_STATE_H
