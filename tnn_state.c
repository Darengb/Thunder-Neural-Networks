/* Thunder Neural Networks State Utilities Implementation
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header implements the following functions:
 * int tnn_state_init(tnn_state *s);
 */

#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include "tnn_state.h"
#include "tnn_error.h"

tnn_error tnn_state_init(tnn_state *s, int n){
  s->valid = false;
  s->size = n;
  return TNN_ERROR_SUCCESS;
}
