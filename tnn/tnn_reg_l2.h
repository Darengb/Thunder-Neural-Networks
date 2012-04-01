/* Thunder Neural Networks Regularizer - l2 utility
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The header defines the following functions:
 * tnn_error tnn_reg_init_l2(tnn_reg *r);
 * tnn_error tnn_reg_l_l2(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_d_l2(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_debug_l2(tnn_reg *r);
 * tnn_error tnn_reg_destroy_l2(tnn_reg *r);
 */

#include <tnn/tnn_error.h>
#include <tnn/tnn_reg.h>
#include <gsl/gsl_vector.h>

#ifndef TNN_REG_L2_H
#define TNN_REG_L2_H

tnn_error tnn_reg_init_l2(tnn_reg *r);

tnn_error tnn_reg_l_l2(tnn_reg *r, gsl_vector *w, double *l);

tnn_error tnn_reg_d_l2(tnn_reg *r, gsl_vector *w, gsl_vector *d);

tnn_error tnn_reg_debug_l2(tnn_reg *r);

tnn_error tnn_reg_destroy_l2(tnn_reg *r);

#endif //TNN_REG_L2_H
