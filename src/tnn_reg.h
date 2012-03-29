/* Thunder Neural Network Regularizer Utility
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The header defines the following structures:
 * tnn_reg(tnn_reg_type t, void *c,
 *         TNN_REG_FUNC_L l,
 *         TNN_REG_FUNC_D d,
 *         TNN_REG_FUNC_DEBUG debug)
 *
 * The header defines the following functions:
 * tnn_error tnn_reg_l(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_add_l(tnn_reg *r, gsl_vector *w, double *l);
 * tnn_error tnn_reg_add_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);
 * tnn_error tnn_reg_debug(tnn_reg *r);
 * tnn_error tnn_reg_destroy(tnn_reg *r);
 */

#include <tnn_error.h>
#include <gsl/gsl_vector.h>

#ifndef TNN_REG_H
#define TNN_REG_H

//Regularizer types
typedef enum __ENUM_tnn_reg_type{
  TNN_REG_TYPE_L1, //L1 regularizer
  TNN_REG_TYPE_L2, //L2 regularizer

  TNN_REG_TYPE_SIZE //Size indicator
} tnn_reg_type;

//Function types
struct __STRUCT_tnn_reg;
typedef tnn_error (*TNN_REG_FUNC_L)(struct __STRUCT_tnn_reg *r, gsl_vector *w, double *l);
typedef tnn_error (*TNN_REG_FUNC_D)(struct __STRUCT_tnn_reg *r, gsl_vector *w, gsl_vector *d);
typedef tnn_error (*TNN_REG_FUNC_DEBUG)(struct __STRUCT_tnn_reg *r);
typedef tnn_error (*TNN_REG_FUNC_DESTROY)(struct __STRUCT_tnn_reg *r);

//The structure
typedef struct __STRUCT_tnn_reg{
  //Regularizer type
  tnn_reg_type t;
  //Constant parameters
  void *c;
  //Loss method
  TNN_REG_FUNC_L l;
  //Derivative method
  TNN_REG_FUNC_D d;
  //Debug method
  TNN_REG_FUNC_DEBUG debug;
  //Destroy method
  TNN_REG_FUNC_DESTROY destroy;
} tnn_reg;

//Polymorphically compute the loss of the regularizer
tnn_error tnn_reg_l(tnn_reg *r, gsl_vector *w, double *l);

//Polymorphically compute the derivatives of the regularizer
tnn_error tnn_reg_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);

//Add the loss of the regularizer to the value l
tnn_error tnn_reg_add_l(tnn_reg *r, gsl_vector *w, double *l);

//Add the derivatives of the regularizer to the vector d
tnn_error tnn_reg_add_d(tnn_reg *r, gsl_vector *w, gsl_vector *d);

//Polymorphically debug the regularizer
tnn_error tnn_reg_debug(tnn_reg *r);

//Polymorphically destroy the regularizer
tnn_error tnn_reg_destroy(tnn_reg *r);

#endif //TNN_REG_H
