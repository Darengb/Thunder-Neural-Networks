/* Thunder Neural Networks Error Codes
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/19/2012
 *
 * This header defines error code with a function to return error messages.
 * TNN_ERROR_SUCCESS is always ensured to be 0.
 */

#ifndef TNN_ERROR_H
#define TNN_ERROR_H

typedef enum __ENUM_tnn_error{
  TNN_ERROR_SUCCESS = 0, //General success
  TNN_ERROR_FAILURE, //General failure
  TNN_ERROR_ALLOC, //Memory allocation error
  TNN_ERROR_GSL, //GSL routine error

  TNN_ERROR_STATE_INVALID, //Invalid state

  TNN_ERROR_NUMERIC_INCOMP, //Numerical data incompatible

  TNN_ERROR_MODULE_MISTYPE, //Module type mismatch

  TNN_ERROR_SIZE //Size indicator
} tnn_error;

#endif //TNN_ERROR_H
