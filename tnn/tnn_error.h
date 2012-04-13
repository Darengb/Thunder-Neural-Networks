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

  TNN_ERROR_PARAM_VALID, //Valid state to add to param (wrong to do so)
  TNN_ERROR_PARAM_NEXIST, //State does not exist in param (for union and sub operation)

  TNN_ERROR_STATE_INVALID, //Invalid state
  TNN_ERROR_STATE_INCOMP, //Incompatible state

  TNN_ERROR_NUMERIC_INCOMP, //Numerical data incompatible

  TNN_ERROR_MODULE_MISTYPE, //Module type mismatch
  TNN_ERROR_MODULE_FUNCNDEF, //Module function undefined

  TNN_ERROR_LOSS_MISTYPE, //Loss type mismatch
  TNN_ERROR_LOSS_FUNCNDEF, //Loss function undefined

  TNN_ERROR_MACHINE_NOMOD, //No modules in the machine

  TNN_ERROR_TRAINER_CLASS_FUNCNDEF, //Trainer - classification function undefined
  TNN_ERROR_TRAINER_CLASS_MISTYPE, //Trainer - classification type mismatch
  TNN_ERROR_TRAINER_CLASS_NVALIDP, //Trainer - classification invalid input parameters
  
  TNN_ERROR_REG_FUNCNDEF, //Regularizer function undefined
  TNN_ERROR_REG_MISTYPE, //Regularizer type mismatch
  TNN_ERROR_REG_INCOMP, //Regularizer numerical incompatibility

  TNN_ERROR_PSTABLE_EXIST, //State already exist in the table
  TNN_ERROR_PSTABLE_NEXIST, //State does not exist in the table

  TNN_ERROR_SIZE //Size indicator
} tnn_error;

#endif //TNN_ERROR_H
