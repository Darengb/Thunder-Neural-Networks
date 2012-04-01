/* Thunder Neural Networks Trainer - Classification - Naive SGD Utility Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * This header defines the following structure:
 * tnn_trainer_class_nsgd(double eta, double epsilon, size_t eiter, size_t niter, size_t titer)
 *
 * This header defines the following functions:
 * tnn_error tnn_trainer_class_init_nsgd(tnn_trainer_class *t, size_t ninput, size_t noutput, gsl_matrix *lset,
 *                                       double lambda, double eta, double epsilon, size_t eiter, size_t niter);
 * tnn_error tnn_trainer_class_learn_nsgd(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_train_nsgd(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 * tnn_error tnn_trainer_class_debug_nsgd(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_destroy_nsgd(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_titer_nsgd(tnn_trainer_class *t, size_t *titer);
 */

#include <stddef.h> //For size_t
#include <tnn/tnn_trainer_class.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#ifndef TNN_TRAINER_CLASS_NSGD_H
#define TNN_TRAINER_CLASS_NSGD_H

//The training parameters
typedef struct __STRUCT_tnn_trainer_class_nsgd{
  double eta; //Step size
  double epsilon; //Exit accuracy criterion: 0 if not used
  size_t eiter; //For speed, after eiter steps we test whether to exit
  size_t niter; //Exit step size criterion
  size_t titer; //True steps executed
} tnn_trainer_class_nsgd;

//Initialize a trainer to be nsgd trainer. The lset is managed by the trainer
tnn_error tnn_trainer_class_init_nsgd(tnn_trainer_class *t, size_t ninput, size_t noutput, gsl_matrix *lset,
                                      double lambda, double eta, double epsilon, size_t eiter, size_t niter);

//Learn one sample using naive stochastic gradient descent
tnn_error tnn_trainer_class_learn_nsgd(tnn_trainer_class *t, gsl_vector *input, size_t label);

//Train all the samples using naive stochastic gradient descent
tnn_error tnn_trainer_class_train_nsgd(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);

//Debug this trainer
tnn_error tnn_trainer_class_debug_nsgd(tnn_trainer_class *t);

//Destroy this trainer
tnn_error tnn_trainer_class_destroy_nsgd(tnn_trainer_class *t);

//Get the true number of iterations executed
tnn_error tnn_trainer_class_titer_nsgd(tnn_trainer_class *t, size_t *titer);

#endif //TNN_TRAINER_CLASS_NSGD_H
