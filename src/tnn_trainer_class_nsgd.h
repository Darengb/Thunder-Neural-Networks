/* Thunder Neural Networks Trainer - Classification - NSGD Utility
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * This header defines the following structure:
 * tnn_trainer_class_nsgd(double eta, double epsilon, size_t niter)
 *
 * This header defines the following functions:
 * tnn_error tnn_trainer_class_init_nsgd(tnn_trainer_class *t, size_t ninput, size_t noutput, gsl_matrix *lset,
 *                                       double lambda, double eta, dboule epsilon, size_t niter);
 * tnn_error tnn_trainer_class_learn_nsgd(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_train_nsgd(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 * tnn_error tnn_trainer_class_debug_nsgd(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_destroy_nsgd(tnn_trainer_class *t);
 */

#include <tnn_trainer_class.h>
#include <gsl/gsl_vector>
#include <gsl/gsl_matrix>

#ifndef TNN_TRAINER_CLASS_NSGD_H
#define TNN_TRAINER_CLASS_NSGD_H

#endif //TNN_TRAINER_CLASS_NSGD_H
