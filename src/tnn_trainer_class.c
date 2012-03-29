/* Thunder Neural Networks Trainer - Classification Utility Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The source implements the following functions:
 * tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label);
 * tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct);
 * tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error);
 * tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 */

#include <string.h>
#include <stdbool.h>
#include <tnn_trainer_class.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

//Polymorphically determine the label of a sample
tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label){
  if(t->run != NULL){
    return (*t->run)(t, input, label);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically learn a sample
tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label){
  if(t->learn != NULL){
    return (*t->learn)(t, input, label);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically try one sample
tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct){
  if(t->try != NULL){
    return (*t->try)(t, input, label, correct);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically test on samples
tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error){
  if(t->test != NULL){
    return (*t->test)(t, inputs, labels, loss, error);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically train on samples
tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels){
  if(t->train != NULL){
    return (*t->train)(t, inputs, labels);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically debug the trainer
tnn_error tnn_trainer_class_debug(tnn_trainer_class *t){
  if(t->debug != NULL){
    return (*t->debug)(t);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

