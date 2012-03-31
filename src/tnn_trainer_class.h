/* Thunder Neural Networks Trainer - Classification Utility Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The header defines the following structure:
 * tnn_trainer_class(tnn_trainer type t, void *c, gsl_matrix *lset, gsl_vector *losses, tnn_machine m, tnn_loss l, tnn_state label,
 *             TNN_TRAINER_CLASS_FUNC_LEARN learn,
 *             TNN_TRAINER_CLASS_FUNC_TRAIN train,
 *             TNN_TRAINER_CLASS_FUNC_DESTROY destroy)
 *
 * The header defines the following functions:
 * tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label, double *loss);
 * tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct);
 * tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error);
 * tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 * tnn_error tnn_trainer_class_destroy(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_debug(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_get_machine(tnn_trainer_class *t, tnn_machine **m);
 * tnn_error tnn_trainer_class_get_loss(tnn_trainer_class *t, tnn_loss **l);
 * tnn_error tnn_trainer_class_get_reg(tnn_trainer_class *t, tnn_reg **r);
 * tnn_error tnn_trainer_class_get_label(tnn_trainer_class *t, tnn_state **s);
 */

#include <string.h> //For size_t
#include <stdbool.h>
#include <tnn_error.h>
#include <tnn_machine.h>
#include <tnn_loss.h>
#include <tnn_reg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#ifndef TNN_TRAINER_CLASS_H
#define TNN_TRAINER_CLASS_H

//Trainer types
typedef enum __ENUM_tnn_trainer_class_type{
  TNN_TRAINER_CLASS_TYPE_NSGD, //Naive stochastic gradient descent classification trainer
  TNN_TRAINER_CLASS_TYPE_TSGD, //Threaded stochastic gradient descent classification trainer

  TNN_TRAINER_TYPE_SIZE //Size indicator (if you want to define your own polymorph-safe trainer, do it above this integer)
} tnn_trainer_class_type;

//Function types
struct __STRUCT_tnn_trainer_class;
typedef tnn_error (*TNN_TRAINER_CLASS_FUNC_LEARN)(struct __STRUCT_tnn_trainer_class *t, gsl_vector *input, size_t label);
typedef tnn_error (*TNN_TRAINER_CLASS_FUNC_TRAIN)(struct __STRUCT_tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
typedef tnn_error (*TNN_TRAINER_CLASS_FUNC_DESTROY)(struct __STRUCT_tnn_trainer_class *t);
typedef tnn_error (*TNN_TRAINER_CLASS_FUNC_DEBUG)(struct __STRUCT_tnn_trainer_class *t);

//The tructure
typedef struct __STRUCT_tnn_trainer_class{
  //Trainer type
  tnn_trainer_class_type t;
  //Constant parameters struct
  void *c;
  //Label set -- data owned by this trainer
  gsl_matrix *lset;
  //Loss vector -- data owned by this trainer
  gsl_vector *losses;
  //Network machine
  tnn_machine m;
  //Network loss
  tnn_loss l;
  //Network label state (to be used in loss and allocated in machine's io)
  tnn_state label;
  //Regularizer
  tnn_reg r;
  //Regularizer constant
  double lambda;
  //Learn method
  TNN_TRAINER_CLASS_FUNC_LEARN learn;
  //Train method
  TNN_TRAINER_CLASS_FUNC_TRAIN train;
  //Debug method
  TNN_TRAINER_CLASS_FUNC_DEBUG debug;
  //Destroy method
  TNN_TRAINER_CLASS_FUNC_DESTROY destroy;
} tnn_trainer_class;

//Determine the label of a sample
tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label, double *loss);

//Polymorphically learn a sample
tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label);

//Try one sample
tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct);

//Test on samples
tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error);

//Polymorphicall train on samples
tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);

//Polymorphically debug the trainer
tnn_error tnn_trainer_class_debug(tnn_trainer_class *t);

//Polymorphically destroy the trainer
tnn_error tnn_trainer_class_destroy(tnn_trainer_class *t);

//Get the address of the machine in the trainer
tnn_error tnn_trainer_class_get_machine(tnn_trainer_class *t, tnn_machine **m);

//Get the address of the loss in the trainer
tnn_error tnn_trainer_class_get_loss(tnn_trainer_class *t, tnn_loss **l);

//Get the address of the regularizer in the trainer
tnn_error tnn_trainer_class_get_reg(tnn_trainer_class *t, tnn_reg **r);

//Get the address of the label
tnn_error tnn_trainer_class_get_label(tnn_trainer_class *t, tnn_state **s);

#endif //TNN_TRAINER_CLASS_H
