/* Thunder Neural Networks Trainer - Classification Utility Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * The source implements the following functions:
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

#include <string.h>
#include <stdbool.h>
#include <tnn_trainer_class.h>
#include <tnn_machine.h>
#include <tnn_loss.h>
#include <tnn_state.h>
#include <tnn_reg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

//Determine the label of a sample
tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label, double* loss){
  tnn_error ret;
  tnn_state *sin;
  gsl_vector_view v;
  int i;

  //Get the machine's input state
  if((ret = tnn_machine_get_sin(&t->m, &sin)) != TNN_ERROR_SUCCESS){
    return ret;
  }

  //Check compatibility of state
  if(sin->size != input->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Check validity of states
  if(sin->valid != true){
    return TNN_ERROR_STATE_INVALID;
  }
  if(t->label.valid != true){
    return TNN_ERROR_STATE_INVALID;
  }

  //Copy input to sin
  if(gsl_blas_dcopy(input, &sin->x) != 0){
    return TNN_ERROR_GSL;
  }

  //Do forward propagation
  if((ret = tnn_machine_fprop(&t->m)) != TNN_ERROR_SUCCESS){
    return ret;
  }

  //Copy each lset to each label, and do forward propagation of loss
  for(i = 0; i < t->lset->size1; i = i + 1){
    v = gsl_matrix_row(t->lset, i);
    if(gsl_blas_dcopy(&v.vector, &t->label.x) != 0){
      return TNN_ERROR_GSL;
    }
    if((ret = tnn_loss_fprop(&t->l)) != TNN_ERROR_SUCCESS){
      return ret;
    }
    gsl_vector_set(t->losses, i, -gsl_vector_get(&t->l.output->x, 0));
  }

  //Find the label with the smallest loss
  *label = gsl_blas_idamax(t->losses);
  *loss = -gsl_vector_get(t->losses, *label);

  //Recover the loss to positive
  gsl_blas_dscal(-1.0, t->losses);

  return TNN_ERROR_SUCCESS;
}

//Polymorphically learn a sample
tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label){
  if(t->learn != NULL){
    return (*t->learn)(t, input, label);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Try one sample
tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct){
  size_t lb;
  double ls;
  tnn_error ret;
  if((ret = tnn_trainer_class_run(t, input, &lb, &ls)) != TNN_ERROR_SUCCESS){
    return ret;
  }
  *correct = (lb == label);
  return TNN_ERROR_SUCCESS;
}

//Test on samples
tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error){
  gsl_vector_view input;
  size_t lb;
  double ls;
  tnn_error ret;
  int i;

  *loss = 0;
  *error = 0;
  for(i = 0; i < inputs->size1; i = i + 1){
    input = gsl_matrix_row(inputs, i);
    if((ret = tnn_trainer_class_run(t, &input.vector, &lb, &ls)) != TNN_ERROR_SUCCESS){
      return ret;
    }
    if(lb == labels[i]){
      *error = *error + 1.0;
    }
    *loss = *loss + ls;
  }
  if(inputs->size1 != 0){
    *error = (*error)/(double)inputs->size1;
  }

  return TNN_ERROR_SUCCESS;
}

//Polymorphically train on samples
tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels){
  if(t->train != NULL){
    return (*t->train)(t, inputs, labels);
  }
  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically destroy the trainer
tnn_error tnn_trainer_class_destroy(tnn_trainer_class *t){
  tnn_error ret;

  //Destroy the label set and losses
  gsl_matrix_free(t->lset);
  gsl_vector_free(t->losses);

  //Destroy machine (along with label)
  if((ret = tnn_machine_destroy(&t->m)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_MODULE_FUNCNDEF){
    return ret;
  }

  //Destroy loss
  if((ret = tnn_loss_destroy(&t->l)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_LOSS_FUNCNDEF){
    return ret;
  }

  //Destroy regularizer
  if((ret = tnn_reg_destroy(&t->r)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_REG_FUNCNDEF){
    return ret;
  }

  //Polymorphic call
  if(t->destroy != NULL){
    return (*t->destroy)(t);
  }

  return TNN_ERROR_TRAINER_CLASS_FUNCNDEF;
}

//Polymorphically debug the trainer
tnn_error tnn_trainer_class_debug(tnn_trainer_class *t){
  tnn_error ret;
  size_t i,j;

  if(t->debug != NULL){
    return (*t->debug)(t);
  }

  ret = TNN_ERROR_TRAINER_CLASS_FUNCNDEF;

  printf("Trainer classification (unknown) = %p, type = %d, constant = %p, label_set = %p, lambda = %g\n", t, t->t, t->c, t->lset, t->lambda);
  printf("losses = %p, learn = %p, train = %p, debug = %p, destroy = %p\n", t->losses, t->learn, t->train, t->debug, t->destroy);

  printf("machine: ");
  if((ret = tnn_machine_debug(&t->m)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_MODULE_FUNCNDEF){
    printf("machine debug error in trainer classsification\n");
    return ret;
  }

  printf("loss: ");
  if((ret = tnn_loss_debug(&t->l)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_LOSS_FUNCNDEF){
    printf("loss debug error in trainer classsification\n");
    return ret;
  }

  printf("label: ");
  if((ret = tnn_state_debug(&t->label)) != TNN_ERROR_SUCCESS){
    printf("label state debug error in trainer classification\n");
    return ret;
  }

  printf("regularizer: ");
  if((ret = tnn_reg_debug(&t->r)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_REG_FUNCNDEF){
    printf("regularizer debug error in classification\n");
    return ret;
  }

  printf("label_set: size1 = %ld, size2 = %ld\n", t->lset->size1, t->lset->size2);
  for(i = 0; i < t->lset->size1; i = i + 1){
    printf("%ld:", i);
    for(j = 0; j < t->lset->size2; j = j + 1){
      printf(" %g", gsl_matrix_get(t->lset, i, j));
    }
    printf("\n");
  }

  printf("losses: size = %ld, values:", t->losses->size);
  for(i = 0; i < t->losses->size; i = i + 1){
    printf(" %g", gsl_vector_get(t->losses, i));
  }
  printf("\n");

  return ret;
}

//Get the address of the machine in the trainer                                                                                                     
tnn_error tnn_trainer_class_get_machine(tnn_trainer_class *t, tnn_machine **m){
  *m = &t->m;
  return TNN_ERROR_SUCCESS;
}

//Get the address of the loss in the trainer                                                                                                        
tnn_error tnn_trainer_class_get_loss(tnn_trainer_class *t, tnn_loss **l){
  *l = &t->l;
  return TNN_ERROR_SUCCESS;
}

//Get the address of the regularizer in the trainer
tnn_error tnn_trainer_class_get_reg(tnn_trainer_class *t, tnn_reg **r){
  *r = &t->r;
  return TNN_ERROR_SUCCESS;
}

//Get the address of the label
tnn_error tnn_trainer_class_get_label(tnn_trainer_class *t, tnn_state **s){
  *s = &t->label;
  return TNN_ERROR_SUCCESS;
}
