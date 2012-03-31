/* Thunder Neural Networks Trainer - Classification - Naive SGD Utility Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 *
 * This source implements the following functions:
 * tnn_error tnn_trainer_class_init_nsgd(tnn_trainer_class *t, size_t ninput, size_t noutput, gsl_matrix *lset,
 *                                       double lambda, double eta, double epsilon, size_t eiter, size_t niter);
 * tnn_error tnn_trainer_class_learn_nsgd(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_train_nsgd(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 * tnn_error tnn_trainer_class_debug_nsgd(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_destroy_nsgd(tnn_trainer_class *t);
 * tnn_error tnn_trainer_class_titer_nsgd(tnn_trainer_class *t, size_t *titer);
 */

#include <string.h> //For size_t
#include <stdio.h>
#include <float.h>
#include <tnn_macro.h>
#include <tnn_trainer_class.h>
#include <tnn_trainer_class_nsgd.h>
#include <tnn_machine.h>
#include <tnn_loss.h>
#include <tnn_reg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

//Initialize a trainer to be nsgd trainer. The lset is managed by the trainer
tnn_error tnn_trainer_class_init_nsgd(tnn_trainer_class *t, size_t ninput, size_t noutput, gsl_matrix *lset,
                                      double lambda, double eta, double epsilon, size_t eiter, size_t niter){
  tnn_error ret;

  //Check the paramters
  if(lambda < 0 || eta < 0 || epsilon < 0){
    return TNN_ERROR_TRAINER_CLASS_NVALIDP;
  }
  if(eiter < 1){
    eiter = 1;
  }
  if(niter < 1 && epsilon == 0){
    return TNN_ERROR_TRAINER_CLASS_NVALIDP;
  }

  //Defined type
  t->t = TNN_TRAINER_CLASS_TYPE_NSGD;

  //Constant paramters
  t->c = (tnn_trainer_class_nsgd *) malloc(sizeof(tnn_trainer_class_nsgd));
  ((tnn_trainer_class_nsgd*)t->c)->eta = eta;
  ((tnn_trainer_class_nsgd*)t->c)->epsilon = epsilon;
  ((tnn_trainer_class_nsgd*)t->c)->eiter = eiter;
  ((tnn_trainer_class_nsgd*)t->c)->niter = niter;
  ((tnn_trainer_class_nsgd*)t->c)->titer = 0;

  //lset
  t->lset = lset;

  //Losses
  t->losses = gsl_vector_alloc(t->lset->size1);

  //Initialize the machine
  TNN_MACRO_ERRORTEST(tnn_machine_init(&t->m, ninput, noutput),ret);

  //Initialize the label
  TNN_MACRO_ERRORTEST(tnn_state_init(&t->label, noutput),ret);
  TNN_MACRO_ERRORTEST(tnn_machine_state_alloc(&t->m, &t->label),ret);

  //Initialize the regularization parameter
  t->lambda = lambda;

  //Initialize methods
  t->learn = tnn_trainer_class_learn_nsgd;
  t->train = tnn_trainer_class_train_nsgd;
  t->debug = tnn_trainer_class_debug_nsgd;
  t->destroy = tnn_trainer_class_destroy_nsgd;

  return TNN_ERROR_SUCCESS;
}

//Learn one sample using naive stochastic gradient descent
tnn_error tnn_trainer_class_learn_nsgd(tnn_trainer_class *t, gsl_vector *input, size_t label){
  tnn_error ret;
  tnn_state *sin;
  tnn_param *p;
  gsl_vector_view lb;

  //Routine check
  if(t->t != TNN_TRAINER_CLASS_TYPE_NSGD){
    return TNN_ERROR_TRAINER_CLASS_MISTYPE;
  }

  //Check the input and label
  TNN_MACRO_ERRORTEST(tnn_machine_get_sin(&t->m, &sin),ret);
  if(label >= t->lset->size1 || input->size != sin->size){
    return TNN_ERROR_STATE_INCOMP;
  }
  lb = gsl_matrix_row(t->lset, label);

  //Set the loss output dx to be 1
  gsl_vector_set(&t->l.output->dx, 0, 1.0);

  //Copy the data into the input/label and do forward and backward propagation
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(input, &sin->x));
  TNN_MACRO_GSLTEST(gsl_blas_dcopy(&lb.vector, &t->label.x));
  TNN_MACRO_ERRORTEST(tnn_machine_fprop(&t->m), ret);
  TNN_MACRO_ERRORTEST(tnn_loss_fprop(&t->l), ret);
  TNN_MACRO_ERRORTEST(tnn_loss_bprop(&t->l), ret);
  TNN_MACRO_ERRORTEST(tnn_machine_bprop(&t->m), ret);

  //Compute the accumulated regularization paramter
  TNN_MACRO_ERRORTEST(tnn_machine_get_param(&t->m, &p), ret);
  TNN_MACRO_ERRORTEST(tnn_reg_addd(&t->r, p->x, p->dx, t->lambda), ret);

  //Compute the parameter update
  TNN_MACRO_GSLTEST(gsl_blas_daxpy(-((tnn_trainer_class_nsgd*)t->c)->eta, p->dx, p->x));

  //Set the titer parameter
  ((tnn_trainer_class_nsgd*)t->c)->titer = 1;

  return TNN_ERROR_SUCCESS;
}

//Train all the samples using naive stochastic gradient descent
tnn_error tnn_trainer_class_train_nsgd(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels){
  tnn_error ret;
  tnn_state *sin;
  tnn_param *p;
  gsl_vector *rd;
  gsl_vector *pw;
  gsl_vector_view in;
  gsl_vector_view lb;
  double eps;
  size_t i,j;

  //Routine check
  if(t->t != TNN_TRAINER_CLASS_TYPE_NSGD){
    return TNN_ERROR_TRAINER_CLASS_MISTYPE;
  }

  //Check the input
  TNN_MACRO_ERRORTEST(tnn_machine_get_sin(&t->m, &sin),ret);
  if(inputs->size2 != sin->size){
    return TNN_ERROR_STATE_INCOMP;
  }

  //Set the loss output dx to be 1
  gsl_vector_set(&t->l.output->dx, 0, 1.0);

  //Get the parameter and allocate rd and pw
  TNN_MACRO_ERRORTEST(tnn_machine_get_param(&t->m, &p), ret);
  rd = gsl_vector_alloc(p->size);
  pw = gsl_vector_alloc(p->size);
  if(rd == NULL || pw == NULL){
    return TNN_ERROR_GSL;
  }

  //Into the main loop
  for(eps = DBL_MAX, ((tnn_trainer_class_nsgd*)t->c)->titer = 0;
      eps > ((tnn_trainer_class_nsgd*)t->c)->epsilon && ((tnn_trainer_class_nsgd*)t->c)->titer < ((tnn_trainer_class_nsgd*)t->c)->niter;
      ((tnn_trainer_class_nsgd*)t->c)->titer = ((tnn_trainer_class_nsgd*)t->c)->titer + ((tnn_trainer_class_nsgd*)t->c)->eiter){

    //Copy the previous pw
    TNN_MACRO_GSLTEST(gsl_blas_dcopy(p->x, pw));

    for(i = 0; i < ((tnn_trainer_class_nsgd*)t->c)->eiter; i = i + 1){

      j = (((tnn_trainer_class_nsgd*)t->c)->titer + i)%inputs->size1;

      //Check the label
      if(labels[j] >= t->lset->size1){
	return TNN_ERROR_STATE_INCOMP;
      }

      //Get the inputs and label vector
      lb = gsl_matrix_row(t->lset, labels[j]);
      in = gsl_matrix_row(inputs, j);

      //Copy the data into the input/label and do forward and backward propagation
      TNN_MACRO_GSLTEST(gsl_blas_dcopy(&in.vector, &sin->x));
      TNN_MACRO_GSLTEST(gsl_blas_dcopy(&lb.vector, &t->label.x));
      TNN_MACRO_ERRORTEST(tnn_machine_fprop(&t->m), ret);
      TNN_MACRO_ERRORTEST(tnn_loss_fprop(&t->l), ret);
      TNN_MACRO_ERRORTEST(tnn_loss_bprop(&t->l), ret);
      TNN_MACRO_ERRORTEST(tnn_machine_bprop(&t->m), ret);

      //Compute the accumulated regularization paramter
      TNN_MACRO_ERRORTEST(tnn_reg_d(&t->r, p->x, rd), ret);
      TNN_MACRO_GSLTEST(gsl_blas_daxpy(t->lambda, rd, p->dx));

      //Compute the parameter update
      TNN_MACRO_GSLTEST(gsl_blas_daxpy(-((tnn_trainer_class_nsgd*)t->c)->eta, p->dx, p->x));
    }

    //Compute the 2 square norm of difference of p as eps
    TNN_MACRO_GSLTEST(gsl_blas_daxpy(-1.0, p->x, pw));
    eps = gsl_blas_dnrm2(pw);
  }
  
  return TNN_ERROR_SUCCESS;
}

//Debug this trainer
tnn_error tnn_trainer_class_debug_nsgd(tnn_trainer_class *t){
  tnn_error ret;
  size_t i,j;

  //Routine check
  if(t->t != TNN_TRAINER_CLASS_TYPE_NSGD){
    printf("Trainer classifcation (Naive SGD) mistype\n");
    return TNN_ERROR_TRAINER_CLASS_MISTYPE;
  }

  ret = TNN_ERROR_SUCCESS;

  printf("Trainer classification (Naive SGD) = %p, type = %d, constant = %p, label_set = %p, lambda = %g\n", t, t->t, t->c, t->lset, t->lambda);
  printf("losses = %p, learn = %p, train = %p, debug = %p, destroy = %p\n", t->losses, t->learn, t->train, t->debug, t->destroy);
  printf("eta = %g, epsilon = %g, eiter = %ld, niter = %ld, titer = %ld\n",
	 ((tnn_trainer_class_nsgd*)t->c)->eta,
	 ((tnn_trainer_class_nsgd*)t->c)->epsilon,
	 ((tnn_trainer_class_nsgd*)t->c)->eiter,
	 ((tnn_trainer_class_nsgd*)t->c)->niter,
	 ((tnn_trainer_class_nsgd*)t->c)->titer);

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

//Destroy this trainer
tnn_error tnn_trainer_class_destroy_nsgd(tnn_trainer_class *t){

  //Routine check
  if(t->t != TNN_TRAINER_CLASS_TYPE_NSGD){
    return TNN_ERROR_TRAINER_CLASS_MISTYPE;
  }

  //Destroy the parameter
  free((tnn_trainer_class_nsgd*)t->c);

  return TNN_ERROR_SUCCESS;
}

//Get the true number of iterations executed
tnn_error tnn_trainer_class_titer_nsgd(tnn_trainer_class *t, size_t *titer){
  //Routine check
  if(t->t != TNN_TRAINER_CLASS_TYPE_NSGD){
    return TNN_ERROR_TRAINER_CLASS_MISTYPE;
  }
  *titer = ((tnn_trainer_class_nsgd*)t->c)->titer;
  return TNN_ERROR_SUCCESS;
}
