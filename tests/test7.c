/* Dummy Test 7 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/31/2012
 *
 * Tests for the following utilities were performed:
 * tnn_trainer_class, tnn_trainer_class_nsgd
 *
 * A 1-layer linear-bias model, with euclidean loss.
 *
 * Results:
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <tnn/tnn_machine.h>
#include <tnn/tnn_loss.h>
#include <tnn/tnn_loss_euclidean.h>
#include <tnn/tnn_reg.h>
#include <tnn/tnn_reg_l2.h>
#include <tnn/tnn_reg_l1.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_module_linear.h>
#include <tnn/tnn_module_bias.h>
#include <tnn/tnn_trainer_class.h>
#include <tnn/tnn_trainer_class_nsgd.h>

#define TEST_FUNC(func) (func==TNN_ERROR_SUCCESS?"YES":"NO")

#define A 10
#define B 2
#define D 5.0
#define L 0.1 //Lambda
#define P 0.95 //Label set values
#define E 0.0001 //eta
#define T 50 //eiter
#define N 500000 //niter
#define S 0.0002 //epsilon
#define Q 20 //Data size

gsl_matrix* label_set(size_t noutput, double p);
gsl_matrix* rdata(size_t size1, size_t size2, double k);
size_t* rlabel(size_t size, size_t class);

int main(){
  tnn_trainer_class t;
  tnn_machine *m;
  tnn_loss *l;
  tnn_reg *r;
  tnn_state *label;
  tnn_module *mout;
  tnn_module *min;
  tnn_param *p;
  tnn_state *sin;
  tnn_state *sout;
  tnn_state *in;
  tnn_state *out;
  gsl_matrix *inputs;
  gsl_vector_view input;
  size_t *labels;
  size_t lb;
  double ls;
  double er;
  bool cr;

  //Initialize the trainer and get every components
  printf("Initializing the trainer: %s\n", TEST_FUNC(tnn_trainer_class_init_nsgd(&t, A, B, label_set(B, P), L, E, S, T, N)));
  printf("Getting the machine of the trainer: %s\n", TEST_FUNC(tnn_trainer_class_get_machine(&t, &m)));
  printf("Getting the loss of the trainer: %s\n", TEST_FUNC(tnn_trainer_class_get_loss(&t, &l)));
  printf("Getting the regularizer of the trainer: %s\n", TEST_FUNC(tnn_trainer_class_get_reg(&t, &r)));
  printf("Getting the label of the trainer: %s\n", TEST_FUNC(tnn_trainer_class_get_label(&t, &label)));

  //Get various things from the initialized machine
  printf("Get machine paramters: %s\n", TEST_FUNC(tnn_machine_get_param(m, &p)));
  printf("Get machine input module: %s\n", TEST_FUNC(tnn_machine_get_min(m, &min)));
  printf("Get machine output module: %s\n", TEST_FUNC(tnn_machine_get_mout(m, &mout)));
  printf("Get machine input state: %s\n", TEST_FUNC(tnn_machine_get_sin(m, &sin)));
  printf("Get machine output state: %s\n", TEST_FUNC(tnn_machine_get_sout(m, &sout)));

  //Allocate modules for the machine and the loss
  out = malloc(sizeof(tnn_state));
  printf("Initializing first hidden state: %s\n", TEST_FUNC(tnn_state_init(out, B)));
  printf("Allocating first hidden state: %s\n", TEST_FUNC(tnn_machine_state_alloc(m, out)));
  printf("Initializing min: %s\n", TEST_FUNC(tnn_module_init_linear(min, sin, out, p)));
  in = out;
  printf("Initializing mout: %s\n", TEST_FUNC(tnn_module_init_bias(mout, in, sout, p)));
  out = malloc(sizeof(tnn_state));
  printf("Initializing loss output: %s\n", TEST_FUNC(tnn_state_init(out, 1)));
  printf("Allocating loss output: %s\n", TEST_FUNC(tnn_machine_state_alloc(m, out)));
  printf("Initializing the loss: %s\n", TEST_FUNC(tnn_loss_init_euclidean(l, sout, label, out)));

  //Initialize the regularization parameter
  printf("Initializing the regularization: %s\n", TEST_FUNC(tnn_reg_init_l1(r)));

  //Randomize the machine
  printf("Randomizing the machine: %s\n", TEST_FUNC(tnn_machine_randomize(m, 1.0)));

  //First debug
  printf("Debugging the trainer: %s\n", TEST_FUNC(tnn_trainer_class_debug(&t)));

  //Generate training data
  inputs = rdata(Q, A, D);
  labels = rlabel(Q, B);

  printf("Data generated.\n");

  //Learn a sample
  input = gsl_matrix_row(inputs, 0);
  printf("Learn a sample: %s\n", TEST_FUNC(tnn_trainer_class_learn(&t, &input.vector, labels[0])));

  //Second debug
  printf("Debugging the trainer: %s\n", TEST_FUNC(tnn_trainer_class_debug(&t)));

  //Train on samples
  printf("Train on samples: %s\n", TEST_FUNC(tnn_trainer_class_train(&t, inputs, labels)));

  //Third debug
  printf("Debugging the trainer: %s\n", TEST_FUNC(tnn_trainer_class_debug(&t)));

  //Test one example
  printf("Run on one example: %s\n", TEST_FUNC(tnn_trainer_class_run(&t, &input.vector, &lb, &ls)));
  printf("Label: %ld, loss: %g\n", lb, ls);
  printf("Try on one example: %s\n", TEST_FUNC(tnn_trainer_class_try(&t, &input.vector, labels[0], &cr)));
  printf("Correct: %s\n", cr?"YES":"NO");
  printf("Test on the data: %s\n", TEST_FUNC(tnn_trainer_class_test(&t, inputs, labels, &ls, &er)));
  printf("Loss: %g, error: %g\n", ls, er);

  //Fourth debug
  printf("Debugging the trainer: %s\n", TEST_FUNC(tnn_trainer_class_debug(&t)));

  free(labels);
  gsl_matrix_free(inputs);
  return 0;
}

gsl_matrix* label_set(size_t noutput, double p){
  gsl_matrix *ret;
  int i,j;
  ret = gsl_matrix_alloc(noutput, noutput);
  for(i = 0; i < noutput; i = i + 1){
    for(j = 0; j < noutput; j = j + 1){
      gsl_matrix_set(ret, i, j, i==j?p:(1.0-p));
    }
  }
  return ret;
}

gsl_matrix* rdata(size_t size1, size_t size2, double k){
  gsl_matrix *ret;
  size_t i,j;
  srand(time(NULL));
  ret = gsl_matrix_alloc(size1, size2);
  for(i = 0; i < size1; i = i + 1){
    for(j = 0; j < size2; j = j + 1){
      gsl_matrix_set(ret,i,j,2.0*k*((double)rand()/(double)RAND_MAX) - k);
    }
  }

  printf("Training data generated:\n");
  for(i = 0; i < size1; i = i + 1){
    for(j = 0; j < size2; j = j + 1){
      printf(" %g", gsl_matrix_get(ret,i,j));
    }
    printf("\n");
  }

  return ret;
}

size_t* rlabel(size_t size, size_t class){
  size_t *ret;
  size_t i;
  srand(time(NULL));
  ret = (size_t *)malloc(sizeof(size_t)*size);
  for(i = 0; i < size; i = i + 1){
    ret[i] = rand()%class;
  }

  printf("Training label generated:");
  for(i = 0; i < size; i = i + 1){
    printf(" %ld", ret[i]);
  }
  printf("\n");

  return ret;
}
