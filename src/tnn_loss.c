/* Thunder Neural Networks Loss Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 02/26/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_loss_bprop(tnn_loss *l);
 * tnn_error tnn_loss_fprop(tnn_loss *l);
 * tnn_error tnn_loss_randomize(tnn_loss *l, double k);
 * tnn_error tnn_loss_destroy(tnn_loss *m);
 * tnn_error tnn_loss_debug(tnn_loss *l);
 */

#include <tnn_error.h>
#include <tnn_state.h>
#include <tnn_loss.h>

//Polymorphic back-propagation method
tnn_error tnn_loss_bprop(tnn_loss *l){
  if (l->bprop != NULL){
    return (*l->bprop)(l);
  }
  return TNN_ERROR_LOSS_FUNCNDEF;
}

//Polymorphic forward-propagation method
tnn_error tnn_loss_fprop(tnn_loss *l){
  if(l->fprop != NULL){
    return (*l->fprop)(l);
  }
  return TNN_ERROR_LOSS_FUNCNDEF;
}

//Polymorphic randomize method
tnn_error tnn_loss_randomize(tnn_loss *l, double k){
  if(l->randomize != NULL){
    return (*l->randomize)(l, k);
  }
  return TNN_ERROR_LOSS_FUNCNDEF;
}

//Polymorphic destroy method
tnn_error tnn_loss_destroy(tnn_loss *l){
  if(l->destroy != NULL){
    return (*l->destroy)(l);
  }
  return TNN_ERROR_LOSS_FUNCNDEF;
}

//Polymorphic debug method
tnn_error tnn_loss_debug(tnn_loss *l){
  tnn_error ret;
  if(l->debug != NULL){
    printf("polymprhic loss: ");
    return (*l->debug)(l);
  }
  printf("polymorphic loss: ");
  printf("loss (unknown) = %p, type = %d, const = %p\n", l, l->t, l->c);
  printf("bprop = %p, fprop = %p, randomize = %p, destroy = %p, debug = %p\n", l->bprop, l->fprop, l->randomize, l-> destroy, l->debug);
  printf("paramters: ");
  if((ret = tnn_state_debug(&l->w))!=TNN_ERROR_SUCCESS){
    printf("loss (unknown) paramter state debug error\n");
    return ret;
  }
  printf("input 1: ");
  if((ret = tnn_state_debug(l->input1)) != TNN_ERROR_SUCCESS){
    printf("loss (unknown) input1 state debug error\n");
    return ret;
  }
  printf("input 2: ");
  if((ret = tnn_state_debug(l->input2)) != TNN_ERROR_SUCCESS){
    printf("loss (unknown) input2 state debug error\n");
    return ret;
  }
  printf("output: ");
  if((ret = tnn_state_debug(l->output)) != TNN_ERROR_SUCCESS){
    printf("loss (unknown) output state debug error\n");
    return ret;
  }

  return TNN_ERROR_LOSS_FUNCNDEF;
}
