/* Thunder Neural Networks Loss - Euclidean Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/19/2012
 *
 * This header defines the following functions:
 * tnn_error tnn_loss_init_euclidean(tnn_loss *l, tnn_state *input1, tnn_state *input2, tnn_state *output);
 * tnn_error tnn_loss_bprop_euclidean(tnn_loss *l);
 * tnn_error tnn_loss_fprop_euclidean(tnn_loss *l);
 * tnn_error tnn_loss_randomize_euclidean(tnn_loss *l, double k);
 * tnn_error tnn_loss_destroy_euclidean(tnn_loss *l);
 */

#include <tnn_error.h>
#include <tnn_state.h>
#include <tnn_param.h>
#include <tnn_loss.h>

#ifndef TNN_LOSS_EUCLIDEAN_H
#define TNN_LOSS_EUCLIDEAN_H

//Function definitions
tnn_error tnn_loss_init_euclidean(tnn_loss *l, tnn_state *input1, tnn_state *input2, tnn_state *output);
tnn_error tnn_loss_bprop_euclidean(tnn_loss *l);
tnn_error tnn_loss_fprop_euclidean(tnn_loss *l);
tnn_error tnn_loss_randomize_euclidean(tnn_loss *l, double k);
tnn_error tnn_loss_destroy_euclidean(tnn_loss *l);
tnn_error tnn_loss_debug_euclidean(tnn_loss *l);

#endif //TNN_LOSS_EUCLIDEAN_H
