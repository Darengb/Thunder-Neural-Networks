/* Thunder Neural Networks Trainer - Classification - NSGD Utility
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/29/2012
 *
 * This header defines the following structure:
 * tnn_trainer_class_nsgd(double lambda, double eta, double epsilon, size_t niter)
 *
 * This header defines the following functions:
 * tnn_error tnn_trainer_class_nsgd_init(tnn_trainer_class *t, size_t ninput, size_t noutput, 
 *                                       double lambda, double eta, dboule epsilon, size_t niter);
 * tnn_error tnn_trainer_class_run(tnn_trainer_class *t, gsl_vector *input, size_t *label);
 * tnn_error tnn_trainer_class_learn(tnn_trainer_class *t, gsl_vector *input, size_t label);
 * tnn_error tnn_trainer_class_try(tnn_trainer_class *t, gsl_vector *input, size_t label, bool* correct);
 * tnn_error tnn_trainer_class_test(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels, double *loss, double *error);
 * tnn_error tnn_trainer_class_train(tnn_trainer_class *t, gsl_matrix *inputs, size_t *labels);
 * tnn_error tnn_trainer_class_debug(tnn_trainer_class *t);
 */
