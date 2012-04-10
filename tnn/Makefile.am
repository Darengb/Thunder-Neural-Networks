lib_LTLIBRARIES = libtnn.la

pkginclude_HEADERS = tnn_error.h tnn_config.h tnn_loss_euclidean.h tnn_loss.h tnn_machine.h tnn_macro.h tnn_module_bias.h tnn_module.h tnn_module_linear.h tnn_numeric.h tnn_param.h tnn_reg.h tnn_reg_l1.h tnn_reg_l2.h tnn_state.h tnn_trainer_class.h tnn_trainer_class_nsgd.h uthash.h utlist.h utarray.h tnn_module_sum.h

libtnn_la_SOURCES = tnn_loss.c tnn_machine.c tnn_module.c tnn_numeric.c tnn_reg.c tnn_reg_l2.c tnn_trainer_class.c tnn_loss_euclidean.c tnn_module_bias.c tnn_module_linear.c tnn_param.c tnn_reg_l1.c tnn_state.c tnn_trainer_class_nsgd.c tnn_module_sum.c

libtnn_la_CFLAGS = -I$(top_srcdir)

libtnn_la_LDFLAGS = -version-info $(TNN_LT_VERSION)