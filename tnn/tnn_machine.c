/* Thunder Neural Networks Machine Utilities Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/25/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_machine_init(tnn_machine *m, size_t ninput, size_t noutput);
 * tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param **p);
 * tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param **io);
 * tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s);
 * tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state **s);
 * tnn_error tnn_machine_get_sout(tnn_machine *m, tnn_state **s);
 * tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod);
 * tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module **mod);
 * tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module **mod);
 * tnn_error tnn_machine_bprop(tnn_machine *m);
 * tnn_error tnn_machine_fprop(tnn_machine *m);
 * tnn_error tnn_machine_randomize(tnn_machine *m, double k);
 * tnn_error tnn_machine_destroy(tnn_machine *m);
 * tnn_error tnn_machine_debug(tnn_machine *m);
 */

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_macro.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_param.h>
#include <tnn/tnn_module.h>
#include <tnn/tnn_machine.h>
#include <tnn/utlist.h>

//Initialize the machine with designated input and output size
tnn_error tnn_machine_init(tnn_machine *m, size_t ninput, size_t noutput){
  tnn_error ret;

  m->sin = malloc(sizeof(tnn_state));
  m->sout = malloc(sizeof(tnn_state));
  if(m->sin == NULL || m->sout == NULL){
    return TNN_ERROR_ALLOC;
  }

  //Initialize input and output
  TNN_MACRO_ERRORTEST(tnn_state_init(m->sin, ninput),ret);
  TNN_MACRO_ERRORTEST(tnn_state_init(m->sout, noutput),ret);

  //Initialize io and p
  TNN_MACRO_ERRORTEST(tnn_param_init(&m->io),ret);
  TNN_MACRO_ERRORTEST(tnn_param_init(&m->p),ret);

  //Initialize the module lists
  m->m = NULL;

  //Allocate input and output
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(&m->io, m->sin),ret);
  TNN_MACRO_ERRORTEST(tnn_param_state_alloc(&m->io, m->sout),ret);
  return TNN_ERROR_SUCCESS;
}

//Get the parameter of this machine
tnn_error tnn_machine_get_param(tnn_machine *m, tnn_param **p){
  *p = &m->p;
  return TNN_ERROR_SUCCESS;
}

//Get the io paramter of this machine
tnn_error tnn_machine_get_io(tnn_machine *m, tnn_param **io){
  *io = &m->io;
  return TNN_ERROR_SUCCESS;
}

//Allocate state in io for this machine
tnn_error tnn_machine_state_alloc(tnn_machine *m, tnn_state *s){
  return tnn_param_state_alloc(&m->io, s);
}

//Allocate state in io for this machine, initialized to 0
tnn_error tnn_machine_state_calloc(tnn_machine *m, tnn_state *s){
  return tnn_param_state_calloc(&m->io, s);
}

//Get the input state of this machine
tnn_error tnn_machine_get_sin(tnn_machine *m, tnn_state **s){
  *s = m->sin;
  return TNN_ERROR_SUCCESS;
}

//Get the output state of this machine
tnn_error tnn_machine_get_sout(tnn_machine *m, tnn_state **s){
  *s = m->sout;
  return TNN_ERROR_SUCCESS;
}

//Append a module to the machine
tnn_error tnn_machine_module_append(tnn_machine *m, tnn_module *mod){
  DL_APPEND(m->m, mod);
  return TNN_ERROR_SUCCESS;
}

//Prepend a module to the machine
tnn_error tnn_machine_module_prepend(tnn_machine *m, tnn_module *mod){
  DL_PREPEND(m->m, mod);
  return TNN_ERROR_SUCCESS;
}

//Get the input module of this machine
tnn_error tnn_machine_get_min(tnn_machine *m, tnn_module **mod){
  *mod = &m->min;
  return TNN_ERROR_SUCCESS;
}

//Get the output module of this machine
tnn_error tnn_machine_get_mout(tnn_machine *m, tnn_module **mod){
  *mod = &m->mout;
  return TNN_ERROR_SUCCESS;
}

//Run back propagation backward with respect to modules
tnn_error tnn_machine_bprop(tnn_machine *m){
  tnn_module *mod;
  tnn_error ret;

  //backward from mout
  TNN_MACRO_ERRORTEST(tnn_module_bprop(&m->mout),ret);

  //backward propagation
  DL_FOREACH_BACKWARD(m->m, mod){
    TNN_MACRO_ERRORTEST(tnn_module_bprop(mod), ret);
  }

  //backward from min
  TNN_MACRO_ERRORTEST(tnn_module_bprop(&m->min), ret);

  return TNN_ERROR_SUCCESS;
}

//Run forward propagation forward with respect to modules
tnn_error tnn_machine_fprop(tnn_machine *m){
  tnn_module *mod;
  tnn_error ret;

  //forward from min
  TNN_MACRO_ERRORTEST(tnn_module_fprop(&m->min), ret);

  //forward propagation
  DL_FOREACH(m->m, mod){
    TNN_MACRO_ERRORTEST(tnn_module_fprop(mod), ret);
  }

  //forward from mout
  TNN_MACRO_ERRORTEST(tnn_module_fprop(&m->mout),ret);

  return TNN_ERROR_SUCCESS;
}

//Run randomize for all of the modules
tnn_error tnn_machine_randomize(tnn_machine *m, double k){
  tnn_module *mod;
  tnn_error ret;
  bool ndef;

  ndef = false;

  //randomize from min
  if((ret = tnn_module_randomize(&m->min, k)) != TNN_ERROR_SUCCESS){
    if(ret == TNN_ERROR_MODULE_FUNCNDEF){
      ndef = true;
    } else {
      return ret;
    }
  }

  //randomize
  DL_FOREACH(m->m, mod){
    if((ret = tnn_module_randomize(mod, k)) != TNN_ERROR_SUCCESS){
      if(ret == TNN_ERROR_MODULE_FUNCNDEF){
	ndef = true;
      } else {
	return ret;
      }
    }
  }

  //randomize from mout
  if((ret = tnn_module_randomize(&m->mout, k)) != TNN_ERROR_SUCCESS){
    if(ret == TNN_ERROR_MODULE_FUNCNDEF){
      ndef = true;
    } else {
      return ret;
    }
  }
  
  //Function non-def detected?
  if (ndef){
    return TNN_ERROR_MODULE_FUNCNDEF;
  } else {
    return TNN_ERROR_SUCCESS;
  }
}

//Destroy this machine
//Note: states in io parameter will be freed!
tnn_error tnn_machine_destroy(tnn_machine *m){
  tnn_state *states, *sel, *stmp;
  tnn_module *mel, *mtmp;
  tnn_error ret;
  states = m->io.states;

  //Destroy all of the parameters
  TNN_MACRO_ERRORTEST(tnn_param_destroy(&m->io), ret);
  TNN_MACRO_ERRORTEST(tnn_param_destroy(&m->p), ret);

  //Destroy all of the modules
  DL_FOREACH_SAFE(m->m, mel, mtmp){
    if((ret = tnn_module_destroy(mel)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_MODULE_FUNCNDEF){
      return ret;
    }
    free(mel);
  }
  m->m = NULL;

  //Destroy input module and output module
  if(((ret = tnn_module_destroy(&m->min)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_MODULE_FUNCNDEF) ||
     ((ret = tnn_module_destroy(&m->mout)) != TNN_ERROR_SUCCESS && ret != TNN_ERROR_MODULE_FUNCNDEF)){
    return ret;
  }

  //Destroy all the io states
  DL_FOREACH_SAFE(states, sel, stmp){
    free(sel);
  }

  return ret;
}

//Debug this machine
//Run debug for all of the components.
tnn_error tnn_machine_debug(tnn_machine *m){
  tnn_error ret;
  tnn_module *mel;
  bool ndef;

  printf("machine = %p, modules = %p\n", m, m->m);

  printf("sin: ");
  if((ret = tnn_state_debug(m->sin)) != TNN_ERROR_SUCCESS){
    printf("sin state debug error in machine\n");
    return ret;
  }

  printf("sout: ");
  if((ret = tnn_state_debug(m->sout)) != TNN_ERROR_SUCCESS){
    printf("sout state debug error in machine\n");
    return ret;
  }

  printf("io: ");
  if((ret = tnn_param_debug(&m->io)) != TNN_ERROR_SUCCESS){
    printf("io parameter debug error in machine \n");
    return ret;
  }

  printf("p: ");
  if((ret = tnn_param_debug(&m->p)) != TNN_ERROR_SUCCESS){
    printf("p parameter debug error in machine\n");
    return ret;
  }

  ndef = false;

  printf("min: ");
  if((ret = tnn_module_debug(&m->min)) != TNN_ERROR_SUCCESS){
    if(ret == TNN_ERROR_MODULE_FUNCNDEF){
      ndef = true;
    } else {
      printf("min module debug error in machine\n");
      return ret;
    }
  }

  DL_FOREACH(m->m, mel){
    if((ret = tnn_module_debug(mel)) != TNN_ERROR_SUCCESS){
      if(ret == TNN_ERROR_MODULE_FUNCNDEF){
	ndef = true;
      } else {
	printf("module debug error in machine\n");
	return ret;
      }
    }
  }

  printf("mout: ");
  if((ret = tnn_module_debug(&m->mout)) != TNN_ERROR_SUCCESS){
    if(ret == TNN_ERROR_MODULE_FUNCNDEF){
      ndef = true;
    } else {
      printf("mout module debug error in machine\n");
      return ret;
    }
  }

  if(ndef){
    return TNN_ERROR_MODULE_FUNCNDEF;
  } else {
    return TNN_ERROR_SUCCESS;
  }
}
