/* Thunder Neural Networks - Macro Utility
 * By Xiang Zhang @ New York University
 * Version 0.1, 03/31/2012
 *
 * This header defines the following macros:
 * TNN_MACRO_ERRORTEST
 * TNN_MACRO_GSLTEST
 */

#ifndef TNN_MACRO_H
#define TNN_MACRO_H

#define TNN_MACRO_ERRORTEST(func,ret)   	\
  if ((ret = func) != TNN_ERROR_SUCCESS){ 	\
    return ret;			    		\
  }

#define TNN_MACRO_GSLTEST(func)			\
  if(func != 0){				\
    return TNN_ERROR_GSL;			\
  }

#endif //TNN_MACRO_H
