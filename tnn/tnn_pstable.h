/* Thunder Neural Networks Pointer - State Hash Table Utility Header
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/12/2012
 *
 * This header defines the following structures:
 * tnn_pstable_el(tnn_state *key, tnn_state *s, UT_hash_handle hh)
 * tnn_pstable(tnn_sptable_el *table)
 *
 * This header defines the following functions:
 * tnn_error tnn_pstable_init(tnn_pstable *t);
 * tnn_error tnn_pstable_add(tnn_pstable *t, tnn_state *key, tnn_state *s);
 * tnn_error tnn_pstable_delete(tnn_pstable *t, tnn_state *key);
 * tnn_error tnn_pstable_find(tnn_pstable *t, tnn_state *key, tnn_state **s);
 * tnn_error tnn_pstable_destroy(tnn_pstable *t);
 * tnn_error tnn_pstable_debug(tnn_pstable *t);
 * tnn_error tnn_pstable_get_length(tnn_pstable *t, size_t *l);
 */

#include <tnn/uthash.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_error.h>

#ifndef TNN_PSTABLE_H
#define TNN_PSTABLE_H

//Element of the pstable
typedef struct __STRUCT_tnn_pstable_el{
  //Key of the state (a pointer to another state)
  tnn_state *key;
  //The state
  tnn_state *s;
  //UTHash handler
  UT_hash_handle hh;
} tnn_pstable_el;

//The pstable type
typedef struct __STRUCT_tnn_pstable{
  //The table
  tnn_pstable_el *table;
  //The length
  size_t length;
} tnn_pstable;

//Initialize the table
tnn_error tnn_pstable_init(tnn_pstable *t);

//Add a key-s pair to the table
tnn_error tnn_pstable_add(tnn_pstable *t, tnn_state *key, tnn_state *s);

//Delete a key-s pair from the table
tnn_error tnn_pstable_delete(tnn_pstable *t, tnn_state *key);

//Find a key-s pair from the table
tnn_error tnn_pstable_find(tnn_pstable *t, tnn_state *key, tnn_state **s);

//Destroy the table
tnn_error tnn_pstable_destroy(tnn_pstable *t);

//Debug the table
tnn_error tnn_pstable_debug(tnn_pstable *t);

//Get the length
tnn_error tnn_pstable_get_length(tnn_pstable *t, size_t *l);

#endif //TNN_PSTABLE_H
