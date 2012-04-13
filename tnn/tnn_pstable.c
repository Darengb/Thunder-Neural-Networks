/* Thunder Neural Networks Pointer - State Hash Table Utility Source
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/13/2012
 *
 * This source implements the following functions:
 * tnn_error tnn_pstable_init(tnn_pstable *t);
 * tnn_error tnn_pstable_add(tnn_pstable *t, tnn_state *key, tnn_state *s);
 * tnn_error tnn_pstable_delete(tnn_pstable *t, tnn_state *key);
 * tnn_error tnn_pstable_find(tnn_pstable *t, tnn_state *key, tnn_state **s);
 * tnn_error tnn_pstable_destroy(tnn_pstable *t);
 * tnn_error tnn_pstable_get_length(tnn_pstable *t, size_t *l);
 */

#include <tnn/uthash.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_error.h>
#include <tnn/tnn_pstable.h>
#include <stdlib.h> //malloc and free

//Initialize the table
tnn_error tnn_pstable_init(tnn_pstable *t){
  t->table = NULL;
  t->length = 0L;
  return TNN_ERROR_SUCCESS;
}

//Add a key-s pair to the table
tnn_error tnn_pstable_add(tnn_pstable *t, tnn_state *key, tnn_state *s){
  tnn_pstable_el *el;

  //Check whether already exists
  HASH_FIND_PTR(t->table, &key, el);
  if(el != NULL){
    return TNN_ERROR_PSTABLE_EXIST;
  }

  //Insert a new pair
  el = (tnn_pstable_el*)malloc(sizeof(tnn_pstable_el));
  el->key = key;
  el->s = s;
  HASH_ADD_PTR(t->table, key, el);
  t->length = t->length + 1;

  return TNN_ERROR_SUCCESS;
}

//Delete a key-s pair from the table
tnn_error tnn_pstable_delete(tnn_pstable *t, tnn_state *key){
  tnn_pstable_el *el;

  //Check whether it exists
  HASH_FIND_PTR(t->table, &key, el);
  if(el == NULL){
    return TNN_ERROR_PSTABLE_NEXIST;
  }

  //Delete it
  HASH_DEL(t->table, el);
  t->length = t->length - 1;

  return TNN_ERROR_SUCCESS;
}

//Find a key-s pair from the table
tnn_error tnn_pstable_find(tnn_pstable *t, tnn_state *key, tnn_state **s){
  tnn_pstable_el *el;

  //Find the pointer
  HASH_FIND_PTR(t->table, &key, el);
  if(el == NULL){
    return TNN_ERROR_PSTABLE_NEXIST;
  }

  //Get the address
  *s = el->s;

  return TNN_ERROR_SUCCESS;
}

//Destroy the table
tnn_error tnn_pstable_destroy(tnn_pstable *t){
  tnn_pstable_el *el, *tmp;

  HASH_ITER(hh, t->table, el, tmp){
    HASH_DEL(t->table, el);
    free(el);
  }
  t->length = 0;
  t->table = NULL;

  return TNN_ERROR_SUCCESS;
}

//Debug the table
tnn_error tnn_pstable_debug(tnn_pstable *t){
  tnn_pstable_el *el, *tmp;

  printf("pstable = %p, table = %p, length = %ld\n", t, t->table, t->length);

  HASH_ITER(hh,t->table, el, tmp){
    printf("pstable_el = %p, key = %p, s = %p\n", el, el->key, el->s);
  }
  return TNN_ERROR_SUCCESS;
}

//Get the length
tnn_error tnn_pstable_get_length(tnn_pstable *t, size_t *l){
  *l = t->length;
  return TNN_ERROR_SUCCESS;
}
