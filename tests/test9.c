/* Dummy Test 9 for TNN Utilities
 * By Xiang Zhang @ New York University
 * Version 0.1, 04/13/2012
 *
 * Tests for the following utilities were performed:
 * tnn_pstable
 *
 * Results:
 * Perfect Programming!
 */

#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <tnn/tnn_state.h>
#include <tnn/tnn_pstable.h>

#define TEST_FUNC(func) (func == TNN_ERROR_SUCCESS?"YES":"NO")

#define A 5000;
#define B 700;
#define C 2;
#define D 1;
#define E 3;
#define N 4;

int main(){
  int i;
  size_t l;
  tnn_state *key;
  tnn_state *s;
  tnn_pstable t;

  printf("Initializing t: %s\n", TEST_FUNC(tnn_pstable_init(&t)));

  //Adding dummy variables
  key = (tnn_state *)A;
  s = (tnn_state *)B;
  for(i = 0; i < 4; i = i + 1){
    printf("Inserting key=%p, s=%p: %s\n", key, s, TEST_FUNC(tnn_pstable_add(&t, key, s)));
    key = key + 1;
    s = s - 1;
  }
  key = key - 1;
  s = s + 1;
  printf("Inserting key=%p, s=%p: %s\n", key, s, TEST_FUNC(tnn_pstable_add(&t, key, s)));

  //Debug
  printf("Debugging t: %s\n", TEST_FUNC(tnn_pstable_debug(&t)));

  //Delete a key
  key = (tnn_state *)A;
  key = key + C;
  printf("Deleting key=%p: %s\n", key, TEST_FUNC(tnn_pstable_delete(&t, key)));
  printf("Debugging t: %s\n", TEST_FUNC(tnn_pstable_debug(&t)));
  key = (tnn_state *)A;
  key = key - D;
  printf("Deleting key=%p: %s\n", key, TEST_FUNC(tnn_pstable_delete(&t, key)));
  printf("Debugging t: %s\n", TEST_FUNC(tnn_pstable_debug(&t)));

  //Find a key
  key = (tnn_state *)A;
  key = key + E;
  printf("Finding key=%p, s=%p: %s\n", key, s, TEST_FUNC(tnn_pstable_find(&t, key, &s)));
  key = (tnn_state *)A;
  key = key + C;
  printf("Finding key=%p, s=%p: %s\n", key, s, TEST_FUNC(tnn_pstable_find(&t, key, &s)));

  //Get the length
  printf("Getting the length=%ld: %s\n", l, TEST_FUNC(tnn_pstable_get_length(&t, &l)));

  //Destroy the table
  printf("Destroying t: %s\n", TEST_FUNC(tnn_pstable_destroy(&t)));
  printf("Debugging t: %s\n", TEST_FUNC(tnn_pstable_debug(&t)));

  return 0;
}
