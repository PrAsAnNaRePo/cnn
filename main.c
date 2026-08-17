#include "src/config.h"
#include "src/nn.c"
#include "src/tensor.h"
#include "src/types.h"
#include "src/arena.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// CONSTANTS

#define VOCAB_SIZE 1024
#define D_MODEL 256
#define MAX_LEN 1024


int main(void){
  Arena new_arena = arena_create(GiB(2));

  WEI_TYPE wei_init = 0.001f;
  EmbeddingLayer* embed_layer = Embedding(&new_arena, D_MODEL, VOCAB_SIZE);
  EmbeddingLayer* pos_embed_layer = Embedding(&new_arena, MAX_LEN, D_MODEL);

  return 0;
}
