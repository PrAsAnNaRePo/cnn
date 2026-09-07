#include "src/config.h"
#include "src/nn.c"
#include "src/tensor.h"
#include "src/types.h"
#include "src/arena.h"
#include "src/autograd.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// CONSTANTS

#define VOCAB_SIZE 1024
#define D_MODEL 256
#define MAX_LEN 1024
#define NUM_HEADS 8
#define NUM_LAYERS 2


int main(void){
  Arena new_arena = arena_create(GiB(2));

  EmbeddingLayer* embed_layer = Embedding(&new_arena, D_MODEL, VOCAB_SIZE);
  EmbeddingLayer* pos_embed_layer = Embedding(&new_arena, MAX_LEN, D_MODEL);
  TransformerLayer* transformer = TransformerNet(&new_arena, D_MODEL, NUM_HEADS, 4 * D_MODEL, NUM_LAYERS);

  uint32 input_shape[2] = {1, 3};
  Tensor *input = create_tensor(&new_arena, input_shape, 2, 0.0);
  input->data[0] = 123;
  input->data[1] = 200;
  input->data[2] = 333;
  Tensor *output = EmbeddingCall(&new_arena, input, embed_layer);
  Tensor *out = TransformerCall(&new_arena, output, transformer);
  print_shape(out);

  backward(out);

  return 0;
}
