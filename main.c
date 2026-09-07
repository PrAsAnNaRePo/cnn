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


int main(void){
  Arena new_arena = arena_create(GiB(2));

  WEI_TYPE wei_init = 0.001f;
  EmbeddingLayer* embed_layer = Embedding(&new_arena, D_MODEL, VOCAB_SIZE);
  EmbeddingLayer* pos_embed_layer = Embedding(&new_arena, MAX_LEN, D_MODEL);
  MHALayer* mha_layer = MHANet(&new_arena, NUM_HEADS, D_MODEL, wei_init, 1e-5f);
  MLPLayer* mlp_layer = MLPNet(&new_arena, D_MODEL, 4 * D_MODEL, wei_init, 1e-5f);

  uint32 input_shape[2] = {1, 3};
  Tensor *input = create_tensor(&new_arena, input_shape, 2, 0.0);
  input->data[0] = 123;
  input->data[1] = 200;
  input->data[2] = 333;
  Tensor *output = EmbeddingCall(&new_arena, input, embed_layer);
  Tensor *attention = MHACall(&new_arena, output, mha_layer);
  Tensor *out = MLPCall(&new_arena, attention, mlp_layer);
  print_shape(out);

  backward(out);

  return 0;
}
