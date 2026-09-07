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
#define NUM_LAYERS 8


int main(void){
  Arena persistent_arena = arena_create(GiB(1));
  Arena temp_arena = arena_create(GiB(2));

  EmbeddingLayer* embed_layer = Embedding(&persistent_arena, D_MODEL, VOCAB_SIZE);
  EmbeddingLayer* pos_embed_layer = Embedding(&persistent_arena, D_MODEL, MAX_LEN);
  TransformerLayer* transformer = TransformerNet(&persistent_arena, D_MODEL, NUM_HEADS, 4 * D_MODEL, NUM_LAYERS);

  uint32 input_shape[2] = {1, 3};
  Tensor *input = create_tensor(&temp_arena, input_shape, 2, 0.0, 0);
  input->data[0] = 123;
  input->data[1] = 200;
  input->data[2] = 333;
  Tensor *pos_input = create_tensor(&temp_arena, input_shape, 2, 0.0, 0);
  for (int i = 0; i < 3; i++) pos_input->data[i] = i;
  Tensor *embed = EmbeddingCall(&temp_arena, input, embed_layer);
  Tensor *pos_embed = EmbeddingCall(&temp_arena, pos_input, pos_embed_layer);
  Tensor *embeddings = add_tensor(&temp_arena, embed, pos_embed);
  Tensor *out = TransformerCall(&temp_arena, embeddings, transformer);
  print_shape(out);

  backward(out);

  AdamState *adam = get_adam_state(&persistent_arena, out, 0.001f);
  printf("adam weights: %u\n", adam->num_weights);

  return 0;
}
