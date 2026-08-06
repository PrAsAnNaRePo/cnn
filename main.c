#include "src/autograd.h"
#include "src/nn.c"
#include "src/tensor.h"
#include "src/types.h"
#include "src/arena.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

int main(void){
  Arena new_arena = arena_create(GiB(2));

  uint32 input_dim = 4;
  uint32 input_shape[4] = {1, 3, 255, 255};
  uint32 flat_input_dim = 2;
  uint32 flat_input_shape[2] = {1, 3*255*255};
  Tensor* input = create_tensor(&new_arena, input_shape, input_dim, 5.0f);
  Tensor* flattened_input = reshape_tensor(&new_arena, input, flat_input_shape, flat_input_dim);
  
  nnLayer* layer1 = NNLayer(&new_arena, 3*255*255, 256, 1, 0.001f);
  nnLayer* layer2 = NNLayer(&new_arena, 256, 4, 1, 0.001f);

  nnLayer **seq = (nnLayer **)arena_alloc(&new_arena, sizeof(nnLayer *) * 2);
  seq[0] = layer1;
  seq[1] = layer2;

  Tensor* output = process_sequence(&new_arena, seq, 2, flattened_input);
  print_tensor(output);

  Tensor *softmax_output = Softmax(&new_arena, input);
  print_tensor(softmax_output);

  backward(softmax_output);

  arena_free(&new_arena);
  return 0;
}
