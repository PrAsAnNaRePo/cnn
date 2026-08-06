#include "arena.h"
#include "config.h"
#include "tensor.h"
#include "types.h"
#include <math.h>
#include <stddef.h>

typedef struct nnLayer {
  uint32 in_ch;
  uint32 out_ch;
  Tensor *bias;
  Tensor *weight;
} nnLayer;

nnLayer *NNLayer(Arena *arena, uint32 in_ch, uint32 out_ch, uint8 bias, WEI_TYPE wei_init);
Tensor *process_sequence(Arena *arena, nnLayer **seq, uint32 num_layers, Tensor *input);
//activation fns
Tensor *ReLU(Arena *arena, Tensor *input);
Tensor *Sigmoid(Arena *arena, Tensor *input);
Tensor *Softmax(Arena *arena, Tensor *input);
// loss fns
Tensor *mse_loss(Arena *arena, Tensor *output, Tensor *target);
Tensor *cross_entropy(Arena *arena, Tensor *output, Tensor *target); // sparse categorical cross-entropy

nnLayer *NNLayer(Arena *arena, uint32 in_ch, uint32 out_ch, uint8 bias, WEI_TYPE wei_init){
  nnLayer *layer = (nnLayer *)arena_alloc(arena, sizeof(nnLayer));
  layer->in_ch = in_ch;
  layer->out_ch = out_ch;

  uint32 wei_shape[] = {in_ch, out_ch};
  uint32 bias_shape[] = {1, out_ch};
  uint32 n_dim = 2;
  layer->bias = bias ? create_tensor(arena, bias_shape, n_dim, 0.0) : NULL;
  layer->weight = create_tensor(arena, wei_shape, n_dim, wei_init);
  return layer;
}

Tensor *process_sequence(Arena *arena, nnLayer **seq, uint32 num_layers, Tensor *input){
  for (uint32 layer = 0; layer < num_layers; layer++){
    input = mul_tensor(arena, input, seq[layer]->weight);
    if (seq[layer]->bias) input = add_tensor(arena, input, seq[layer]->bias);
  }
  return input;
}

Tensor *ReLU(Arena *arena, Tensor *input){
  Tensor *output = create_tensor(arena, input->shape, input->num_dim, 0.0);
  for (uint32 i = 0; i < input->numel; i++){
    output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
  }

  output->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  output->grad_fn->ops = RELU;
  output->grad_fn->num_inputs = 1;
  output->grad_fn->inputs[0] = input;
  output->grad_fn->output = output;
  output->grad_fn->visited = 0;
  return output;
}

Tensor *Sigmoid(Arena *arena, Tensor *input){
  if (!input) return NULL;

  Tensor *output = create_tensor(arena, input->shape, input->num_dim, 0.0f);
  if (!output) return NULL;

  for (uint32 i = 0; i < input->numel; i++){
    output->data[i] = 1.0f / (1.0f + expf(-input->data[i]));
  }

  output->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  output->grad_fn->ops = SIGMOID;
  output->grad_fn->num_inputs = 1;
  output->grad_fn->inputs[0] = input;
  output->grad_fn->output = output;
  output->grad_fn->visited = 0;
  return output;
}

Tensor *Softmax(Arena *arena, Tensor *input){
  if (!input) return NULL;

  Tensor *output = create_tensor(arena, input->shape, input->num_dim, 0.0f);
  if (!output) return NULL;

  size_t total_batch = 1;
  for (uint32 i = 0; i < input->num_dim; i++) {
    if (i < input->num_dim - 2) {
      total_batch *= input->shape[i];
    }
  }

  for (size_t b = 0; b < total_batch; b++) {
    for (size_t row = 0; row < input->shape[input->num_dim - 2]; row++) {
      WEI_TYPE *col_vals = (WEI_TYPE *) arena_alloc(arena, sizeof(WEI_TYPE) * input->shape[input->num_dim - 1]);
      for (size_t col = 0; col < input->shape[input->num_dim - 1]; col++) {
        col_vals[col] = input->data[b * input->shape[input->num_dim - 2] * input->shape[input->num_dim - 1] + row * input->shape[input->num_dim - 1] + col];
      }
      WEI_TYPE max = col_vals[0];
      for (size_t col = 1; col < input->shape[input->num_dim - 1]; col++) {
        max = col_vals[col] > max ? col_vals[col] : max;
      }
      WEI_TYPE sum = 0;
      for (size_t col = 0; col < input->shape[input->num_dim - 1]; col++) {
        col_vals[col] = expf(col_vals[col] - max);
        sum += col_vals[col];
      }
      for (size_t col = 0; col < input->shape[input->num_dim - 1]; col++) {
        output->data[b * input->shape[input->num_dim - 2] * input->shape[input->num_dim - 1] + row * input->shape[input->num_dim - 1] + col] = col_vals[col] / sum;
      }
    }
  }

  output->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  output->grad_fn->ops = SOFTMAX;
  output->grad_fn->num_inputs = 1;
  output->grad_fn->inputs[0] = input;
  output->grad_fn->output = output;
  output->grad_fn->visited = 0;

  return output;
}

Tensor *mse_loss(Arena *arena, Tensor *output, Tensor *target){
  if (!output || !target) return NULL;
  if (output->num_dim != target->num_dim) return NULL;
  if (output->num_dim > 2 || target->num_dim > 2) return NULL;
  if (output->shape[0] != target->shape[0] || output->shape[1] != target->shape[1]) return NULL;

  size_t bz = target->shape[0];
  size_t num_outputs = target->shape[target->num_dim-1];

  WEI_TYPE total_loss = 0.0f;
  for (size_t b = 0; b < bz; b++) {
    WEI_TYPE intermediate = 0.0f;
    size_t s_row = b * output->shape[output->num_dim-1];
    for (size_t o = 0; o < num_outputs; o++) {
      intermediate += (target->data[s_row + o] - output->data[s_row + o]) * (target->data[s_row + o] - output->data[s_row + o]);
    }
    total_loss += intermediate / num_outputs;
  }

  uint32 *loss_shape = arena_alloc(arena, sizeof(uint32) * 2);
  loss_shape[0] = 1;
  loss_shape[1] = 1;
  Tensor *loss = create_tensor(arena, loss_shape, 2, total_loss / bz);
  loss->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  loss->grad_fn->ops = MSE_LOSS;
  loss->grad_fn->num_inputs = 2;
  loss->grad_fn->inputs[0] = output;
  loss->grad_fn->inputs[1] = target;
  loss->grad_fn->output = loss;
  loss->grad_fn->visited = 0;
  return loss;
}

Tensor *cross_entropy(Arena *arena, Tensor *output, Tensor *target){
  // output :- (B, D*, C)
  // target :- (B, 1) -> should contains class index

  if (!output || !target) return NULL;
  if (target->num_dim != 2 || target->shape[target->num_dim-1] != 1) return NULL;
  
  size_t total_batch = 1;
  for (uint32 i = 0; i < output->num_dim; i++) {
    if (i < output->num_dim - 2) {
      total_batch *= output->shape[i];
    }
  }

  WEI_TYPE cumm_loss = 0.0f;

  for (size_t b = 0; b < total_batch; b++) {
    for (size_t row = 0; row < output->shape[output->num_dim - 2]; row++) {
      WEI_TYPE *col_vals = (WEI_TYPE *) arena_alloc(arena, sizeof(WEI_TYPE) * output->shape[output->num_dim - 1]);
      for (size_t col = 0; col < output->shape[output->num_dim - 1]; col++) {
        col_vals[col] = output->data[b * output->shape[output->num_dim - 2] * output->shape[output->num_dim - 1] + row * output->shape[output->num_dim - 1] + col];
      }
      WEI_TYPE z_true = col_vals[(uint32)target->data[b * output->shape[output->num_dim - 2] + row]];
      WEI_TYPE max = col_vals[0];
      for (size_t col = 1; col < output->shape[output->num_dim - 1]; col++) {
        max = col_vals[col] > max ? col_vals[col] : max;
      }
      WEI_TYPE sum = 0;
      for (size_t col = 0; col < output->shape[output->num_dim - 1]; col++) {
        col_vals[col] = expf(col_vals[col] - max);
        sum += col_vals[col];
      }

      cumm_loss += max + logf(sum) - z_true;
      for (size_t col = 0; col < output->shape[output->num_dim - 1]; col++) {
        output->data[b * output->shape[output->num_dim - 2] * output->shape[output->num_dim - 1] + row * output->shape[output->num_dim - 1] + col] = col_vals[col] / sum;
      }
    }
  }

  uint32 *loss_shape = arena_alloc(arena, sizeof(uint32) * 2);
  loss_shape[0] = 1;
  loss_shape[1] = 1;
  Tensor *loss = create_tensor(arena, loss_shape, 2, cumm_loss / (total_batch * output->shape[output->num_dim - 2]));
  loss->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  loss->grad_fn->ops = SOFTMAX_CROSS_ENTROPY;
  loss->grad_fn->num_inputs = 1;
  loss->grad_fn->inputs[0] = output;
  loss->grad_fn->inputs[1] = target;
  loss->grad_fn->output = loss;
  loss->grad_fn->visited = 0;
  return loss;
}
