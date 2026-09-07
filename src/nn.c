#include "arena.h"
#include "config.h"
#include "tensor.h"
#include "types.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "utils.c"

typedef struct LinearLayer {
  uint32 in_ch;
  uint32 out_ch;
  Tensor *bias;
  Tensor *weight;
} LinearLayer;

typedef struct EmbeddingLayer {
  uint32 d_model;
  uint32 vocab_size;
  Tensor* weights;
} EmbeddingLayer;

typedef struct LayerNormLayer {
  uint32 d_model;
  Tensor* gamma;
  Tensor* beta;
  WEI_TYPE epsilon;
} LayerNormLayer;

typedef struct MHALayer {
  uint32 num_heads;
  uint32 d_model;

  LinearLayer* qkv;
  LinearLayer* fc;
  LayerNormLayer* ln;
} MHALayer;

typedef struct MLPLayer {
  uint32 d_model;
  uint32 hidden_size;

  LinearLayer* fc1;
  LinearLayer* fc2;
  LayerNormLayer* ln;
} MLPLayer;

typedef struct TransformerLayer {
  uint32 d_model;
  uint32 num_heads;
  uint32 hidden_size;
  uint32 num_layers;

  MHALayer **mha_layers;
  MLPLayer **mlp_layers;
} TransformerLayer;

LinearLayer *NNLayer(Arena *arena, uint32 in_ch, uint32 out_ch, uint8 bias);

EmbeddingLayer *Embedding(Arena *arena, uint32 d_model, uint32 vocab_size);
Tensor *EmbeddingCall(Arena *arena, Tensor *input, EmbeddingLayer* layer);

LayerNormLayer *LayerNorm(Arena *arena, uint32 d_model, WEI_TYPE epsilon);
Tensor *LayerNormCall(Arena *arena, Tensor *input, LayerNormLayer* layer);

MHALayer *MHANet(Arena *arena, uint32 num_heads, uint32 d_model, WEI_TYPE epsilon);
Tensor *MHACall(Arena *arena, Tensor *input, MHALayer* layer);

MLPLayer *MLPNet(Arena *arena, uint32 d_model, uint32 hidden_size, WEI_TYPE epsilon);
Tensor *MLPCall(Arena *arena, Tensor *input, MLPLayer* layer);

TransformerLayer *TransformerNet(Arena *arena, uint32 d_model, uint32 num_heads, uint32 hidden_size, uint32 num_layers);
Tensor *TransformerCall(Arena *arena, Tensor *input, TransformerLayer* layer);

Tensor *process_sequence(Arena *arena, LinearLayer **seq, uint32 num_layers, Tensor *input);
//activation fns
Tensor *ReLU(Arena *arena, Tensor *input);
Tensor *Sigmoid(Arena *arena, Tensor *input);
Tensor *Softmax(Arena *arena, Tensor *input);
// loss fns
Tensor *mse_loss(Arena *arena, Tensor *output, Tensor *target);
Tensor *cross_entropy(Arena *arena, Tensor *output, Tensor *target); // sparse categorical cross-entropy

LinearLayer *NNLayer(Arena *arena, uint32 in_ch, uint32 out_ch, uint8 bias){
  LinearLayer *layer = (LinearLayer *)arena_alloc(arena, sizeof(LinearLayer));
  layer->in_ch = in_ch;
  layer->out_ch = out_ch;

  uint32 wei_shape[] = {in_ch, out_ch};
  uint32 bias_shape[] = {1, out_ch};
  uint32 n_dim = 2;
  layer->bias = bias ? create_tensor(arena, bias_shape, n_dim, 0.0) : NULL;
  layer->weight = create_tensor(arena, wei_shape, n_dim, 0.0f);
  init_uniform(layer->weight, 1.0f / sqrtf((WEI_TYPE)in_ch));
  return layer;
}

EmbeddingLayer *Embedding(Arena *arena, uint32 d_model, uint32 vocab_size){
  EmbeddingLayer *layer = (EmbeddingLayer *)arena_alloc(arena, sizeof(EmbeddingLayer));
  layer->vocab_size = vocab_size;
  layer->d_model = d_model;
  
  uint32 wei_shape[2] = {vocab_size, d_model};
  layer->weights = create_tensor(arena, wei_shape, 2, 0.0f);
  init_uniform(layer->weights, 0.02f);

  return layer;
};

Tensor *EmbeddingCall(Arena *arena, Tensor *input, EmbeddingLayer* layer){
  // input -> (B, seq_len)
  // output -> (B, seq_len, embed_dim)
  
  if (input->num_dim != 2) return NULL;

  uint32 bsz = input->shape[0];
  uint32 seq_len = input->shape[1];
  uint32 *output_shape = (uint32 *)arena_alloc(arena, sizeof(uint32) * 3);
  output_shape[0] = bsz;
  output_shape[1] = seq_len;
  output_shape[2] = layer->d_model;
  Tensor *output = create_tensor(arena, output_shape, 3, 0.0);
  for (uint32 b = 0; b < input->shape[0]; b++){
    uint32 s_batch = b * seq_len * layer->d_model;

    for (uint32 s = 0; s < seq_len; s++){
      uint32 tok = input->data[b * input->shape[1] + s];

      uint32 s_row = s * layer->d_model;
      for (uint32 d = 0; d < layer->d_model; d++){
        output->data[s_batch + s_row + d] = layer->weights->data[tok * layer->d_model + d];
      }
    }
  }

  output->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  output->grad_fn->ops = EMBEDDING;
  output->grad_fn->num_inputs = 2;
  output->grad_fn->inputs[0] = input;
  output->grad_fn->inputs[1] = layer->weights;
  output->grad_fn->output = output;
  output->grad_fn->visited = 0;
  return output;
};

LayerNormLayer *LayerNorm(Arena *arena, uint32 d_model, WEI_TYPE epsilon){
  LayerNormLayer *layer = (LayerNormLayer *)arena_alloc(arena, sizeof(LayerNormLayer));
  layer->d_model = d_model;
  layer->epsilon = epsilon;
  uint32 gamma_shape[2] = {1, d_model};
  uint32 beta_shape[2] = {1, d_model};
  layer->gamma = create_tensor(arena, gamma_shape, 2, 1.0);
  layer->beta = create_tensor(arena, beta_shape, 2, 0.0);
  return layer;
}


Tensor *LayerNormCall(Arena *arena, Tensor *input, LayerNormLayer* layer){
  Tensor *output = create_tensor(arena, input->shape, input->num_dim, 0.0);

  uint32 B = input->shape[0];
  uint32 S = input->shape[1];
  uint32 D = input->shape[2];

  for (uint32 b = 0; b < B; b++){
    uint32 s_batch = b * S * D;
    for (uint32 s = 0; s < S; s++){
      uint32 s_row = s * D;

      WEI_TYPE *x_tok = &input->data[s_batch + s_row];
      WEI_TYPE _u = mean(x_tok, D);
      WEI_TYPE _v = var(x_tok, _u, D);
      WEI_TYPE inv_std = 1.0f / sqrtf(_v + layer->epsilon);

      for (uint32 d = 0; d < D; d++){
        WEI_TYPE x_hat = (x_tok[d] - _u) * inv_std;
        output->data[s_batch + s_row + d] = x_hat * layer->gamma->data[d] + layer->beta->data[d];
      }
    }
  }

  output->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  output->grad_fn->ops = LAYER_NORM;
  output->grad_fn->num_inputs = 3;
  output->grad_fn->inputs[0] = input;
  output->grad_fn->inputs[1] = layer->gamma;
  output->grad_fn->inputs[2] = layer->beta;
  output->grad_fn->output = output;
  output->grad_fn->visited = 0;
  output->grad_fn->epsilon = layer->epsilon;
  return output;
}

MHALayer *MHANet(Arena *arena, uint32 num_heads, uint32 d_model, WEI_TYPE epsilon){
  if (d_model % num_heads != 0) return NULL;
  MHALayer *layer = (MHALayer *)arena_alloc(arena, sizeof(MHALayer));
  layer->num_heads = num_heads;
  layer->d_model = d_model;

  layer->qkv = NNLayer(arena, d_model, d_model * 3, 0);
  layer->fc = NNLayer(arena, d_model, d_model, 0);
  layer->ln = LayerNorm(arena, d_model, epsilon);
  return layer;
}

Tensor *MHACall(Arena *arena, Tensor *input, MHALayer* layer){
  // input -> (B, S, D)
  // QKV_w -> (D, D * 3)
  // output -> (B, S, D)

  uint32 bz = input->shape[0];
  uint32 seq_len = input->shape[1];
  uint32 d_model = input->shape[2];
  
  uint32 new_inp_shape[2] = {bz * seq_len, d_model};
  Tensor *input_reshaped = reshape_tensor(arena, input, new_inp_shape, 2);
  Tensor *qkv = mul_tensor(arena, input_reshaped, layer->qkv->weight);

  uint32 qkv_shape[3] = {bz, seq_len, d_model * 3};
  qkv = reshape_tensor(arena, qkv, qkv_shape, 3);

  Tensor *q = split_tensor(arena, qkv, 0, layer->d_model);
  Tensor *k = split_tensor(arena, qkv, layer->d_model, layer->d_model);
  Tensor *v = split_tensor(arena, qkv, layer->d_model * 2, layer->d_model);

  // deviding for the head
  uint32 head_dim = d_model / layer->num_heads; 
  uint32 new_qkv_shape[4] = {bz, seq_len, layer->num_heads, head_dim};
  uint32 mid_out_shape[2] = {bz * seq_len, d_model};
  uint32 out_shape[3] = {bz, seq_len, d_model};
  
  uint32 q_shape[4], k_shape[4], v_shape[4];
  memcpy(q_shape, new_qkv_shape, sizeof(uint32) * 4);
  memcpy(k_shape, new_qkv_shape, sizeof(uint32) * 4);
  memcpy(v_shape, new_qkv_shape, sizeof(uint32) * 4);

  Tensor *q_head = reshape_tensor(arena, q, q_shape, 4); // (B, seq_len, num_heads, head_dim)
  Tensor *q_head_transposed = transpose_tensor(arena, q_head, 1, 2); // (B, num_heads, seq_len, head_dim)

  Tensor *k_head = reshape_tensor(arena, k, k_shape, 4); // (B, seq_len, num_heads, head_dim)
  Tensor *k_head_transposed = transpose_tensor(arena, transpose_tensor(arena, k_head, 1, 3), 1, 2);

  Tensor *v_head = reshape_tensor(arena, v, v_shape, 4);
  Tensor *v_head_transposed = transpose_tensor(arena, v_head, 1, 2);

  Tensor *scores = mul_tensor(arena, q_head_transposed, k_head_transposed);
  Tensor *normalized_scores = scale_tensor(arena, scores, (WEI_TYPE)(1.0f / sqrtf(head_dim)));
  Tensor *masked_scores = attention_mask_tensor(arena, normalized_scores);
  Tensor *attention = Softmax(arena, masked_scores);
  Tensor *weighted = mul_tensor(arena, attention, v_head_transposed);
  
  Tensor *weighted_t = transpose_tensor(arena, weighted, 1, 2); // (B, seq_len, num_heads, head_dim)
  Tensor *weighted_t_reshaped = reshape_tensor(arena, weighted_t, mid_out_shape, 2); // (B, seq_len, d_model)
  
  Tensor *out = mul_tensor(arena, weighted_t_reshaped, layer->fc->weight);
  Tensor *out_reshaped = reshape_tensor(arena, out, out_shape, 3);

  Tensor *out_add = add_tensor(arena, input, out_reshaped);
  Tensor *out_norm = LayerNormCall(arena, out_add, layer->ln);
  return out_norm;
}

MLPLayer *MLPNet(Arena *arena, uint32 d_model, uint32 hidden_size, WEI_TYPE epsilon){
  
  MLPLayer *layer = (MLPLayer *)arena_alloc(arena, sizeof(MLPLayer));
  layer->d_model = d_model;
  layer->hidden_size = hidden_size;
  layer->fc1 = NNLayer(arena, d_model, hidden_size, 0);
  layer->fc2 = NNLayer(arena, hidden_size, d_model, 0);
  layer->ln = LayerNorm(arena, d_model, epsilon);

  return layer;
}

Tensor *MLPCall(Arena *arena, Tensor *input, MLPLayer* layer){
  if (!input) return NULL;
  if (input->num_dim != 3) return NULL;
  
  uint32 bz = input->shape[0];
  uint32 seq_len = input->shape[1];
  uint32 d_model = input->shape[2];

  uint32 new_shape[2] = {bz * seq_len, d_model};
  Tensor *input_reshaped = reshape_tensor(arena, input, new_shape, 2);

  Tensor *x = mul_tensor(arena, input_reshaped, layer->fc1->weight);
  Tensor *x_relu = ReLU(arena, x);
  Tensor *x_out = mul_tensor(arena, x_relu, layer->fc2->weight);

  uint32 out_shape[3] = {bz, seq_len, d_model};
  Tensor *x_out_reshaped = reshape_tensor(arena, x_out, out_shape, 3);
  Tensor *out = add_tensor(arena, input, x_out_reshaped);
  Tensor *out_norm = LayerNormCall(arena, out, layer->ln);
  return out_norm;
}

TransformerLayer *TransformerNet(Arena *arena, uint32 d_model, uint32 num_heads, uint32 hidden_size, uint32 num_layers){
  TransformerLayer *layer = (TransformerLayer *)arena_alloc(arena, sizeof(TransformerLayer));
  layer->d_model = d_model;
  layer->num_heads = num_heads;
  layer->hidden_size = hidden_size;
  layer->num_layers = num_layers;

  layer->mha_layers = (MHALayer **)arena_alloc(arena, sizeof(MHALayer *) * num_layers);
  layer->mlp_layers = (MLPLayer **)arena_alloc(arena, sizeof(MLPLayer *) * num_layers);

  for (uint32 i = 0; i < num_layers; i++){
    layer->mha_layers[i] = MHANet(arena, num_heads, d_model, 1e-5);
    layer->mlp_layers[i] = MLPNet(arena, d_model, hidden_size, 1e-5);
  }
  return layer;
}

Tensor *TransformerCall(Arena *arena, Tensor *input, TransformerLayer* layer){
  for (uint32 i = 0; i < layer->num_layers; i++){
    input = MHACall(arena, input, layer->mha_layers[i]);
    input = MLPCall(arena, input, layer->mlp_layers[i]);
  }
  return input;
}

Tensor *process_sequence(Arena *arena, LinearLayer **seq, uint32 num_layers, Tensor *input){
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
  // output :- (B, D*, C) -> logits; left unmodified
  // target :- (B, 1) -> should contains class index

  if (!output || !target) return NULL;
  if (target->num_dim != 2 || target->shape[target->num_dim-1] != 1) return NULL;

  size_t total_batch = 1;
  for (uint32 i = 0; i < output->num_dim; i++) {
    if (i < output->num_dim - 2) {
      total_batch *= output->shape[i];
    }
  }

  size_t R = output->shape[output->num_dim - 2];
  size_t C = output->shape[output->num_dim - 1];

  WEI_TYPE cumm_loss = 0.0f;

  for (size_t b = 0; b < total_batch; b++) {
    for (size_t row = 0; row < R; row++) {
      WEI_TYPE *row_vals = &output->data[(b * R + row) * C];
      WEI_TYPE z_true = row_vals[(size_t)target->data[b * R + row]];
      WEI_TYPE max = row_vals[0];
      for (size_t col = 1; col < C; col++) {
        max = row_vals[col] > max ? row_vals[col] : max;
      }
      WEI_TYPE sum = 0.0f;
      for (size_t col = 0; col < C; col++) {
        sum += expf(row_vals[col] - max);
      }

      cumm_loss += max + logf(sum) - z_true;
    }
  }

  uint32 *loss_shape = arena_alloc(arena, sizeof(uint32) * 2);
  loss_shape[0] = 1;
  loss_shape[1] = 1;
  Tensor *loss = create_tensor(arena, loss_shape, 2, cumm_loss / (total_batch * output->shape[output->num_dim - 2]));
  loss->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  loss->grad_fn->ops = SOFTMAX_CROSS_ENTROPY;
  loss->grad_fn->num_inputs = 2;
  loss->grad_fn->inputs[0] = output;
  loss->grad_fn->inputs[1] = target;
  loss->grad_fn->output = loss;
  loss->grad_fn->visited = 0;
  return loss;
}
