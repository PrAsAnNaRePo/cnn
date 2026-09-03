#include "autograd.h"
#include "arena.h"
#include "config.h"
#include "tensor.h"
#include "types.h"
#include <math.h>
#include <stddef.h>

void build_topo(Tensor *t, Tensor **topo_list, uint32 *topo_size){
  if (t==NULL) return;

  if (t->grad_fn != NULL && t->grad_fn->visited == 0){
    t->grad_fn->visited = 1;

    for (uint32 i = 0; i < t->grad_fn->num_inputs; i++){
      build_topo(t->grad_fn->inputs[i], topo_list, topo_size);
    }
    topo_list[*topo_size] = t->grad_fn->output;
    (*topo_size)++;
  }
}

void backward(Tensor *t){
  for (uint32 i = 0; i < t->numel; i++){
    t->grad[i] = 1.0f;
  }

  Tensor *topo_list[1024];
  uint32 topo_size = 0;
  build_topo(t, topo_list, &topo_size);

  for (int32 i = (int32)topo_size - 1; i >= 0; i--){
    Tensor *curr = topo_list[i];

    switch (curr->grad_fn->ops){
      case ADD:
        for (uint32 j = 0; j < curr->grad_fn->num_inputs; j++){
          for (uint32 k = 0; k < curr->grad_fn->inputs[j]->numel; k++){
            curr->grad_fn->inputs[j]->grad[k] += curr->grad[k];
          }
        }
        break;

      case MATMUL: {
        Tensor *A = curr->grad_fn->inputs[0];
        Tensor *B = curr->grad_fn->inputs[1];

        uint32 ndim = A->num_dim;
        size_t M = A->shape[ndim - 2];
        size_t K = A->shape[ndim - 1];
        size_t N = B->shape[ndim - 1];

        size_t total_batch = 1;
        for (uint32 b_idx = 0; b_idx < ndim - 2; b_idx++) {
          total_batch *= A->shape[b_idx];
        }

        for (size_t b = 0; b < total_batch; b++) {
          size_t s_batch_a = b * M * K;
          size_t s_batch_b = b * K * N;
          size_t s_batch_c = b * M * N;

          // 1. dA = dC * B^T
          for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
              WEI_TYPE sum = 0.0f;
              for (size_t j = 0; j < N; j++) {
                sum += curr->grad[s_batch_c + i * N + j] * B->data[s_batch_b + k * N + j];
              }
              A->grad[s_batch_a + i * K + k] += sum;
            }
          }

          // 2. dB = A^T * dC
          for (size_t k = 0; k < K; k++) {
            for (size_t j = 0; j < N; j++) {
              WEI_TYPE sum = 0.0f;
              for (size_t i = 0; i < M; i++) {
                sum += A->data[s_batch_a + i * K + k] * curr->grad[s_batch_c + i * N + j];
              }
              B->grad[s_batch_b + k * N + j] += sum;
            }
          }
        }
        break;
      }

      case SUB:
        for (uint32 k = 0; k < curr->grad_fn->inputs[0]->numel; k++) {
          curr->grad_fn->inputs[0]->grad[k] += curr->grad[k];
          curr->grad_fn->inputs[1]->grad[k] -= curr->grad[k];
        }
        break;

      case RELU:
        for (uint32 k = 0; k < curr->grad_fn->inputs[0]->numel; k++) {
          if (curr->grad_fn->inputs[0]->data[k] > 0.0f) {
            curr->grad_fn->inputs[0]->grad[k] += curr->grad[k];
          }
        }
        break;

      case SIGMOID:
        for (uint32 k = 0; k < curr->grad_fn->inputs[0]->numel; k++) {
          WEI_TYPE s = curr->data[k];
          curr->grad_fn->inputs[0]->grad[k] += curr->grad[k] * s * (1.0f - s);
        }
        break;
      
      case SOFTMAX: {
        size_t total_batch = 1;
        for (uint32 i = 0; i < curr->grad_fn->inputs[0]->num_dim; i++) {
          if (i < curr->grad_fn->inputs[0]->num_dim - 2) {
            total_batch *= curr->grad_fn->inputs[0]->shape[i];
          }
        }
        size_t R = curr->grad_fn->inputs[0]->shape[curr->grad_fn->inputs[0]->num_dim - 2];
        size_t C = curr->grad_fn->inputs[0]->shape[curr->grad_fn->inputs[0]->num_dim - 1];
        for (size_t b = 0; b < total_batch; b++) {
          size_t s_batch = b * R * C;
          for (size_t row = 0; row < R; row++) {
            size_t s_row = row * C;
            for (size_t col = 0; col < C; col++) {
              WEI_TYPE sum = 0.0f;
              for (size_t n_col = 0; n_col < C; n_col++) {
                WEI_TYPE si = curr->data[s_batch + s_row + n_col];
                WEI_TYPE sj = curr->data[s_batch + s_row + col];
                WEI_TYPE jacobian_val = (n_col == col) ? si * (1.0f - si) : -si * sj;
                WEI_TYPE grad_out_i = curr->grad[s_batch + s_row + n_col];
                sum += grad_out_i * jacobian_val;
              }
              curr->grad_fn->inputs[0]->grad[s_batch + s_row + col] += sum;
            }
          }
        }
        break;
      
      case MSE_LOSS: {
        uint32 numel = curr->grad_fn->inputs[0]->numel;
        for (uint32 k = 0; k < numel; k++) {
          curr->grad_fn->inputs[0]->grad[k] += curr->grad[k] * 2.0f / numel * (curr->grad_fn->inputs[0]->data[k] - curr->grad_fn->inputs[1]->data[k]);
        }
        break;
        }

      case SOFTMAX_CROSS_ENTROPY: {
        size_t total_batch = 1;
        for (uint32 i = 0; i < curr->grad_fn->inputs[0]->num_dim; i++) {
          if (i < curr->grad_fn->inputs[0]->num_dim - 2) {
            total_batch *= curr->grad_fn->inputs[0]->shape[i];
          }
        }
        size_t R = curr->grad_fn->inputs[0]->shape[curr->grad_fn->inputs[0]->num_dim - 2];
        size_t C = curr->grad_fn->inputs[0]->shape[curr->grad_fn->inputs[0]->num_dim - 1];
        
        size_t N = total_batch * R;
        
        WEI_TYPE incoming_grad = curr->grad[0];
        
        for (size_t b = 0; b < total_batch; b++) {
          size_t s_batch = b * R * C;
          for (size_t row = 0; row < R; row++) {
            size_t s_row = row * C;
            size_t true_idx = (size_t)curr->grad_fn->inputs[1]->data[s_batch / C + row]; // Adjusted target index lookup
            
            for (size_t col = 0; col < C; col++) {
              size_t current_idx = s_batch + s_row + col;
              
              WEI_TYPE indicator = (col == true_idx) ? 1.0f : 0.0f;
              
              WEI_TYPE prob = curr->grad_fn->inputs[0]->data[current_idx];
              
              curr->grad_fn->inputs[0]->grad[current_idx] += (incoming_grad / (WEI_TYPE)N) * (prob - indicator);
            }
          }
        }
        break;
        }

      case EMBEDDING: {
        Tensor *input = curr->grad_fn->inputs[0];
        Tensor *weights = curr->grad_fn->inputs[1];

        uint32 B = input->shape[0];
        uint32 T = input->shape[1];
        uint32 D = weights->shape[1];
        for (uint32 b = 0; b < B; b++){
          for (uint32 seq_len = 0; seq_len < T; seq_len++){
            uint32 tok_idx = input->data[b * T + seq_len];

            for (uint32 d = 0; d < D; d++){
              weights->grad[tok_idx * D + d] += curr->grad[b * T * D + seq_len * D + d];
            }
          }
        }
        break;
        }

      case LAYER_NORM: {
        Tensor *input = curr->grad_fn->inputs[0];
        Tensor *gamma = curr->grad_fn->inputs[1];
        Tensor *beta  = curr->grad_fn->inputs[2];

        uint32 B = input->shape[0];
        uint32 S = input->shape[1];
        uint32 D = input->shape[2];
        WEI_TYPE eps = curr->grad_fn->epsilon;

        for (uint32 b = 0; b < B; b++){
          for (uint32 s = 0; s < S; s++){
            uint32 s_row = (b * S + s) * D;

            WEI_TYPE *x_tok = &input->data[s_row];
            WEI_TYPE *g_tok = &curr->grad[s_row];

            WEI_TYPE _u = 0.0f;
            for (uint32 d = 0; d < D; d++) _u += x_tok[d];
            _u /= (WEI_TYPE)D;

            WEI_TYPE _v = 0.0f;
            for (uint32 d = 0; d < D; d++) _v += (x_tok[d] - _u) * (x_tok[d] - _u);
            _v /= (WEI_TYPE)D;

            WEI_TYPE inv_std = 1.0f / sqrtf(_v + eps);

            // S_1 = sum_i dL/dx_hat_i, S_2 = sum_i (dL/dx_hat_i)(x_hat_i);
            WEI_TYPE s1 = 0.0f;
            WEI_TYPE s2 = 0.0f;
            for (uint32 d = 0; d < D; d++){
              WEI_TYPE dx_hat = g_tok[d] * gamma->data[d];
              WEI_TYPE x_hat = (x_tok[d] - _u) * inv_std;

              s1 += dx_hat;
              s2 += dx_hat * x_hat;

              gamma->grad[d] += g_tok[d] * x_hat;
              beta->grad[d]  += g_tok[d];
            }

            // dL/dx_d = inv_std * (dL/dx_hat_d - S_1/D - (x_hat_d/D) * S_2)
            for (uint32 d = 0; d < D; d++){
              WEI_TYPE dx_hat = g_tok[d] * gamma->data[d];
              WEI_TYPE x_hat = (x_tok[d] - _u) * inv_std;
              input->grad[s_row + d] += inv_std * (dx_hat - s1 / (WEI_TYPE)D
                                                   - (x_hat / (WEI_TYPE)D) * s2);
            }
          }
        }
        break;
        }
      }
    }
  }
}

