#include "autograd.h"
#include "arena.h"
#include "config.h"
#include "tensor.h"
#include "types.h"
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
                WEI_TYPE grad_out_i = curr->grad[s_batch + s_row + i];
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
      }
    }
  }
}

