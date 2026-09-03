#include "types.h"
#include "tensor.h"

WEI_TYPE mean(WEI_TYPE *arr, size_t size){
  WEI_TYPE sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += arr[i];
  }
  return sum / size;
}

WEI_TYPE var(WEI_TYPE *arr, WEI_TYPE mean, size_t size){
  WEI_TYPE sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (arr[i] - mean) * (arr[i] - mean);
  }
  return sum / size;
}
