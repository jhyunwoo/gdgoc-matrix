#include "matmul.h"

#include <stddef.h>

void matmul_ref(const float* A, const float* B, float* C, int M, int N, int K) {
  int i;
  int j;
  int k;

  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (k = 0; k < K; ++k) {
        acc += A[(size_t)i * (size_t)K + (size_t)k] *
               B[(size_t)k * (size_t)N + (size_t)j];
      }
      C[(size_t)i * (size_t)N + (size_t)j] = acc;
    }
  }
}
