#ifndef MATMUL_H
#define MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*matmul_fn)(const float* A, const float* B, float* C, int M, int N, int K);

/* Reference implementation used for correctness validation. */
void matmul_ref(const float* A, const float* B, float* C, int M, int N, int K);

/* Participant submission entry point. */
void matmul(const float* A, const float* B, float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_H */
