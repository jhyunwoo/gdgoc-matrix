#include "matmul.h"

#include <arm_neon.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/sysctl.h>

typedef struct {
  const float* A;
  const float* B;
  float* C;
  int N;
  int K;
  int row_start;
  int row_end;
} worker_ctx_t;

static int parse_env_threads(const char* key) {
  const char* value = getenv(key);
  if (value != NULL && value[0] != '\0') {
    char* end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0 && parsed <= 256) {
      return (int)parsed;
    }
  }
  return 0;
}

static int default_thread_count(void) {
  static int cached = 0;
  int logical = 0;
  size_t sz = sizeof(logical);

  if (cached > 0) {
    return cached;
  }

  if (sysctlbyname("hw.logicalcpu", &logical, &sz, NULL, 0) == 0 && logical > 0) {
    cached = logical;
  } else {
    cached = 1;
  }

  return cached;
}

static int choose_thread_count(int M, int N, int K) {
  int threads = parse_env_threads("MATRIX_THREADS");
  if (threads <= 0) {
    threads = parse_env_threads("OMP_NUM_THREADS");
  }
  if (threads <= 0) {
    threads = default_thread_count();
  }
  if (threads > M) {
    threads = M;
  }
  if (threads < 1) {
    threads = 1;
  }

  if ((int64_t)M * (int64_t)N * (int64_t)K < 2000000LL) {
    threads = 1;
  }

  return threads;
}

static void compute_rows(const worker_ctx_t* ctx) {
  const float* A = ctx->A;
  const float* B = ctx->B;
  float* C = ctx->C;
  const int N = ctx->N;
  const int K = ctx->K;
  int i;

  for (i = ctx->row_start; i < ctx->row_end; ++i) {
    const float* a_row = A + (size_t)i * (size_t)K;
    float* c_row = C + (size_t)i * (size_t)N;
    int j = 0;

    for (; j + 8 <= N; j += 8) {
      float32x4_t c0 = vdupq_n_f32(0.0f);
      float32x4_t c1 = vdupq_n_f32(0.0f);
      int k;

      for (k = 0; k < K; ++k) {
        const float a = a_row[k];
        const float* b_ptr = B + (size_t)k * (size_t)N + (size_t)j;
        c0 = vfmaq_n_f32(c0, vld1q_f32(b_ptr), a);
        c1 = vfmaq_n_f32(c1, vld1q_f32(b_ptr + 4), a);
      }

      vst1q_f32(c_row + j, c0);
      vst1q_f32(c_row + j + 4, c1);
    }

    for (; j + 4 <= N; j += 4) {
      float32x4_t c0 = vdupq_n_f32(0.0f);
      int k;

      for (k = 0; k < K; ++k) {
        const float a = a_row[k];
        const float* b_ptr = B + (size_t)k * (size_t)N + (size_t)j;
        c0 = vfmaq_n_f32(c0, vld1q_f32(b_ptr), a);
      }

      vst1q_f32(c_row + j, c0);
    }

    for (; j < N; ++j) {
      float acc = 0.0f;
      int k;

      for (k = 0; k < K; ++k) {
        acc += a_row[k] * B[(size_t)k * (size_t)N + (size_t)j];
      }

      c_row[j] = acc;
    }
  }
}

static void* worker_main(void* arg) {
  const worker_ctx_t* ctx = (const worker_ctx_t*)arg;
  compute_rows(ctx);
  return NULL;
}

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
  int threads;
  int t;
  int row = 0;
  worker_ctx_t* contexts;
  pthread_t* tids;
  unsigned char* started;

  if (A == NULL || B == NULL || C == NULL || M <= 0 || N <= 0 || K <= 0) {
    return;
  }

  threads = choose_thread_count(M, N, K);
  if (threads <= 1) {
    worker_ctx_t ctx;
    ctx.A = A;
    ctx.B = B;
    ctx.C = C;
    ctx.N = N;
    ctx.K = K;
    ctx.row_start = 0;
    ctx.row_end = M;
    compute_rows(&ctx);
    return;
  }

  contexts = (worker_ctx_t*)malloc((size_t)threads * sizeof(worker_ctx_t));
  tids = (pthread_t*)malloc((size_t)threads * sizeof(pthread_t));
  started = (unsigned char*)calloc((size_t)threads, sizeof(unsigned char));

  if (contexts == NULL || tids == NULL || started == NULL) {
    worker_ctx_t ctx;
    free(contexts);
    free(tids);
    free(started);

    ctx.A = A;
    ctx.B = B;
    ctx.C = C;
    ctx.N = N;
    ctx.K = K;
    ctx.row_start = 0;
    ctx.row_end = M;
    compute_rows(&ctx);
    return;
  }

  for (t = 0; t < threads; ++t) {
    const int rows_for_t = M / threads + (t < (M % threads) ? 1 : 0);

    contexts[t].A = A;
    contexts[t].B = B;
    contexts[t].C = C;
    contexts[t].N = N;
    contexts[t].K = K;
    contexts[t].row_start = row;
    contexts[t].row_end = row + rows_for_t;

    row += rows_for_t;
  }

  for (t = 1; t < threads; ++t) {
    if (contexts[t].row_start < contexts[t].row_end) {
      if (pthread_create(&tids[t], NULL, worker_main, &contexts[t]) == 0) {
        started[t] = 1;
      } else {
        compute_rows(&contexts[t]);
      }
    }
  }

  compute_rows(&contexts[0]);

  for (t = 1; t < threads; ++t) {
    if (started[t]) {
      pthread_join(tids[t], NULL);
    }
  }

  free(contexts);
  free(tids);
  free(started);
}
