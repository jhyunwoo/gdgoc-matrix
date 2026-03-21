#include "matmul.h"
#include "utils.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZES "512,1024,2048"
#define DEFAULT_REPEAT 10
#define DEFAULT_WARMUP 3
#define DEFAULT_SEED UINT64_C(42)
#define DEFAULT_EPS 1e-6f
#define DEFAULT_REL_ERR_THRESHOLD 1e-4

typedef struct {
  int* sizes;
  int size_count;
  int repeat;
  int warmup;
  int validate;
  int threads;
  uint64_t seed;
} benchmark_options_t;

typedef struct {
  const char* name;
  matmul_fn fn;
} impl_entry_t;

static void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("\n");
  printf("Options:\n");
  printf("  --sizes <csv>        Matrix size list (square): 512,1024,2048\n");
  printf("  --repeat <n>         Timed repetitions per size (default: %d)\n", DEFAULT_REPEAT);
  printf("  --warmup <n>         Warmup repetitions per size (default: %d)\n", DEFAULT_WARMUP);
  printf("  --threads <n>        Set MATRIX_THREADS and OMP_NUM_THREADS\n");
  printf("  --seed <n>           RNG seed (default: %llu)\n", (unsigned long long)DEFAULT_SEED);
  printf("  --validate           Validate output against reference GEMM\n");
  printf("  --help               Show this help\n");
}

static char* dup_cstr(const char* s) {
  const size_t n = strlen(s);
  char* copy = (char*)malloc(n + 1u);
  if (copy == NULL) {
    return NULL;
  }
  memcpy(copy, s, n + 1u);
  return copy;
}

static char* trim_space(char* s) {
  char* end;
  while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') {
    ++s;
  }
  if (*s == '\0') {
    return s;
  }
  end = s + strlen(s) - 1;
  while (end > s && (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r')) {
    *end-- = '\0';
  }
  return s;
}

static int parse_positive_int(const char* s, int* out_value) {
  char* end = NULL;
  long value;

  errno = 0;
  value = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || value <= 0 || value > INT_MAX) {
    return 0;
  }

  *out_value = (int)value;
  return 1;
}

static int parse_seed_u64(const char* s, uint64_t* out_seed) {
  char* end = NULL;
  unsigned long long value;

  errno = 0;
  value = strtoull(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    return 0;
  }

  *out_seed = (uint64_t)value;
  return 1;
}

static int parse_sizes_csv(const char* csv, int** out_sizes, int* out_count) {
  char* work = dup_cstr(csv);
  char* saveptr = NULL;
  char* token = NULL;
  int* sizes = NULL;
  int count = 0;
  int capacity = 8;

  if (work == NULL) {
    return 0;
  }

  sizes = (int*)malloc((size_t)capacity * sizeof(int));
  if (sizes == NULL) {
    free(work);
    return 0;
  }

  token = strtok_r(work, ",", &saveptr);
  while (token != NULL) {
    char* trimmed = trim_space(token);
    int parsed = 0;
    if (!parse_positive_int(trimmed, &parsed)) {
      free(sizes);
      free(work);
      return 0;
    }

    if (count == capacity) {
      int new_capacity = capacity * 2;
      int* grown = (int*)realloc(sizes, (size_t)new_capacity * sizeof(int));
      if (grown == NULL) {
        free(sizes);
        free(work);
        return 0;
      }
      sizes = grown;
      capacity = new_capacity;
    }

    sizes[count++] = parsed;
    token = strtok_r(NULL, ",", &saveptr);
  }

  free(work);

  if (count == 0) {
    free(sizes);
    return 0;
  }

  *out_sizes = sizes;
  *out_count = count;
  return 1;
}

static int init_default_options(benchmark_options_t* opts) {
  memset(opts, 0, sizeof(*opts));
  opts->repeat = DEFAULT_REPEAT;
  opts->warmup = DEFAULT_WARMUP;
  opts->validate = 0;
  opts->threads = 0;
  opts->seed = DEFAULT_SEED;
  return parse_sizes_csv(DEFAULT_SIZES, &opts->sizes, &opts->size_count);
}

static void free_options(benchmark_options_t* opts) {
  free(opts->sizes);
  opts->sizes = NULL;
  opts->size_count = 0;
}

static void print_options(const benchmark_options_t* opts) {
  int i;
  printf("Sizes: ");
  for (i = 0; i < opts->size_count; ++i) {
    printf("%d", opts->sizes[i]);
    if (i + 1 < opts->size_count) {
      printf(",");
    }
  }
  printf("\n");
  printf("Repeat: %d, Warmup: %d, Validate: %s\n",
         opts->repeat,
         opts->warmup,
         opts->validate ? "yes" : "no");
  printf("Seed: %llu\n", (unsigned long long)opts->seed);
  if (opts->threads > 0) {
    printf("Threads override: %d\n", opts->threads);
  } else {
    printf("Threads override: auto\n");
  }
}

static uint64_t derive_state(uint64_t seed, int size) {
  uint64_t x = seed;
  x ^= UINT64_C(0xBF58476D1CE4E5B9) * (uint64_t)size;
  x ^= x >> 30;
  x *= UINT64_C(0xBF58476D1CE4E5B9);
  x ^= x >> 27;
  x *= UINT64_C(0x94D049BB133111EB);
  x ^= x >> 31;
  return x ? x : UINT64_C(1);
}

static int run_impl(const benchmark_options_t* opts, const impl_entry_t* impl) {
  double score_sum = 0.0;
  int all_valid = 1;
  int i;

  printf("\n=== Implementation: %s ===\n", impl->name);
  printf("%8s %12s %12s %14s %14s %8s %14s\n",
         "size",
         "avg_ms",
         "best_ms",
         "avg_GFLOPS",
         "best_GFLOPS",
         "valid",
         "max_rel_err");

  for (i = 0; i < opts->size_count; ++i) {
    const int M = opts->sizes[i];
    const int N = opts->sizes[i];
    const int K = opts->sizes[i];
    const size_t a_count = (size_t)M * (size_t)K;
    const size_t b_count = (size_t)K * (size_t)N;
    const size_t c_count = (size_t)M * (size_t)N;
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    float* A = aligned_alloc_f32(a_count);
    float* B = aligned_alloc_f32(b_count);
    float* C = aligned_alloc_f32(c_count);
    float* C_ref = NULL;
    uint64_t state = derive_state(opts->seed, opts->sizes[i]);
    double total_sec = 0.0;
    double best_sec = HUGE_VAL;
    double avg_sec;
    double avg_gflops;
    double best_gflops;
    double max_rel_err = 0.0;
    int r;

    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "memory allocation failed for size %d\n", opts->sizes[i]);
      free(A);
      free(B);
      free(C);
      free(C_ref);
      return 1;
    }

    fill_random_f32(A, a_count, &state);
    fill_random_f32(B, b_count, &state);

    if (opts->validate) {
      C_ref = aligned_alloc_f32(c_count);
      if (C_ref == NULL) {
        fprintf(stderr, "reference buffer allocation failed for size %d\n", opts->sizes[i]);
        free(A);
        free(B);
        free(C);
        return 1;
      }
      matmul_ref(A, B, C_ref, M, N, K);
    }

    for (r = 0; r < opts->warmup; ++r) {
      impl->fn(A, B, C, M, N, K);
    }

    for (r = 0; r < opts->repeat; ++r) {
      double t0 = wall_time_seconds();
      impl->fn(A, B, C, M, N, K);
      double t1 = wall_time_seconds();
      const double elapsed = t1 - t0;
      total_sec += elapsed;
      if (elapsed < best_sec) {
        best_sec = elapsed;
      }
    }

    avg_sec = total_sec / (double)opts->repeat;
    avg_gflops = flops / avg_sec / 1e9;
    best_gflops = flops / best_sec / 1e9;

    if (opts->validate) {
      max_rel_err = max_relative_error_f32(C, C_ref, c_count, DEFAULT_EPS);
      if (!(max_rel_err <= DEFAULT_REL_ERR_THRESHOLD)) {
        all_valid = 0;
        printf("%8d %12.3f %12.3f %14.2f %14.2f %8s %14.6e\n",
               opts->sizes[i],
               avg_sec * 1e3,
               best_sec * 1e3,
               avg_gflops,
               best_gflops,
               "FAIL",
               max_rel_err);
      } else {
        printf("%8d %12.3f %12.3f %14.2f %14.2f %8s %14.6e\n",
               opts->sizes[i],
               avg_sec * 1e3,
               best_sec * 1e3,
               avg_gflops,
               best_gflops,
               "PASS",
               max_rel_err);
      }
    } else {
      printf("%8d %12.3f %12.3f %14.2f %14.2f %8s %14s\n",
             opts->sizes[i],
             avg_sec * 1e3,
             best_sec * 1e3,
             avg_gflops,
             best_gflops,
             "-",
             "-");
    }

    score_sum += avg_gflops;

    free(A);
    free(B);
    free(C);
    free(C_ref);
  }

  printf("Score (mean avg GFLOPS): %.2f\n", score_sum / (double)opts->size_count);
  if (opts->validate) {
    printf("Validation summary: %s (threshold %.1e)\n",
           all_valid ? "PASS" : "FAIL",
           DEFAULT_REL_ERR_THRESHOLD);
  }

  return all_valid ? 0 : 2;
}

int main(int argc, char** argv) {
  benchmark_options_t opts;
  impl_entry_t submission_impl;
  int i;

  if (!init_default_options(&opts)) {
    fprintf(stderr, "failed to initialize benchmark options\n");
    return 1;
  }

  for (i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0) {
      print_usage(argv[0]);
      free_options(&opts);
      return 0;
    }

    if (strcmp(argv[i], "--validate") == 0) {
      opts.validate = 1;
      continue;
    }

    if (strcmp(argv[i], "--sizes") == 0) {
      int* parsed_sizes = NULL;
      int parsed_count = 0;
      if (i + 1 >= argc) {
        fprintf(stderr, "--sizes requires a value\n");
        free_options(&opts);
        return 1;
      }
      if (!parse_sizes_csv(argv[++i], &parsed_sizes, &parsed_count)) {
        fprintf(stderr, "invalid --sizes value: %s\n", argv[i]);
        free_options(&opts);
        return 1;
      }
      free(opts.sizes);
      opts.sizes = parsed_sizes;
      opts.size_count = parsed_count;
      continue;
    }

    if (strcmp(argv[i], "--repeat") == 0) {
      if (i + 1 >= argc || !parse_positive_int(argv[++i], &opts.repeat)) {
        fprintf(stderr, "invalid --repeat value\n");
        free_options(&opts);
        return 1;
      }
      continue;
    }

    if (strcmp(argv[i], "--warmup") == 0) {
      if (i + 1 >= argc || !parse_positive_int(argv[++i], &opts.warmup)) {
        fprintf(stderr, "invalid --warmup value\n");
        free_options(&opts);
        return 1;
      }
      continue;
    }

    if (strcmp(argv[i], "--threads") == 0) {
      if (i + 1 >= argc || !parse_positive_int(argv[++i], &opts.threads)) {
        fprintf(stderr, "invalid --threads value\n");
        free_options(&opts);
        return 1;
      }
      continue;
    }

    if (strcmp(argv[i], "--seed") == 0) {
      if (i + 1 >= argc || !parse_seed_u64(argv[++i], &opts.seed)) {
        fprintf(stderr, "invalid --seed value\n");
        free_options(&opts);
        return 1;
      }
      continue;
    }

    fprintf(stderr, "unknown option: %s\n", argv[i]);
    print_usage(argv[0]);
    free_options(&opts);
    return 1;
  }

  if (opts.threads > 0) {
    char thread_buf[32];
    snprintf(thread_buf, sizeof(thread_buf), "%d", opts.threads);
    setenv("MATRIX_THREADS", thread_buf, 1);
    setenv("OMP_NUM_THREADS", thread_buf, 1);
  }

  printf("=== Matrix Multiplication Benchmark (FP32, No Accelerate) ===\n");
  printf("Compiler: %s\n", __VERSION__);
  print_options(&opts);

  submission_impl.name = "submission";
  submission_impl.fn = matmul;

  i = run_impl(&opts, &submission_impl);
  free_options(&opts);
  return i;
}
