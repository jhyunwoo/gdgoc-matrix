#include "utils.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

double wall_time_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

float* aligned_alloc_f32(size_t count) {
  void* ptr = NULL;
  const size_t bytes = count * sizeof(float);
  if (bytes == 0) {
    return NULL;
  }
  if (posix_memalign(&ptr, 64u, bytes) != 0) {
    return NULL;
  }
  return (float*)ptr;
}

static uint64_t xorshift64star(uint64_t* state) {
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * UINT64_C(2685821657736338717);
}

void fill_random_f32(float* data, size_t count, uint64_t* state) {
  size_t i;
  for (i = 0; i < count; ++i) {
    const uint64_t r = xorshift64star(state);
    const float normalized = (float)((double)(r >> 11) * (1.0 / 9007199254740992.0));
    data[i] = normalized * 2.0f - 1.0f;
  }
}

double max_relative_error_f32(const float* actual, const float* reference, size_t count, float eps) {
  size_t i;
  double max_err = 0.0;

  for (i = 0; i < count; ++i) {
    const double a = (double)actual[i];
    const double r = (double)reference[i];
    const double denom = fmax(fabs(r), (double)eps);
    const double err = fabs(a - r) / denom;
    if (err > max_err) {
      max_err = err;
    }
  }

  return max_err;
}
