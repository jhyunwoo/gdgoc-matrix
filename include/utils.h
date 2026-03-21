#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

double wall_time_seconds(void);
float* aligned_alloc_f32(size_t count);
void fill_random_f32(float* data, size_t count, uint64_t* state);
double max_relative_error_f32(const float* actual, const float* reference, size_t count, float eps);

#endif /* UTILS_H */
