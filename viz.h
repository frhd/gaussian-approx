#ifndef viz_h
#define viz_h

#include "matrix.h"

void viz_bar(float value, float max_val, int width);
void viz_vector(Matrix v);
void viz_gaussian_1d(float mean, float sigma, int width, int height);

#endif
