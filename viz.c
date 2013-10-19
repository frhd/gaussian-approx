#include <stdio.h>
#include <math.h>
#include "viz.h"
#include "tracker.h"

void viz_bar(float value, float max_val, int width) {
	int i, n, center;
	if (max_val <= 0) max_val = 1.0;

	if (value >= 0) {
		center = width / 2;
		n = (int)(value / max_val * (width / 2));
		if (n > width / 2) n = width / 2;
		for (i = 0; i < center; i++) printf(" ");
		for (i = 0; i < n; i++) printf("#");
		printf("\n");
	} else {
		center = width / 2;
		n = (int)(-value / max_val * (width / 2));
		for (i = 0; i < center - n; i++) printf(" ");
		for (i = 0; i < n; i++) printf("#");
		printf("\n");
	}
}

void viz_vector(Matrix v) {
	int i;
	float max_val = 0;

	for (i = 0; i < v->height; i++) {
		float val = fabs(elem(v, i, 0));
		if (val > max_val) max_val = val;
	}
	if (max_val == 0) max_val = 1.0;

	printf("%-30s%30s\n", "", "");
	printf("%-30.2f%30.2f\n", -max_val, max_val);

	for (i = 0; i < v->height; i++) {
		printf("[%2d] ", i);
		viz_bar(elem(v, i, 0), max_val, 54);
	}
}

void viz_gaussian_1d(float mean, float sigma, int width, int height) {
	int x;
	float xmin, xmax, dx;
	float *pdf;

	xmin = mean - 3 * sigma;
	xmax = mean + 3 * sigma;
	dx = (xmax - xmin) / (width - 1);

	pdf = (float *)malloc(width * sizeof(float));
	for (x = 0; x < width; x++) {
		float xv = xmin + x * dx;
		float z = (xv - mean) / sigma;
		pdf[x] = exp(-0.5 * z * z) / (sigma * sqrt(2 * pi));
	}

	for (x = 0; x < width; x++)
		printf("%.4f ", pdf[x]);
	printf("\n");

	free(pdf);
}
